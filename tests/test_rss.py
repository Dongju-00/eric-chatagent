"""RSS 상시 코퍼스 수집.

RSS는 키워드 질의가 불가능하므로 네이버 검색을 대체하지 않는다. 종목
메타데이터를 붙여 미리 색인해 두고, 키워드 검색이 0건이거나 오래됐을 때
보충원으로 쓴다. 피드 하나가 죽어도 수집 전체와 요청 경로가 살아야 한다.
"""

import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.news.rss import FEEDS, fetch_all_feeds, fetch_feed, tag_companies

COMPANIES_FIXTURE = [
    {"canonical_name": "삼성전자", "ticker": "005930",
     "aliases": ["삼성전자", "005930", "삼전"]},
    {"canonical_name": "GS", "ticker": "078930", "aliases": ["GS", "078930"]},
]

_FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <title>삼성전자 3분기 실적 발표</title>
    <link>https://example.com/1</link>
    <description>영업이익이 늘었다</description>
    <pubDate>Thu, 23 Oct 2026 09:00:00 +0900</pubDate>
  </item>
  <item>
    <title>코스피 강보합 마감</title>
    <link>https://example.com/2</link>
    <description>지수가 소폭 올랐다</description>
    <pubDate>Thu, 23 Oct 2026 16:00:00 +0900</pubDate>
  </item>
</channel></rss>""".encode("utf-8")


class FeedListTests(unittest.TestCase):
    def test_feeds_are_nonempty_https_urls(self):
        self.assertTrue(FEEDS)
        for url in FEEDS:
            self.assertTrue(url.startswith("https://"), url)


class FetchFeedTests(unittest.TestCase):
    def test_dead_feed_returns_empty_without_raising(self):
        """피드 하나가 403/404/타임아웃이어도 예외를 던지지 않는다."""
        for exc in [
            urllib.error.HTTPError("u", 403, "Forbidden", {}, None),
            urllib.error.HTTPError("u", 404, "Not Found", {}, None),
            urllib.error.URLError("timed out"),
        ]:
            with self.subTest(exc=type(exc).__name__):
                with mock.patch("urllib.request.urlopen", side_effect=exc):
                    self.assertEqual(fetch_feed("https://example.com/rss"), [])

    def test_malformed_xml_returns_empty_without_raising(self):
        resp = mock.MagicMock()
        resp.read.return_value = b"<not-xml"
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=resp):
            self.assertEqual(fetch_feed("https://example.com/rss"), [])

    def test_parses_items_into_naver_like_dicts(self):
        resp = mock.MagicMock()
        resp.read.return_value = _FEED_XML
        resp.__enter__ = mock.Mock(return_value=resp)
        resp.__exit__ = mock.Mock(return_value=False)
        with mock.patch("urllib.request.urlopen", return_value=resp):
            items = fetch_feed("https://example.com/rss")

        self.assertEqual(len(items), 2)
        first = items[0]
        # collect.extract_and_chunk 가 쓰는 키와 같은 모양이어야 한다
        for key in ("title", "description", "pubDate", "originallink", "link"):
            self.assertIn(key, first)
        self.assertEqual(first["title"], "삼성전자 3분기 실적 발표")
        self.assertEqual(first["originallink"], "https://example.com/1")


class TagCompaniesTests(unittest.TestCase):
    def test_assigns_ticker_from_title(self):
        items = [{"title": "삼성전자 신기록", "description": "", "pubDate": "x"}]
        out = tag_companies(items, COMPANIES_FIXTURE)
        self.assertEqual(out[0]["metadata"]["ticker"], "005930")
        self.assertEqual(out[0]["metadata"]["company"], "삼성전자")

    def test_untagged_items_are_dropped(self):
        """종목을 못 붙인 기사는 코퍼스에 넣지 않는다.

        상시 코퍼스는 종목 메타데이터로 1차 필터한 뒤 임베딩 검색을 하는
        용도다. 종목이 없는 기사는 그 필터에 걸리지 않아 쓸모가 없고,
        코퍼스만 키운다.
        """
        items = [{"title": "코스피 강보합 마감", "description": "", "pubDate": "x"}]
        self.assertEqual(tag_companies(items, COMPANIES_FIXTURE), [])

    def test_ascii_alias_requires_word_boundary(self):
        """종목 식별과 같은 경계 규칙을 쓴다. "earnings" 가 GS로 잡히면 안 된다."""
        items = [{"title": "earnings season 전망", "description": "", "pubDate": "x"}]
        self.assertEqual(tag_companies(items, COMPANIES_FIXTURE), [])


class DeduplicationTests(unittest.TestCase):
    def test_same_article_in_two_feeds_is_collected_once(self):
        """연합뉴스 market/economy/industry 는 같은 기사를 공유한다.

        중복을 남기면 corpus id(link 해시 기반)가 한 배치 안에서 충돌해
        Chroma upsert 가 DuplicateIDError 로 터진다. 실제 수집에서 410건 중
        31건이 충돌했다.
        """
        dup = {"title": "삼성전자 실적", "description": "", "pubDate": "x",
               "originallink": "https://example.com/same", "link": "https://example.com/same"}
        other = {**dup, "title": "기아 실적",
                 "originallink": "https://example.com/other",
                 "link": "https://example.com/other"}
        with mock.patch("model.news.rss.fetch_feed", side_effect=[[dup], [dup, other]]):
            out = fetch_all_feeds(["https://a/rss", "https://b/rss"])

        links = [i["link"] for i in out]
        self.assertEqual(sorted(links),
                         ["https://example.com/other", "https://example.com/same"])


class CorpusIndexDuplicateIdTests(unittest.TestCase):
    def test_duplicate_ids_in_one_batch_do_not_raise(self):
        """호출자가 중복을 넘겨도 색인이 터지지 않아야 한다."""
        import tempfile
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _fakes import FakeEmbedder
        from model.store.vector_store import ChromaNewsVectorStore

        with mock.patch("model.store.vector_store.KoreanGPTEmbedder", FakeEmbedder):
            store = ChromaNewsVectorStore(persist_path=tempfile.mkdtemp())

        same = {"text": "제목: 같은 기사 본문",
                "metadata": {"title": "같은 기사", "chunk_id": 0,
                             "link": "https://example.com/same",
                             "company": "삼성전자", "ticker": "005930"}}
        saved = store.index_corpus_chunks([same, dict(same)])
        self.assertEqual(saved, 1)
        self.assertEqual(store.news_collection.count(), 1)


if __name__ == "__main__":
    unittest.main()
