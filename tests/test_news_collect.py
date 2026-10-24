"""뉴스 수집(검색·필터·본문추출·청킹)이 저장/검색과 분리됐는지 검증한다.

`select_articles` 는 네트워크를 타지 않는 순수 함수여야 한다. Task 3 의
종목 필터 테스트가 그 경계 위에 올라간다.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.news.collect import extract_and_chunk, fetch_candidates, select_articles


def _pub(days_ago: int = 0) -> str:
    """네이버 pubDate 와 같은 RFC 822 형식 문자열."""
    return format_datetime(datetime.now(timezone.utc) - timedelta(days=days_ago))


RECENT = _pub(0)


ARTICLE_FIXTURE = {
    "title": "삼성전자 3분기 실적 발표",
    "description": "영업이익이 전년 대비 늘었다",
    "pubDate": RECENT,
    "originallink": "https://example.com/a",
    "link": "https://n.news.naver.com/a",
}


class SelectArticlesTests(unittest.TestCase):
    def test_select_articles_is_pure_and_returns_at_most_limit(self):
        items = [
            {"title": "삼성전자 신기록", "description": "", "pubDate": RECENT},
            {"title": "무관 기사", "description": "", "pubDate": RECENT},
        ]
        out = select_articles(items, "삼성전자", ["삼성전자", "삼전"], limit=1)
        self.assertEqual(len(out), 1)
        self.assertIn("삼성전자", out[0]["title"])

    def test_select_articles_makes_no_network_call(self):
        """순수 함수 경계. urlopen 이 불리면 경계가 깨진 것이다."""
        items = [{"title": "삼성전자 신기록", "description": "", "pubDate": RECENT}]
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("네트워크 호출")):
            select_articles(items, "삼성전자", ["삼성전자"], limit=1)

    def test_old_articles_are_dropped(self):
        items = [
            {"title": "삼성전자 최신", "description": "", "pubDate": _pub(0)},
            {"title": "삼성전자 작년", "description": "", "pubDate": _pub(400)},
        ]
        out = select_articles(items, "삼성전자", ["삼성전자"], limit=5)
        self.assertEqual([i["title"] for i in out], ["삼성전자 최신"])


class ChunkTests(unittest.TestCase):
    def test_every_chunk_carries_title_and_date_prefix(self):
        with mock.patch(
            "model.news.collect.extract_news_body",
            # (본문, 링크, content_type) 3-튜플을 반환한다
            return_value=("본문 " * 400, "https://example.com/a", "article_body"),
        ):
            chunks = extract_and_chunk([ARTICLE_FIXTURE])
        self.assertGreater(len(chunks), 1, "본문이 길면 여러 청크로 나뉘어야 한다")
        for c in chunks:
            self.assertTrue(c["text"].startswith("제목: "), c["text"][:40])
            self.assertIn("작성일: ", c["text"])
            self.assertIn("내용: ", c["text"])

    def test_chunk_metadata_carries_chunk_id(self):
        with mock.patch(
            "model.news.collect.extract_news_body",
            # (본문, 링크, content_type) 3-튜플을 반환한다
            return_value=("본문 " * 400, "https://example.com/a", "article_body"),
        ):
            chunks = extract_and_chunk([ARTICLE_FIXTURE])
        self.assertEqual([c["metadata"]["chunk_id"] for c in chunks],
                         list(range(len(chunks))))


class MetadataPreservationTests(unittest.TestCase):
    def test_incoming_metadata_survives_chunking(self):
        """호출자가 붙인 메타데이터를 청킹이 버리면 안 된다.

        RSS 경로는 tag_companies 가 company/ticker 를 붙인 뒤
        extract_and_chunk 를 거친다. 여기서 메타데이터를 새로 만들면
        종목 태그가 사라져 코퍼스 필터가 영구히 0건이 된다(실측: 156청크
        전부 company="").
        """
        tagged = {**ARTICLE_FIXTURE,
                  "metadata": {"company": "삼성전자", "ticker": "005930"}}
        with mock.patch(
            "model.news.collect.extract_news_body",
            return_value=("본문 " * 400, "https://example.com/a", "article_body"),
        ):
            chunks = extract_and_chunk([tagged])

        self.assertTrue(chunks)
        for c in chunks:
            self.assertEqual(c["metadata"]["company"], "삼성전자")
            self.assertEqual(c["metadata"]["ticker"], "005930")
            # 청킹이 만드는 값도 함께 있어야 한다
            self.assertEqual(c["metadata"]["title"], ARTICLE_FIXTURE["title"])


class FetchCandidatesTests(unittest.TestCase):
    def test_fetch_candidates_delegates_to_search_and_returns_items(self):
        with mock.patch(
            "model.news.collect.search_stock_news",
            return_value={"items": [ARTICLE_FIXTURE]},
        ) as search:
            out = fetch_candidates("삼성전자 주가", display=9, sort="sim")
        search.assert_called_once_with(query="삼성전자 주가", display=9, sort="sim")
        self.assertEqual(out, [ARTICLE_FIXTURE])


class StoreNoLongerCollectsTests(unittest.TestCase):
    def test_vector_store_has_no_collection_entrypoint(self):
        """수집은 collect.py 책임이다. 저장소에 남아 있으면 분리가 안 된 것이다."""
        from model.store.vector_store import ChromaNewsVectorStore

        self.assertFalse(hasattr(ChromaNewsVectorStore, "build_news_vector_db"))
        self.assertTrue(hasattr(ChromaNewsVectorStore, "index_chunks"))


if __name__ == "__main__":
    unittest.main()
