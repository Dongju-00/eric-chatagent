"""질문별 청크 TTL과 검색 범위 확장.

질문별 청크는 자기 `question_id` 로만 조회되므로 그 질문이 끝나면 다시
읽히지 않는다. 실측에서 질문 53개가 청크 2,146개를 쌓았고 삭제 로직이
없었다. 재사용 가치가 있는 최신 코퍼스는 RSS 쪽에 상시로 두고, 질문별
청크는 TTL로 버린다.

`scope` 로 두 종류를 구분한다. `source` 는 이미 발행처(`naver_news`)
의미로 쓰이고 있어 재사용하지 않는다.
"""

import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fakes import FakeEmbedder


def _store():
    """실제 Chroma를 임시 디렉터리에 띄우고 임베더만 가짜로 바꾼다."""
    from model.store.vector_store import ChromaNewsVectorStore

    with mock.patch("model.store.vector_store.KoreanGPTEmbedder", FakeEmbedder):
        return ChromaNewsVectorStore(persist_path=tempfile.mkdtemp())


def _chunk(text, **meta):
    return {"text": text, "metadata": {"title": text[:10], "chunk_id": 0, **meta}}


class ScopeMetadataTests(unittest.TestCase):
    def test_question_chunks_are_scoped_question(self):
        store = _store()
        store.index_chunks("q1", [_chunk("제목: 가 본문")])
        meta = store.news_collection.get(include=["metadatas"])["metadatas"][0]
        self.assertEqual(meta["scope"], "question")
        # source 는 발행처 의미를 유지한다
        self.assertEqual(meta["source"], "naver_news")

    def test_corpus_chunks_are_scoped_corpus(self):
        store = _store()
        store.index_corpus_chunks([_chunk("제목: 나 본문", company="삼성전자",
                                          ticker="005930", link="https://a")])
        meta = store.news_collection.get(include=["metadatas"])["metadatas"][0]
        self.assertEqual(meta["scope"], "corpus")
        self.assertEqual(meta["company"], "삼성전자")


class TTLTests(unittest.TestCase):
    def test_purge_removes_only_expired_question_chunks(self):
        store = _store()
        store.index_chunks("fresh", [_chunk("제목: 신선 본문")])
        store.index_chunks("stale", [_chunk("제목: 오래됨 본문")])
        store.index_corpus_chunks([_chunk("제목: 코퍼스 본문", company="삼성전자",
                                          ticker="005930", link="https://a")])

        # stale 의 색인 시각을 TTL 밖으로 되돌린다
        got = store.news_collection.get(where={"question_id": {"$eq": "stale"}},
                                        include=["metadatas"])
        old = (datetime.now() - timedelta(hours=100)).isoformat()
        store.news_collection.update(
            ids=got["ids"],
            metadatas=[{**m, "indexed_at": old} for m in got["metadatas"]],
        )

        before_questions = store.question_collection.count()
        removed = store.purge_expired_question_chunks(ttl_hours=24)

        self.assertEqual(removed, 1)
        remaining = store.news_collection.get(include=["metadatas"])["metadatas"]
        scopes = sorted(m["scope"] for m in remaining)
        self.assertEqual(scopes, ["corpus", "question"])
        self.assertNotIn("stale", [m.get("question_id") for m in remaining])
        # user_questions 는 TTL 대상이 아니다
        self.assertEqual(store.question_collection.count(), before_questions)

    def test_purge_is_noop_when_nothing_expired(self):
        store = _store()
        store.index_chunks("fresh", [_chunk("제목: 신선 본문")])
        self.assertEqual(store.purge_expired_question_chunks(ttl_hours=24), 0)
        self.assertEqual(store.news_collection.count(), 1)

    def test_legacy_chunk_with_old_created_at_is_purged(self):
        """scope/indexed_at 이 없는 기존 청크가 삭제에서 빠지면 안 된다.

        backfill 이 created_at 을 indexed_at 으로 승계하므로, 오래된
        created_at 을 가진 레거시 청크는 만료로 판정돼야 한다.
        """
        store = _store()
        store.index_chunks("legacy", [_chunk("제목: 레거시 본문")])
        got = store.news_collection.get(include=["metadatas"])
        store.news_collection.update(ids=got["ids"], metadatas=[{
            # Chroma update 는 병합이다. 키를 지우려면 None 을 대입한다.
            "scope": None,
            "indexed_at": None,
            "created_at": (datetime.now() - timedelta(hours=100)).isoformat(),
        }])

        self.assertEqual(store.purge_expired_question_chunks(ttl_hours=24), 1)

    def test_legacy_chunk_with_no_timestamp_is_purged(self):
        """시각을 전혀 모르는 청크는 만료로 본다. 영구히 남으면 안 된다."""
        store = _store()
        store.index_chunks("legacy", [_chunk("제목: 레거시 본문")])
        got = store.news_collection.get(include=["metadatas"])
        store.news_collection.update(ids=got["ids"], metadatas=[
            {"scope": None, "indexed_at": None, "created_at": None},
        ])

        self.assertEqual(store.purge_expired_question_chunks(ttl_hours=24), 1)

    def test_legacy_chunk_with_recent_created_at_survives(self):
        """레거시라도 최근 것이면 지우지 않는다. TTL 의미를 지킨다."""
        store = _store()
        store.index_chunks("legacy", [_chunk("제목: 레거시 본문")])
        got = store.news_collection.get(include=["metadatas"])
        store.news_collection.update(ids=got["ids"], metadatas=[
            {"scope": None, "indexed_at": None},
        ])

        self.assertEqual(store.purge_expired_question_chunks(ttl_hours=24), 0)


class SearchScopeTests(unittest.TestCase):
    def test_corpus_supplements_when_question_chunks_are_empty(self):
        store = _store()
        store.index_corpus_chunks([_chunk("제목: 삼성전자 코퍼스 본문", company="삼성전자",
                                          ticker="005930", link="https://a")])
        res = store.search_similar_news(
            question="삼성전자 주가", question_id="no-such-question",
            top_k=2, max_distance=None, company="삼성전자",
        )
        self.assertEqual(len(res["documents"][0]), 1)

    def test_corpus_is_filtered_by_company(self):
        store = _store()
        store.index_corpus_chunks([_chunk("제목: 기아 코퍼스 본문", company="기아",
                                          ticker="000270", link="https://a")])
        res = store.search_similar_news(
            question="삼성전자 주가", question_id="no-such-question",
            top_k=2, max_distance=None, company="삼성전자",
        )
        self.assertEqual(res["documents"][0], [])

    def test_question_chunks_are_still_found(self):
        store = _store()
        store.index_chunks("q1", [_chunk("제목: 질문분 본문")])
        res = store.search_similar_news(
            question="아무 질문", question_id="q1", top_k=2, max_distance=None,
        )
        self.assertEqual(len(res["documents"][0]), 1)


if __name__ == "__main__":
    unittest.main()
