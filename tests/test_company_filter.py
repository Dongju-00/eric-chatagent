"""종목 필터의 fail-open 경로와 영문 별칭 오탐을 막는다.

실측 배경:
- 사전에 없는 종목(카카오)은 `company=None` 을 만들고, 그러면 필터가 조용히
  꺼져 "대전 대덕구, 체납 1억 7000만 원 징수" 가 수집됐다.
- 삼성전자 질문에서 후보 9건 중 8건이 제목 불일치/본문 일치였고 실제 주제는
  SK하이닉스·삼성E&A였다.
- 사전을 30종에서 299종으로 늘리면서 `GS`·`LS`·`NC`·`CU` 같은 짧은 영문
  별칭이 일반 단어에 걸리게 됐다("earnings" -> GS).
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.graph.query_rewrite import rewrite_stock_query
from model.news.collect import select_articles


def _pub(days_ago: int = 0) -> str:
    return format_datetime(datetime.now(timezone.utc) - timedelta(days=days_ago))


RECENT = _pub(0)


class FilterFailOpenTests(unittest.TestCase):
    def test_unknown_company_does_not_disable_filter(self):
        """company=None 일 때 무관한 기사를 통과시키면 안 된다."""
        items = [{"title": "대전 대덕구, 체납 1억 7000만 원 징수",
                  "description": "", "pubDate": RECENT}]
        out = select_articles(items, None, [], limit=3, question="카카오 실적 어때?")
        self.assertEqual(out, [])

    def test_unknown_company_still_keeps_loosely_matching_article(self):
        """미식별이어도 질문 토큰이 제목에 있으면 살린다. 필터를 끄는 것과 다르다."""
        items = [{"title": "카카오 3분기 실적 발표", "description": "", "pubDate": RECENT}]
        out = select_articles(items, None, [], limit=3, question="카카오 실적 어때?")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["metadata"]["match_kind"], "loose")
        self.assertFalse(out[0]["metadata"]["company_resolved"])

    def test_title_match_ranks_above_description_only_match(self):
        title_hit = {"title": "삼성전자 3분기 실적", "description": "", "pubDate": RECENT}
        desc_only = {"title": "SK하이닉스 급반등", "description": "삼성전자도 언급",
                     "pubDate": RECENT}
        out = select_articles([desc_only, title_hit], "삼성전자", ["삼성전자"], limit=1)
        self.assertEqual(out[0]["title"], "삼성전자 3분기 실적")
        self.assertEqual(out[0]["metadata"]["match_kind"], "title")

    def test_description_only_match_is_used_when_titles_run_out(self):
        """제목 일치가 부족하면 본문 일치로 채운다. 버리는 게 아니다."""
        desc_only = {"title": "SK하이닉스 급반등", "description": "삼성전자도 언급",
                     "pubDate": RECENT}
        out = select_articles([desc_only], "삼성전자", ["삼성전자"], limit=3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["metadata"]["match_kind"], "description")

    def test_empty_result_is_returned_as_empty(self):
        """필터 공백 시 원본으로 되돌아가면 안 된다."""
        items = [{"title": "무관", "description": "무관", "pubDate": RECENT}]
        self.assertEqual(select_articles(items, "삼성전자", ["삼성전자"], limit=3), [])

    def test_company_resolved_flag_is_recorded(self):
        items = [{"title": "삼성전자 신기록", "description": "", "pubDate": RECENT}]
        out = select_articles(items, "삼성전자", ["삼성전자"], limit=1)
        self.assertTrue(out[0]["metadata"]["company_resolved"])


class AsciiAliasFalsePositiveTests(unittest.TestCase):
    """짧은 영문 별칭이 일반 단어의 부분문자열로 걸리면 안 된다.

    사전 확장(30->299종)으로 GS/LS/NC/CU/ISC가 들어와 경로가 늘었다.
    잘못된 종목으로 필터링하는 것은 필터가 꺼지는 것보다 나쁠 수 있다.
    """

    def test_ascii_alias_needs_word_boundary(self):
        for question, bad in [
            ("earnings 전망 어때", "GS"),
            ("risk 관리 어떻게", "SK"),
        ]:
            with self.subTest(question=question):
                self.assertNotEqual(rewrite_stock_query(question)["company"], bad)

    def test_standalone_ascii_alias_still_resolves(self):
        """경계가 맞으면 영문 별칭은 계속 동작해야 한다."""
        self.assertEqual(rewrite_stock_query("GS 주가 어때")["company"], "GS")

    def test_korean_alias_is_unaffected_by_boundary_rule(self):
        self.assertEqual(rewrite_stock_query("삼전 주가 전망")["company"], "삼성전자")
        self.assertEqual(rewrite_stock_query("카카오 실적 어때?")["company"], "카카오")


class EmptyContextTests(unittest.TestCase):
    def test_generate_node_returns_fixed_message_without_calling_llm(self):
        """컨텍스트 0건이면 모델을 호출하지 않는다."""
        import model.graph.graph as graph_module
        from _fakes import fake_resources, no_network_collect

        resources = fake_resources(documents=())  # 검색 결과 0건
        rewritten = {
            "company": "삼성전자", "ticker": "005930", "company_aliases": ["삼전"],
            "rewritten_query": "삼성전자 주가", "sort": "sim", "matched_intents": [],
        }
        with mock.patch.object(graph_module, "classify", return_value="stock_rag"), \
             mock.patch.object(graph_module, "rewrite_stock_query", return_value=rewritten), \
             no_network_collect():
            graph = graph_module.build_graph(resources)
            result = graph.invoke(
                {"question": "삼성전자 주가 알려줘"},
                {"configurable": {"thread_id": "empty-context"}},
            )

        self.assertEqual(result["answer"],
                         "질문과 관련성이 충분한 뉴스를 \n찾지 못했습니다.")
        self.assertEqual(resources.news_llm.prompts, [],
                         "컨텍스트가 없으면 뉴스 모델을 호출하지 않아야 한다")


if __name__ == "__main__":
    unittest.main()
