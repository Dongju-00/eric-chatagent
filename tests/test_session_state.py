"""세션 상태 보존 정책.

`app.py` 가 `thread_id` 를 받고, 같은 값을 재사용하면 `MemorySaver` 가
이전 상태를 이어받는다. 두 가지가 문제였다.

1. `trace` 에만 `operator.add` 리듀서가 붙어 턴마다 누적됐다. 실측: 같은
   thread_id 로 3회 호출하면 길이가 2 -> 4 -> 6. `app.py` 가 이를 응답에
   담으므로 세션이 길어질수록 응답이 계속 커졌고 상한이 없었다.
2. 리듀서 없는 필드는 그 턴의 노드가 쓰지 않으면 체크포인트 값이 살아남아,
   스몰톡 답변에 직전 주식 턴의 뉴스 컨텍스트가 근거처럼 따라붙었다.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model.graph.graph as graph_module
from _fakes import fake_resources, no_network_collect
from model.graph.graph import trim_history
from model.paths import SP_PREFIX
from model.train.sp_tokenizer import load_sp

RAG_REWRITE = {
    "company": "삼성전자", "ticker": "005930", "company_aliases": ["삼전"],
    "rewritten_query": "삼성전자 주가", "sort": "sim", "matched_intents": [],
}


def _tokens(text: str) -> int:
    return len(load_sp(SP_PREFIX).encode(text, out_type=int))


class TraceResetTests(unittest.TestCase):
    def test_trace_does_not_accumulate_across_turns(self):
        """같은 thread_id 로 3회 호출해도 trace 가 늘지 않는다. 기준선은 2->4->6."""
        resources = fake_resources()
        cfg = {"configurable": {"thread_id": "trace-reset"}}
        with mock.patch.object(graph_module, "classify", return_value="fallback"):
            graph = graph_module.build_graph(resources)
            lengths = [
                len(graph.invoke({"question": f"질문 {i}"}, cfg)["trace"])
                for i in range(3)
            ]
        self.assertEqual(lengths, [lengths[0]] * 3, lengths)

    def test_trace_still_records_the_path_within_one_turn(self):
        """턴 안에서는 거친 노드가 순서대로 남아야 한다. 리셋이 기록을 지우면 안 된다."""
        resources = fake_resources()
        with mock.patch.object(graph_module, "classify", return_value="stock_rag"), \
             mock.patch.object(graph_module, "rewrite_stock_query", return_value=RAG_REWRITE), \
             no_network_collect():
            graph = graph_module.build_graph(resources)
            result = graph.invoke(
                {"question": "삼성전자 주가"},
                {"configurable": {"thread_id": "trace-path"}},
            )
        self.assertEqual(result["trace"], [
            "router_node", "rewrite_query_node", "search_news_node",
            "retrieve_news_node", "generate_node", "record_history_node",
        ])


class StaleMetadataTests(unittest.TestCase):
    def test_previous_turn_contexts_do_not_leak(self):
        """스몰톡 턴 응답에 직전 주식 턴의 contexts/question_id 가 남지 않는다."""
        resources = fake_resources()
        cfg = {"configurable": {"thread_id": "leak-check"}}
        graph = graph_module.build_graph(resources)

        with mock.patch.object(graph_module, "classify", return_value="stock_rag"), \
             mock.patch.object(graph_module, "rewrite_stock_query", return_value=RAG_REWRITE), \
             no_network_collect():
            first = graph.invoke({"question": "삼성전자 주가 알려줘"}, cfg)
        self.assertTrue(first["contexts"], "1턴이 컨텍스트를 만들어야 의미 있는 검증이다")

        with mock.patch.object(graph_module, "classify", return_value="smalltalk"):
            second = graph.invoke({"question": "안녕"}, cfg)

        self.assertEqual(second.get("contexts"), [])
        self.assertFalse(second.get("question_id"))
        self.assertEqual(second.get("retrieved_docs"), [])


class HistoryTests(unittest.TestCase):
    def test_history_accumulates_question_and_answer(self):
        resources = fake_resources()
        cfg = {"configurable": {"thread_id": "history-grow"}}
        with mock.patch.object(graph_module, "classify", return_value="fallback"):
            graph = graph_module.build_graph(resources)
            graph.invoke({"question": "첫 질문"}, cfg)
            second = graph.invoke({"question": "둘째 질문"}, cfg)

        self.assertEqual([h["question"] for h in second["history"]],
                         ["첫 질문", "둘째 질문"])

    def test_trim_history_respects_token_budget(self):
        long_history = [
            {"question": f"질문 {i} " * 20, "answer": f"답변 {i} " * 20}
            for i in range(20)
        ]
        kept = trim_history(long_history, max_tokens=100)
        serialized = "".join(
            f"질문: {h['question']}\n답변: {h['answer']}\n" for h in kept
        )
        self.assertLessEqual(_tokens(serialized), 100)
        self.assertEqual(kept[-1], long_history[-1], "최신 턴을 버리면 안 된다")

    def test_trim_history_keeps_latest_even_if_it_alone_exceeds_budget(self):
        huge = [{"question": "질문 " * 500, "answer": "답변 " * 500}]
        self.assertEqual(trim_history(huge, max_tokens=10), huge)

    def test_trim_history_on_empty_is_empty(self):
        self.assertEqual(trim_history([], max_tokens=100), [])


class HistoryRenderContractTests(unittest.TestCase):
    """Task 1 이월: `{history}` 는 호출자가 머리말과 끝 빈 줄까지 넘기는 계약이다.

    끝 빈 줄이 없으면 `참고 뉴스:` 가 이력 마지막 줄에 붙어버린다. Task 1 에서
    주석으로만 남겨뒀던 것을 여기서 테스트로 고정한다.
    """

    def test_rendered_history_block_ends_with_blank_line(self):
        block = graph_module.render_history([
            {"question": "삼성전자 알아?", "answer": "응 알아."},
        ])
        self.assertTrue(block.endswith("\n\n"), repr(block[-10:]))

    def test_empty_history_renders_empty_string(self):
        self.assertEqual(graph_module.render_history([]), "")

    def test_prompt_keeps_news_header_on_its_own_line(self):
        from model.rag.prompts import STOCK_NEWS_PROMPT

        block = graph_module.render_history([
            {"question": "삼성전자 알아?", "answer": "응 알아."},
        ])
        rendered = STOCK_NEWS_PROMPT.format(
            context_text="뉴스 본문", question="그 회사 어때?", history=block,
        )
        self.assertIn("\n참고 뉴스:\n", rendered)


if __name__ == "__main__":
    unittest.main()
