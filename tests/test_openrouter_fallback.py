"""fallback 자유 답변을 OpenRouter로 교체.

배포된 vLLM 라우터(Modal)가 503을 반환해, 라우터 분류가 실패하고 모든
질문이 fallback으로 간다. 자유 답변까지 같은 서버에 묶여 있어 고정 문구만
나오는 상태였다. 자유 답변을 외부 LLM으로 옮겨 라우터 장애와 분리한다.

라우터 분류(`classify`)는 vLLM에 남긴다. 교체 대상이 아니다.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.llm.openrouter import (
    FALLBACK_MESSAGE,
    REASONING_OFF,
    build_openrouter_llm,
)

KEY_ENV = "ERIC_CHATAGENT_OPENROUTER_KEY"


class BuildClientTests(unittest.TestCase):
    def test_missing_key_returns_none_not_raise(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(build_openrouter_llm())

    def test_blank_key_returns_none(self):
        with mock.patch.dict(os.environ, {KEY_ENV: "   "}, clear=True):
            self.assertIsNone(build_openrouter_llm())

    def test_key_present_builds_client_pointed_at_openrouter(self):
        with mock.patch.dict(os.environ, {KEY_ENV: "sk-test"}, clear=True):
            llm = build_openrouter_llm()
        self.assertIsNotNone(llm)
        self.assertIn("openrouter.ai", str(llm.openai_api_base))


    def test_reasoning_is_disabled(self):
        """추론을 끄지 않으면 토큰 예산이 생각 단계에서 소모돼 content 가 빈다.

        무료 티어 채팅 모델은 전부 추론 모델이라, 이 설정이 빠지면 fallback 이
        항상 고정 문구로 떨어진다. 실측: 추론 비활성 0.80s/24토큰 정상 응답 vs
        effort=low 3.80s/400토큰 전부 추론, content 빈 문자열.
        """
        with mock.patch.dict(os.environ, {KEY_ENV: "sk-test"}, clear=True):
            llm = build_openrouter_llm()
        self.assertEqual(llm.extra_body, REASONING_OFF)
        self.assertIs(REASONING_OFF["reasoning"]["enabled"], False)


class FallbackNodeTests(unittest.TestCase):
    """`fallback_node(state, llm)` 는 어떤 경우에도 예외를 밖으로 내지 않는다."""

    def _node(self):
        import model.graph.graph as graph_module
        return graph_module.fallback_node

    def test_uses_fixed_message_when_llm_is_none(self):
        out = self._node()({"question": "안녕"}, llm=None)
        self.assertEqual(out["answer"], FALLBACK_MESSAGE)

    def test_falls_back_when_llm_raises(self):
        boom = mock.Mock(invoke=mock.Mock(side_effect=RuntimeError("502 Bad Gateway")))
        out = self._node()({"question": "안녕"}, llm=boom)
        self.assertEqual(out["answer"], FALLBACK_MESSAGE)

    def test_returns_llm_answer_on_success(self):
        ok = mock.Mock(invoke=mock.Mock(return_value=mock.Mock(content="안녕하세요")))
        out = self._node()({"question": "안녕"}, llm=ok)
        self.assertEqual(out["answer"], "안녕하세요")

    def test_blank_llm_answer_falls_back(self):
        """빈 응답을 그대로 사용자에게 내보내지 않는다."""
        blank = mock.Mock(invoke=mock.Mock(return_value=mock.Mock(content="   ")))
        out = self._node()({"question": "안녕"}, llm=blank)
        self.assertEqual(out["answer"], FALLBACK_MESSAGE)

    def test_api_key_never_appears_in_answer(self):
        secret = "sk-or-v1-SECRETVALUE"
        boom = mock.Mock(invoke=mock.Mock(
            side_effect=RuntimeError(f"401 Unauthorized for key {secret}")))
        with mock.patch.dict(os.environ, {KEY_ENV: secret}, clear=True):
            out = self._node()({"question": "안녕"}, llm=boom)
        self.assertNotIn(secret, out["answer"])

    def test_trace_records_the_node(self):
        out = self._node()({"question": "안녕"}, llm=None)
        self.assertIn("fallback_node", out["trace"])


class RouterStaysOnVllmTests(unittest.TestCase):
    def test_classify_still_calls_vllm_not_openrouter(self):
        """라우터 분류는 vLLM에 남긴다. OpenRouter로 옮기지 않는다."""
        import model.graph.graph as graph_module

        with mock.patch.object(graph_module.httpx, "post") as post:
            post.return_value = mock.Mock(
                raise_for_status=mock.Mock(),
                json=mock.Mock(return_value={
                    "choices": [{"message": {"content": "smalltalk"}}]
                }),
            )
            with mock.patch.object(graph_module, "VLLM_BASE_URL", "https://vllm.test/v1"):
                label = graph_module.classify("안녕")

        self.assertEqual(label, "smalltalk")
        post.assert_called_once()
        self.assertIn("vllm.test", post.call_args.args[0])


class GraphWiringTests(unittest.TestCase):
    def test_graph_uses_injected_fallback_llm(self):
        import model.graph.graph as graph_module
        from _fakes import fake_resources

        resources = fake_resources()
        resources.fallback_llm = mock.Mock(
            invoke=mock.Mock(return_value=mock.Mock(content="OPENROUTER-SENTINEL")))

        with mock.patch.object(graph_module, "classify", return_value="fallback"):
            graph = graph_module.build_graph(resources)
            result = graph.invoke(
                {"question": "날씨 어때"},
                {"configurable": {"thread_id": "fallback-wiring"}},
            )

        self.assertEqual(result["answer"], "OPENROUTER-SENTINEL")

    def test_graph_falls_back_when_no_fallback_llm(self):
        import model.graph.graph as graph_module
        from _fakes import fake_resources

        resources = fake_resources()
        resources.fallback_llm = None

        with mock.patch.object(graph_module, "classify", return_value="fallback"):
            graph = graph_module.build_graph(resources)
            result = graph.invoke(
                {"question": "날씨 어때"},
                {"configurable": {"thread_id": "fallback-none"}},
            )

        self.assertEqual(result["answer"], FALLBACK_MESSAGE)


if __name__ == "__main__":
    unittest.main()
