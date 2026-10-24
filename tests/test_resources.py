"""리소스 수명주기 테스트.

import 자체가 가중치를 적재하지 않아야 하고, `build_resources()`가 임베더
1벌과 LLM 2종만 만들어야 한다. 두 검증 모두 **반드시 subprocess로 격리한다**:
같은 프로세스에서 `importlib.import_module`은 이미 적재된 모듈의 캐시를
돌려주므로, 다른 테스트가 먼저 import했으면 `torch.load`가 한 번도 호출되지
않아 테스트가 공허하게 통과한다.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

# 단위 테스트가 LangSmith로 트레이스를 보내지 않게 한다. 유효하지 않은 키 때문에
# 403 로그가 쏟아지고, 네트워크에 의존하는 테스트가 되기 때문이다.
os.environ["LANGSMITH_TRACING"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fakes import no_network_collect, fake_resources  # noqa: E402
from model.graph import graph as graph_module  # noqa: E402
from model.graph.graph import build_graph  # noqa: E402
from model.paths import CHECKPOINT_DIR, SP_PREFIX  # noqa: E402
from model.rag.prompts import STOCK_NEWS_PROMPT  # noqa: E402

REQUIRED_FILES = [
    CHECKPOINT_DIR / "KoreanGPT.pt",
    CHECKPOINT_DIR / "KoreanGPT_news_v2.pt",
    CHECKPOINT_DIR / "KoreanGPT_smalltalk.pt",
    Path(f"{SP_PREFIX}.model"),
]

HAVE_WEIGHTS = all(p.is_file() for p in REQUIRED_FILES)


def _run_probe(code: str, **env_overrides) -> subprocess.CompletedProcess:
    """프로젝트 루트에서 `python -c code`를 돌린다.

    LANGSMITH_TRACING=false는 유효하지 않은 키 때문에 403 로그가 쏟아지는 것을
    막기 위한 것이다.
    """
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANGSMITH_TRACING": "false",
        **env_overrides,
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


class ImportPurityTests(unittest.TestCase):
    def test_importing_graph_and_baseline_loads_no_weights(self):
        """import만으로 torch.load나 Chroma 연결이 일어나지 않아야 한다."""
        probe = (
            "import torch, chromadb\n"
            "n = [0]\n"
            "torch.load = lambda *a, **k: n.__setitem__(0, n[0] + 1)\n"
            "chromadb.PersistentClient = lambda *a, **k: n.__setitem__(0, n[0] + 1)\n"
            "import model.rag.baseline, model.graph.graph, model.rag.prompts\n"
            "print(n[0])\n"
        )
        out = _run_probe(probe)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "0", out.stdout)


class BuildResourcesTests(unittest.TestCase):
    @unittest.skipUnless(
        HAVE_WEIGHTS,
        "체크포인트나 토크나이저가 없다: " + ", ".join(
            str(p) for p in REQUIRED_FILES if not p.is_file()
        ),
    )
    def test_build_resources_creates_one_store_and_two_llms(self):
        """임베더 1벌, LLM 2종 -> torch.load 3회, PersistentClient 1회."""
        probe = (
            "import json, torch, chromadb\n"
            "counts = {'torch_load': 0, 'persistent_client': 0}\n"
            "real_load, real_client = torch.load, chromadb.PersistentClient\n"
            "def load(*a, **k):\n"
            "    counts['torch_load'] += 1\n"
            "    return real_load(*a, **k)\n"
            "def client(*a, **k):\n"
            "    counts['persistent_client'] += 1\n"
            "    return real_client(*a, **k)\n"
            "torch.load, chromadb.PersistentClient = load, client\n"
            "from model.resources import build_resources\n"
            "r = build_resources()\n"
            "assert r.store is not None and r.news_llm is not None\n"
            "assert r.small_talk_llm is not None\n"
            "counts['fallback_llm_is_none'] = r.fallback_llm is None\n"
            "import os\n"
            "counts['key_present'] = bool((os.getenv('ERIC_CHATAGENT_OPENROUTER_KEY') or '').strip())\n"
            "print('RESULT ' + json.dumps(counts))\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            # 실제 storage/chroma를 건드리지 않도록 임시 경로로 격리한다.
            out = _run_probe(probe, ERIC_CHROMA_PATH=tmp)

        self.assertEqual(out.returncode, 0, out.stderr)
        line = next(
            (l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None
        )
        self.assertIsNotNone(line, out.stdout)
        counts = json.loads(line[len("RESULT "):])

        # 핵심 불변식: 임베더 1벌 + LLM 2종 = torch.load 3회, 클라이언트 1개.
        self.assertEqual(counts["torch_load"], 3, counts)
        self.assertEqual(counts["persistent_client"], 1, counts)

        # fallback_llm 은 OpenRouter 클라이언트다(Task 5). 가중치를 적재하지
        # 않으므로 위 3회에 영향이 없다. 키가 있으면 생기고 없으면 None 이다.
        self.assertEqual(
            counts["fallback_llm_is_none"], not counts["key_present"], counts,
        )


# 변경 전(Task 1 이전) `baseline.py` 의 프롬프트 템플릿을 렌더한 결과.
# `{history}` 슬롯이 비어 있을 때 프롬프트는 이것과 글자 단위로 같아야 한다.
PRE_CHANGE_RENDER = """당신은 한국 주식 뉴스를 요약하는 챗봇입니다.
아래 참고 뉴스만 근거로 답변하세요. 뉴스에 없는 내용은 추측하지 마세요.
답변은 5문장 이내로 간결하게 작성하세요.
투자 판단(매수/매도)을 할만하다, 할만하지 않다 정도만 전달하고, 정보만 전달하세요.
그리고 지금 주가 흐름도 설명하시오.

참고 뉴스:
뉴스본문

질문: 삼성전자 어때?

답변:"""


class StockNewsPromptTests(unittest.TestCase):
    def test_empty_history_renders_identically_to_pre_change_template(self):
        """history 를 넘기지 않으면 변경 전 프롬프트와 글자 단위로 같아야 한다."""
        rendered = STOCK_NEWS_PROMPT.format(
            context_text="뉴스본문", question="삼성전자 어때?"
        )
        self.assertEqual(rendered, PRE_CHANGE_RENDER)

    def test_history_is_outside_the_evidence_block(self):
        """대화 이력이 참고 뉴스 블록 안에 들어가면 안 된다.

        `아래 참고 뉴스만 근거로 답변하세요` 아래의 증거 블록에 이력이 섞이면
        모델이 이력을 검색된 뉴스로 읽는다. 근거 없는 생성이 유입되는 경로다.
        """
        rendered = STOCK_NEWS_PROMPT.format(
            context_text="뉴스본문",
            question="그럼 지금은?",
            history="이전 대화:\n질문: 삼성전자 어때?\n답변: 좋다.\n",
        )

        history_at = rendered.index("이전 대화:")
        evidence_at = rendered.index("참고 뉴스:")
        context_at = rendered.index("뉴스본문")
        question_at = rendered.index("질문: 그럼 지금은?")

        # 이력은 증거 블록 머리말보다 앞에 있어야 한다.
        self.assertLess(history_at, evidence_at, rendered)
        # 증거 블록(머리말 + 본문)은 끊기지 않고 붙어 있어야 한다.
        self.assertLess(evidence_at, context_at, rendered)
        self.assertLess(context_at, question_at, rendered)
        self.assertIn("참고 뉴스:\n뉴스본문", rendered)


class BuildGraphInjectionTests(unittest.TestCase):
    """`build_graph` 가 주입받은 객체를 실제로 쓰는지 본다.

    시그니처만 보는 테스트는 `build_graph(resources)` 가 인자를 받고도 내부에서
    `ChromaNewsVectorStore()` 나 `build_llm()` 을 다시 만드는 회귀를 잡지 못한다.
    import 순수성 프로브는 import 시점만 보고, `build_resources` 카운트는
    `build_graph` 를 거치지 않기 때문이다. 여기서는 가짜 리소스를 주입하고
    그래프를 실제로 돌려 그 가짜가 쓰였는지 확인한다. 가중치는 적재하지 않는다.
    """

    def test_small_talk_node_uses_injected_llm(self):
        resources = fake_resources(small_talk_answer="SMALLTALK-SENTINEL")
        with unittest.mock.patch.object(
            graph_module, "classify", return_value="smalltalk"
        ):
            graph = build_graph(resources)
            result = graph.invoke(
                {"question": "안녕"},
                {"configurable": {"thread_id": "injection-smalltalk"}},
            )

        self.assertEqual(result["answer"], "SMALLTALK-SENTINEL")
        # record_history_node 는 Task 6 에서 모든 경로 끝에 추가됐다.
        self.assertEqual(
            result["trace"],
            ["router_node", "small_talk_node", "record_history_node"],
        )
        self.assertEqual(resources.small_talk_llm.prompts, ["질문: 안녕\n 답변: "])

    def test_rag_path_makes_no_network_call(self):
        """단위 테스트가 네이버 API나 기사 본문 HTTP를 타면 안 된다.

        수집 오케스트레이션이 저장소 밖(`collect_and_index`)으로 나갔으므로
        FakeStore 만으로는 네트워크를 막을 수 없다. 사용자 쿼터를 쓰고
        네트워크 상태에 따라 깨지는 테스트가 되므로 명시적으로 막는다.
        """
        resources = fake_resources(documents=("가짜 기사 본문",))
        rewritten = {
            "company": "삼성전자", "ticker": "005930", "company_aliases": ["삼전"],
            "rewritten_query": "삼성전자 주가", "sort": "sim", "matched_intents": [],
        }
        with unittest.mock.patch.object(
            graph_module, "classify", return_value="stock_rag"
        ), unittest.mock.patch.object(
            graph_module, "rewrite_stock_query", return_value=rewritten
        ), unittest.mock.patch(
            "model.news.collect.search_stock_news",
            side_effect=AssertionError("네이버 API 호출"),
        ), unittest.mock.patch(
            "model.news.collect.extract_article_body",
            side_effect=AssertionError("기사 본문 HTTP 호출"),
        ), no_network_collect():
            graph = build_graph(resources)
            graph.invoke(
                {"question": "삼성전자 주가 알려줘"},
                {"configurable": {"thread_id": "injection-no-network"}},
            )

    def test_rag_nodes_use_injected_store_and_news_llm(self):
        resources = fake_resources(
            news_answer="NEWS-SENTINEL", documents=("가짜 기사 본문",)
        )
        rewritten = {
            "company": "삼성전자",
            "ticker": "005930",
            "company_aliases": ["삼전"],
            "rewritten_query": "삼성전자 주가",
            "sort": "sim",
            "matched_intents": [],
        }
        with unittest.mock.patch.object(
            graph_module, "classify", return_value="stock_rag"
        ), unittest.mock.patch.object(
            graph_module, "rewrite_stock_query", return_value=rewritten
        ), no_network_collect(chunks=[{
            "text": "제목: 가짜 제목\n작성일: x\n내용: 가짜 기사 본문",
            "metadata": {"title": "가짜 제목", "chunk_id": 0},
        }]):
            graph = build_graph(resources)
            result = graph.invoke(
                {"question": "삼성전자 주가 알려줘"},
                {"configurable": {"thread_id": "injection-rag"}},
            )

        # 주입된 store 가 쓰였다.
        self.assertEqual(len(resources.store.build_calls), 1)
        self.assertEqual(len(resources.store.search_calls), 1)
        self.assertEqual(resources.store.build_calls[0]["query"], "삼성전자 주가")
        self.assertEqual(result["question_id"], "FAKE-QID")
        self.assertEqual(result["contexts"], ["가짜 기사 본문"])

        # 주입된 news_llm 으로 체인이 조립됐다.
        self.assertEqual(result["answer"], "NEWS-SENTINEL")
        self.assertEqual(len(resources.news_llm.prompts), 1)
        self.assertIn("가짜 기사 본문", resources.news_llm.prompts[0])
        # 스몰톡 LLM 은 이 경로에서 쓰이지 않는다.
        self.assertEqual(resources.small_talk_llm.prompts, [])


class BuildGraphSignatureTests(unittest.TestCase):
    def test_build_graph_requires_resources(self):
        with self.assertRaises(TypeError):
            build_graph()


if __name__ == "__main__":
    unittest.main()
