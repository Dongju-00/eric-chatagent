"""FastAPI `lifespan` 이 리소스를 만들어 그래프에 주입하는지 본다.

Task 1의 핵심 변경이 "리소스 생성을 모듈 레벨에서 lifespan으로 옮긴다"이므로
그 경로에 실제 테스트가 있어야 한다. `build_resources` 만 가짜로 바꿔치기하고
`build_graph` 는 진짜를 돌리기 때문에, 주입 배선이 끊기면 이 테스트가 깨진다.
가중치나 Chroma는 건드리지 않는다.
"""

import os
import sys
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

from _fakes import fake_resources  # noqa: E402

import app.app as app_module  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class LifespanTests(unittest.TestCase):
    def setUp(self):
        self.resources = fake_resources()
        patcher = unittest.mock.patch.object(
            app_module, "build_resources", return_value=self.resources
        )
        self.build_resources_mock = patcher.start()
        self.addCleanup(patcher.stop)
        # 이전 테스트가 남긴 전역 상태가 있으면 지운다.
        app_module.resources.clear()
        self.addCleanup(app_module.resources.clear)

    def test_startup_builds_resources_once_and_injects_them(self):
        with TestClient(app_module.app) as client:
            self.assertEqual(self.build_resources_mock.call_count, 1)

            self.assertEqual(
                sorted(app_module.resources), ["graph", "resources"]
            )
            # lifespan이 만든 바로 그 객체가 보관된다.
            self.assertIs(app_module.resources["resources"], self.resources)
            self.assertIsNotNone(app_module.resources["graph"])

            body = client.get("/health").json()
            self.assertEqual(body["status"], "ok")
            self.assertTrue(body["graph_ready"])

    def test_shutdown_clears_resources(self):
        with TestClient(app_module.app):
            self.assertIn("graph", app_module.resources)

        self.assertEqual(app_module.resources, {})

    def test_graph_is_built_from_the_injected_resources(self):
        """lifespan이 만든 그래프가 주입된 리소스를 쓴다."""
        with unittest.mock.patch(
            "model.graph.graph.classify", return_value="smalltalk"
        ):
            with TestClient(app_module.app):
                graph = app_module.resources["graph"]
                result = graph.invoke(
                    {"question": "안녕"},
                    {"configurable": {"thread_id": "lifespan-smalltalk"}},
                )

        self.assertEqual(result["answer"], "SMALLTALK-FAKE")
        self.assertEqual(self.resources.small_talk_llm.prompts, ["질문: 안녕\n 답변: "])

    def test_chat_endpoint_503_before_startup(self):
        """lifespan 전에는 그래프가 없으므로 503이다."""
        client = TestClient(app_module.app)  # 컨텍스트 매니저로 쓰지 않는다
        response = client.post("/agent/chat", json={"question": "안녕"})
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
