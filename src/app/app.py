import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model.graph.graph import build_graph
from model.resources import build_resources

# ──────────────────────────────────────────────
# 전역 리소스 (서버 시작 시 1회 초기화)
# ──────────────────────────────────────────────
resources: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("모델 및 그래프 로딩 중...", flush=True)
    # 리소스 생성은 여기 한 곳에서만 일어난다.
    # 임베더 1벌 + KoreanGPT 2종 = torch.load 3회, Chroma 클라이언트 1개.
    app_resources = build_resources()
    resources["resources"] = app_resources
    resources["graph"] = build_graph(app_resources)
    print("준비 완료. http://127.0.0.1:8000/docs", flush=True)
    yield
    resources.clear()


app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시엔 실제 도메인으로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# 요청 스키마
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="사용자 질문")
    thread_id: str | None = Field(
        default=None,
        description="대화 세션 ID. 같은 값을 쓰면 이전 상태가 유지된다.",
    )


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "graph_ready": "graph" in resources,
    }


@app.post("/agent/chat", summary="Agent 대화 (메인 엔드포인트)")
async def agent_chat(request: ChatRequest) -> dict[str, Any]:
    """
    라우터가 질문을 분류해 세 경로 중 하나로 보낸다.
      smalltalk  → 스몰톡 전용 모델
      stock_rag  → 뉴스 수집 → 검색 → 뉴스 전용 모델
      fallback   → 정형 응답
    """
    graph = resources.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="그래프가 아직 준비되지 않았습니다.")

    thread_id = request.thread_id or f"session-{uuid.uuid4().hex[:8]}"

    try:
        result = await asyncio.to_thread(
            graph.invoke,
            {"question": request.question},
            {"configurable": {"thread_id": thread_id}},
        )
        return {
            "question": request.question,
            "answer": result.get("answer", ""),
            "route": result.get("route"),
            "trace": result.get("trace", []),
            "contexts": result.get("contexts", []),
            "question_id": result.get("question_id"),
            "thread_id": thread_id,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent / "static"), html=True), name="home")
