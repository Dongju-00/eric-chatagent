import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model.graph.graph import build_graph
from model.rag.search import search_stock_news
from model.store.vector_store import ChromaNewsVectorStore, parse_news_items

# ──────────────────────────────────────────────
# 전역 리소스 (서버 시작 시 1회 초기화)
# ──────────────────────────────────────────────
resources: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("모델 및 그래프 로딩 중...", flush=True)
    resources["graph"] = build_graph()      # KoreanGPT 2종 + 라우터 로딩
    resources["store"] = ChromaNewsVectorStore()
    print("준비 완료. http://127.0.0.1:8000/docs", flush=True)
    yield
    resources.clear()


app = FastAPI(
    title="Korean Stock Chatbot Agent API",
    description="LangGraph 라우팅 Agent (smalltalk / stock_rag / fallback)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class IndexRequest(BaseModel):
    query: str = Field(..., min_length=1)
    display: int = Field(default=10, ge=1, le=30)


class NewsSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    display: int = Field(default=10, ge=1, le=30)
    start: int = Field(default=1, ge=1)
    sort: Literal["sim", "date"] = "date"


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


""" 검색이 제대로 되는지 확인하기 위한 디버깅 코드
@app.post("/rag/index", include_in_schema=False)
async def index_news(request: IndexRequest) -> dict[str, Any]:
    store = resources.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail="스토어가 준비되지 않았습니다.")

    try:
        question_id = await asyncio.to_thread(
            store.build_news_vector_db,
            request.query,
            request.display,
        )
        return {"query": request.query, "question_id": question_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/rag/search", include_in_schema=False)
async def search_indexed(
    question: str = Query(..., min_length=1),
    top_k: int = Query(default=3, ge=1, le=10),
) -> dict[str, Any]:
    store = resources.get("store")
    if store is None:
        raise HTTPException(status_code=503, detail="스토어가 준비되지 않았습니다.")

    try:
        result = await asyncio.to_thread(store.search_similar_news, question, top_k)
        return {
            "question": question,
            "documents": result.get("documents", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "distances": result.get("distances", [[]])[0],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
"""