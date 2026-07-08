import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from model.rag.generator import generate_reply_rag, stream_reply_rag
from model.train.model import KoreanGPT, device
from model.rag.search import search_stock_news
from model.train.sp_tokenizer import load_sp
from model.store.vector_store import ChromaNewsVectorStore, parse_news_items

BASE_DIR = Path(__file__).resolve().parent
SP_PREFIX = str(BASE_DIR / "data" / "sp_korean")
QA_CKPT = BASE_DIR / "checkpoints" / "KoreanGPT_qa.pt"


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    display: int = Field(default=10, ge=1, le=30)
    max_new_tokens: int = Field(default=120, ge=1, le=300)


class IndexNewsRequest(BaseModel):
    query: str = Field(..., min_length=1)
    display: int = Field(default=10, ge=1, le=30)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    display: int = Field(default=10, ge=1, le=30)
    start: int = Field(default=1, ge=1)
    sort: Literal["sim", "date"] = "sim"


class Runtime:
    def __init__(self):
        if not os.path.exists(f"{SP_PREFIX}.model"):
            raise FileNotFoundError(f"{SP_PREFIX}.model 파일이 없습니다.")

        if not QA_CKPT.exists():
            raise FileNotFoundError(f"{QA_CKPT} 파일이 없습니다. train_qa를 먼저 실행하세요.")

        self.sp = load_sp(SP_PREFIX)
        self.model = KoreanGPT(self.sp.get_piece_size()).to(device)
        self.model.load_state_dict(torch.load(QA_CKPT, map_location=device))
        self.model.eval()


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime()


@lru_cache(maxsize=1)
def get_store() -> ChromaNewsVectorStore:
    return ChromaNewsVectorStore()


app = FastAPI(
    title="Korean Chatbot RAG API",
    description="KoreanGPT 기반 뉴스 RAG 챗봇 API",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/news/search")
async def search_news(request: SearchRequest) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(
            search_stock_news,
            request.query,
            request.display,
            request.start,
            request.sort,
        )
        documents = parse_news_items(result.get("items", []))
        return {
            "query": request.query,
            "total": result.get("total", 0),
            "start": result.get("start", request.start),
            "display": result.get("display", request.display),
            "items": documents,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/index")
async def index_news(request: IndexNewsRequest) -> dict[str, Any]:
    try:
        store = get_store()
        return await asyncio.to_thread(
            store.build_news_vector_db,
            request.query,
            request.display,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/rag/search")
async def search_indexed_news(
    question: str = Query(..., min_length=1),
    top_k: int = Query(default=3, ge=1, le=10),
) -> dict[str, Any]:
    try:
        store = get_store()
        result = await asyncio.to_thread(store.search_similar_news, question, top_k)
        return {
            "question": question,
            "documents": result.get("documents", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "distances": result.get("distances", [[]])[0],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/chat")
async def chat_rag(request: ChatRequest) -> dict[str, Any]:
    try:
        runtime = get_runtime()
        store = get_store()

        def run_rag():
            index_result = store.build_news_vector_db(
                request.question,
                display=request.display,
            )
            search_result = store.search_similar_news(
                request.question,
                top_k=request.top_k,
            )
            contexts = search_result.get("documents", [[]])[0]
            metadatas = search_result.get("metadatas", [[]])[0]
            distances = search_result.get("distances", [[]])[0]

            answer = generate_reply_rag(
                model=runtime.model,
                sp=runtime.sp,
                question=request.question,
                contexts=contexts,
                max_new_tokens=request.max_new_tokens,
            )

            return {
                "question": request.question,
                "question_id": index_result["question_id"],
                "chunk_count": index_result["chunk_count"],
                "answer": answer,
                "contexts": contexts,
                "metadatas": metadatas,
                "distances": distances,
            }

        return await asyncio.to_thread(run_rag)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/rag/chat/stream")
async def chat_rag_stream(request: ChatRequest) -> StreamingResponse:
    return _rag_stream_response(request)


@app.get("/rag/chat/stream")
async def chat_rag_stream_get(
    question: str = Query(..., min_length=1),
    top_k: int = Query(default=3, ge=1, le=10),
    display: int = Query(default=10, ge=1, le=30),
    max_new_tokens: int = Query(default=120, ge=1, le=300),
) -> StreamingResponse:
    request = ChatRequest(
        question=question,
        top_k=top_k,
        display=display,
        max_new_tokens=max_new_tokens,
    )
    return _rag_stream_response(request)


def _rag_stream_response(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        try:
            runtime = get_runtime()
            store = get_store()

            yield _sse("status", {"message": "news_indexing"})
            index_result = await asyncio.to_thread(
                store.build_news_vector_db,
                request.question,
                request.display,
            )

            yield _sse("status", {"message": "retrieving", **index_result})
            search_result = await asyncio.to_thread(
                store.search_similar_news,
                request.question,
                request.top_k,
            )
            contexts = search_result.get("documents", [[]])[0]

            yield _sse("status", {"message": "generating"})
            previous = ""

            for text in stream_reply_rag(
                model=runtime.model,
                sp=runtime.sp,
                question=request.question,
                contexts=contexts,
                max_new_tokens=request.max_new_tokens,
            ):
                delta = text[len(previous):]
                previous = text
                if delta:
                    yield _sse("token", {"text": delta})

            yield _sse("done", {"answer": previous, **index_result})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
