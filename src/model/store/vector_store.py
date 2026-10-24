"""Chroma 저장과 검색.

뉴스 검색·종목 필터·본문 추출·청킹은 :mod:`model.news.collect` 에 있다.
이 파일은 받은 청크를 색인하고 다시 꺼내는 일만 한다.
"""

import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import chromadb

from model.embed.embedder import KoreanGPTEmbedder
from model.paths import CHROMA_PATH

__all__ = ["ChromaNewsVectorStore"]


class ChromaNewsVectorStore:
    def __init__(self, persist_path=None):
        path = Path(persist_path) if persist_path else CHROMA_PATH
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))

        self.question_collection = self.client.get_or_create_collection(
            name="user_questions"
        )

        self.news_collection = self.client.get_or_create_collection(
            name="naver_news_chunks"
        )

        self.embedder = KoreanGPTEmbedder()

    def create_question_id(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = uuid.uuid4().hex[:8]
        return f"q_{now}_{random_id}"

    def save_question(self, question):
        question_id = self.create_question_id()
        question_embedding = self.embedder.embed_text(question)

        self.question_collection.add(
            ids=[question_id],
            documents=[question],
            embeddings=[question_embedding],
            metadatas=[{
                "type": "user_question",
                "created_at": datetime.now().isoformat(),
            }],
        )

        return question_id

    def index_chunks(self, question_id: str, chunks: list[dict]) -> int:
        """청크를 질문 단위로 색인한다. 원소는 ``{"text", "metadata"}``."""
        if not chunks:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for idx, doc in enumerate(chunks):
            text = doc["text"]
            metadata = doc["metadata"]

            ids.append(f"{question_id}_chunk_{idx}")
            documents.append(text)
            embeddings.append(self.embedder.embed_text(text))
            metadatas.append({
                "question_id": question_id,
                "chunk_id": metadata.get("chunk_id", idx),
                "title": metadata.get("title", ""),
                "link": metadata.get("link", ""),
                "pubDate": metadata.get("pubDate", ""),
                "source": metadata.get("source", "naver_news"),
                "content_type": metadata.get("content_type", "naver_description"),
                "created_at": datetime.now().isoformat(),
                # scope 는 청크의 수명을 가른다. question 은 TTL 대상,
                # corpus 는 상시 유지. source 는 발행처 의미로 남긴다.
                "scope": "question",
                "indexed_at": datetime.now().isoformat(),
                "match_kind": metadata.get("match_kind", ""),
            })

        self.news_collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(documents)

    def index_corpus_chunks(self, chunks: list[dict]) -> int:
        """RSS 코퍼스 청크를 색인한다. 질문에 묶이지 않고 상시 유지된다.

        같은 기사가 다시 수집돼도 중복되지 않도록 링크와 청크 번호로 id를
        만든다. Chroma의 ``upsert`` 는 같은 id를 덮어쓴다.
        """
        if not chunks:
            return 0

        ids, documents, embeddings, metadatas = [], [], [], []
        now = datetime.now().isoformat()
        seen_ids = set()

        for idx, doc in enumerate(chunks):
            text = doc["text"]
            metadata = doc["metadata"]
            link = metadata.get("link", "") or f"nolink-{idx}"
            chunk_id = metadata.get("chunk_id", idx)

            chunk_key = f"corpus_{hashlib.sha1(link.encode()).hexdigest()[:16]}_{chunk_id}"
            # 같은 기사가 여러 피드에 실리면 같은 id가 한 배치에 두 번 들어온다.
            # Chroma upsert 는 배치 내 중복을 거부하므로 먼저 걸러낸다.
            if chunk_key in seen_ids:
                continue
            seen_ids.add(chunk_key)

            ids.append(chunk_key)
            documents.append(text)
            embeddings.append(self.embedder.embed_text(text))
            metadatas.append({
                "chunk_id": chunk_id,
                "title": metadata.get("title", ""),
                "link": metadata.get("link", ""),
                "pubDate": metadata.get("pubDate", ""),
                "source": metadata.get("source", "rss"),
                "content_type": metadata.get("content_type", "article_body"),
                "company": metadata.get("company", ""),
                "ticker": metadata.get("ticker", ""),
                "created_at": now,
                "scope": "corpus",
                "indexed_at": now,
            })

        self.news_collection.upsert(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas,
        )
        return len(documents)

    def backfill_scope(self) -> int:
        """``scope`` 가 없는 기존 청크를 ``question`` 으로 표시한다.

        Chroma의 ``where`` 로는 "키가 없음"을 표현하기 어렵다. 표시가 없는
        레거시 청크가 TTL 삭제에서 조용히 빠지는 것을 막기 위한 1회성 보정이다.
        """
        got = self.news_collection.get(include=["metadatas"])
        targets = [
            (cid, meta) for cid, meta in zip(got["ids"], got["metadatas"])
            if "scope" not in meta or "indexed_at" not in meta
        ]
        if not targets:
            return 0

        self.news_collection.update(
            ids=[cid for cid, _ in targets],
            metadatas=[{
                **meta,
                "scope": meta.get("scope", "question"),
                # 시각을 모르면 created_at 을 쓰고, 그것도 없으면 즉시 만료로 본다
                "indexed_at": meta.get("indexed_at", meta.get("created_at", "")),
            } for _, meta in targets],
        )
        return len(targets)

    def purge_expired_question_chunks(self, ttl_hours: int) -> int:
        """TTL이 지난 질문별 청크를 지운다.

        ``scope == "corpus"`` 인 RSS 코퍼스와 ``user_questions`` 컬렉션은
        대상이 아니다. ``indexed_at`` 을 못 읽는 청크는 만료로 본다 —
        레거시 청크가 영구히 남는 것을 막는다.
        """
        self.backfill_scope()

        got = self.news_collection.get(
            where={"scope": {"$eq": "question"}}, include=["metadatas"],
        )
        cutoff = datetime.now() - timedelta(hours=ttl_hours)

        expired = []
        for cid, meta in zip(got["ids"], got["metadatas"]):
            stamp = meta.get("indexed_at") or meta.get("created_at") or ""
            try:
                indexed_at = datetime.fromisoformat(stamp)
            except ValueError:
                expired.append(cid)  # 시각을 모르면 만료로 본다
                continue
            if indexed_at < cutoff:
                expired.append(cid)

        if expired:
            self.news_collection.delete(ids=expired)
            print(f"[TTL] 만료된 질문별 청크 {len(expired)}건 삭제", flush=True)

        return len(expired)

    def search_similar_news(self, question: str, question_id: str, top_k=10,
                            max_distance: float = 1.3, company: str | None = None):
        query_embedding = self.embedder.embed_text(question)

        query_args = {
            "query_embeddings": [query_embedding],
            # 필터링 후 부족할 수 있으므로 조금 더 많이 검색
            "n_results": max(top_k * 3, top_k),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        # 이번 질문으로 수집한 청크 + 상시 RSS 코퍼스를 함께 본다.
        # 코퍼스 쪽은 종목 메타데이터로 1차 필터해 키워드 검색의 정확도를
        # 임베더에만 맡기지 않는다.
        scopes = []
        if question_id:
            scopes.append({"question_id": {"$eq": question_id}})
        if company:
            scopes.append({
                "$and": [
                    {"scope": {"$eq": "corpus"}},
                    {"company": {"$eq": company}},
                ]
            })

        if len(scopes) == 1:
            query_args["where"] = scopes[0]
        elif len(scopes) > 1:
            query_args["where"] = {"$or": scopes}

        result = self.news_collection.query(**query_args)

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        filtered = []

        for document, metadata, distance in zip(
                documents,
                metadatas,
                distances,
        ):
            # max_distance가 없으면 일단 모두 허용
            if max_distance is None or distance <= max_distance:
                filtered.append(
                    (document, metadata, distance)
                )

        filtered = filtered[:top_k]

        return {
            "documents": [[item[0] for item in filtered]],
            "metadatas": [[item[1] for item in filtered]],
            "distances": [[item[2] for item in filtered]],
        }

    def get_chunks_by_question_id(self, question_id):
        return self.news_collection.get(
            where={"question_id": question_id},
            include=["documents", "metadatas"],
        )
