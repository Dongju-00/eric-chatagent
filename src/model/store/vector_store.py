import uuid
from datetime import datetime
from html.parser import HTMLParser
from html import unescape
import chromadb
from model.rag.search import search_stock_news
from model.embed.embedder import KoreanGPTEmbedder

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.texts = []

    def handle_data(self, data):
        data = data.strip()
        if data:
            self.texts.append(data)

    def get_text(self):
        return " ".join(self.texts)


def html_to_text(html_text):
    parser = HTMLTextExtractor()
    parser.feed(unescape(html_text or ""))
    return parser.get_text()


def parse_news_items(items):
    documents = []

    for item in items:
        title = html_to_text(item.get("title", ""))
        description = html_to_text(item.get("description", ""))
        link = item.get("link", "")
        pub_date = item.get("pubDate", "")

        text = f"""
제목: {title}
내용: {description}
링크: {link}
작성일: {pub_date}
""".strip()

        documents.append({
            "text": text,
            "metadata": {
                "title": title,
                "link": link,
                "pubDate": pub_date,
                "source": "naver_news",
            }
        })

    return documents


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks

class ChromaNewsVectorStore:
    def __init__(self, persist_path="storage/chroma"):
        self.client = chromadb.PersistentClient(path=persist_path)

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

    def save_news_chunks(self, question_id, chunked_docs):
        if not chunked_docs:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for idx, doc in enumerate(chunked_docs):
            chunk_id = f"{question_id}_chunk_{idx}"
            text = doc["text"]
            metadata = doc["metadata"]

            ids.append(chunk_id)
            documents.append(text)

            # 임베딩
            embeddings.append(self.embedder.embed_text(text))
            metadatas.append({
                "question_id": question_id,
                "chunk_id": metadata.get("chunk_id", idx),
                "title": metadata.get("title", ""),
                "link": metadata.get("link", ""),
                "pubDate": metadata.get("pubDate", ""),
                "source": metadata.get("source", "naver_news"),
                "created_at": datetime.now().isoformat(),
            })

        self.news_collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def build_news_vector_db(self, query, display=10):
        # 사용자 질문 저장
        question_id = self.save_question(query)

        # 네이버 뉴스 검색
        result = search_stock_news(query, display=display)
        documents = parse_news_items(result["items"])

        # 파싱
        chunked_docs = []

        # 청킹
        for doc in documents:
            chunks = chunk_text(doc["text"])

            for idx, chunk in enumerate(chunks):
                chunked_docs.append({
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": idx,
                    }
                })

        # 인덱싱
        self.save_news_chunks(question_id, chunked_docs)

        print(f"질문 ID: {question_id}")
        print(f"{len(chunked_docs)}개 chunk 저장 완료")

        return question_id

    # Retrieval 임베딩
    def search_similar_news(self, question, top_k=3):
        query_embedding = self.embedder.embed_text(question)

        return self.news_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def get_chunks_by_question_id(self, question_id):
        return self.news_collection.get(
            where={"question_id": question_id},
            include=["documents", "metadatas"],
        )
