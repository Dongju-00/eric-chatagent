import uuid
from datetime import datetime
from html.parser import HTMLParser
from html import unescape
import chromadb
from model.rag.search import search_stock_news
from model.embed.embedder import KoreanGPTEmbedder
import trafilatura
from pathlib import Path

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from model.graph.query_rewrite import rewrite_stock_query

from concurrent.futures import ThreadPoolExecutor
from trafilatura.settings import use_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "storage" / "chroma"

# 응답 없는 사이트에서 무한 대기하는 걸 방지
_TRAFILATURA_CFG = use_config()
_TRAFILATURA_CFG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "5")

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

def extract_article_body(url: str) -> str:
    if not url:
        return ""

    try:
        downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CFG) # config=_TRAFILATURA_CFG

        if not downloaded:
            return ""

        body = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )

        return body.strip() if body else ""

    except Exception as e:
        print(f"기사 본문 추출 실패: {url} / {e}")
        return ""

# def parse_news_items(items):
#     documents = []
#
#     for item in items:
#         title = html_to_text(item.get("title", ""))
#         description = html_to_text(item.get("description", ""))
#         link = item.get("link", "")
#         pub_date = item.get("pubDate", "")
#
#         text = f"""
# 제목: {title}
# 내용: {description}
# 링크: {link}
# 작성일: {pub_date}
# """.strip()
#
#         documents.append({
#             "text": text,
#             "metadata": {
#                 "title": title,
#                 "link": link,
#                 "pubDate": pub_date,
#                 "source": "naver_news",
#             }
#         })
#
#     return documents

def extract_news_body(original_link, naver_link, min_length=100):
    body = extract_article_body(original_link)

    if body and len(body) >= min_length:
        return body, original_link, "article_body"

    body = extract_article_body(naver_link)

    if body and len(body) >= min_length:
        return body, naver_link, "article_body"

    return "", original_link or naver_link, "naver_description"

def parse_news_items(items):
    def build(item):
        title = html_to_text(item.get("title", ""))
        description = html_to_text(item.get("description", ""))

        # 원문 링크를 우선 사용하고, 없으면 네이버 링크 사용
        original_link = item.get("originallink", "")
        naver_link = item.get("link", "")
        pub_date = item.get("pubDate", "")

        article_body, article_link, content_type = extract_news_body(original_link, naver_link)
        content = article_body if article_body else description
        # 너무 긴 기사는 최대 길이 제한
        content = content[:5000]

        # text = f"""
        #         제목: {title}
        #         작성일: {pub_date}
        #         내용: {content}
        #         """.strip()

        return ({
            # 아직 제목과 날짜를 합치지 않고 본문만 저장
            "content": content,
            "metadata": {
                "title": title,
                "link": article_link,
                "pubDate": pub_date,
                "source": "naver_news",
                "content_type": content_type,
            }
        })

    with ThreadPoolExecutor(max_workers=10) as ex:
        return list(ex.map(build, items))


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

CUTOFF_DAYS = 30


def is_recent(item, days=CUTOFF_DAYS):
    """네이버 pubDate(RFC 822 형식)를 파싱해 최근 기사인지 확인"""
    try:
        pub = parsedate_to_datetime(item["pubDate"])
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return pub > cutoff
    except Exception:
        return True  # 파싱 실패 시 일단 통과


class ChromaNewsVectorStore:
    def __init__(self, persist_path=None):
        path = Path(persist_path) if persist_path else DEFAULT_CHROMA_PATH
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
                "content_type": metadata.get("content_type", "naver_description"),
                "created_at": datetime.now().isoformat(),
            })

        self.news_collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(documents)

    def build_news_vector_db(self, query, display=10, sort="sim", company = None, aliases = None):
        # 사용자 질문 저장
        question_id = self.save_question(query)
        result = search_stock_news(query=query, display=display*3, sort=sort)
        items = result["items"]

        # 최근 + 제목에 종목명(별칭 포함) 매칭
        def matches_company(item):
            if not company:
                return True
            text = (html_to_text(item.get("title", "")) + " " +
                    html_to_text(item.get("description", "")))
            return any(a.lower() in text.lower() for a in [company, *(aliases or [])])

        recent = [it for it in items if is_recent(it) and matches_company(it)]
        print(f"[수집] 전체 {len(items)}건 → 최근+종목매칭 {len(recent)}건", flush=True)

        # 필터 후 남은 게 없으면 원본 사용 (검색 결과 0건 방지)
        if not recent:
            recent = items[:display]

        documents = parse_news_items(recent[:display])

        # 파싱
        chunked_docs = []

        # 청킹
        for doc in documents:
            # chunks = chunk_text(doc["text"])
            content = doc["content"]
            metadata = doc["metadata"]

            # 직접 만든 모델의 block_size가 작으므로
            # 500자보다 조금 작게 자르는 편이 좋음
            chunks = chunk_text(content, chunk_size=358, overlap=50,)

            for idx, chunk in enumerate(chunks):
                # 모든 chunk에 제목과 작성일을 다시 붙임
                formatted_chunk = (
                    f"제목: {metadata.get('title', '')}\n"
                    f"작성일: {metadata.get('pubDate', '')}\n"
                    f"내용: {chunk}"
                )

                chunked_docs.append({
                    "text": formatted_chunk,
                    "metadata": {
                        **metadata,
                        "chunk_id": idx,
                    }
                })

        # 인덱싱
        save_count = self.save_news_chunks(question_id, chunked_docs)

        print(f"질문 ID: {question_id}")
        print(f"{save_count}개 chunk 저장 완료")

        return question_id

    # Retrieval 임베딩
    def search_similar_news(self, question: str, question_id: str, top_k=3, max_distance: float = 1.25):
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

        # 현재 질문으로 수집한 뉴스만 검색
        if question_id:
            query_args["where"] = {
                "question_id": question_id
            }

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

        # return self.news_collection.query(
        #     query_embeddings=[query_embedding],
        #     n_results=max(top_k * 3, top_k),
        #     where={"question_id": question_id},
        #     include=["documents", "metadatas", "distances"],
        # )

    def get_chunks_by_question_id(self, question_id):
        return self.news_collection.get(
            where={"question_id": question_id},
            include=["documents", "metadatas"],
        )
