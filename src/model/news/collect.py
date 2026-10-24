"""뉴스 수집 — 검색, 종목 필터, 본문 추출, 청킹.

`vector_store.py` 가 수집·추출·청킹·저장·검색을 모두 들고 있었다. 저장과
검색만 거기 남기고 수집을 여기로 옮겼다. 경계를 나눈 이유는 종목 필터 강화와
RSS 코퍼스 도입이 각각 이 파일과 저장소 쪽만 건드리게 하기 위한 것이다.

`select_articles` 는 네트워크를 타지 않는다. 입력 dict 리스트만 보고 고른다.
본문을 받아오는 일은 `extract_and_chunk` 가 한다.
"""

import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser

import trafilatura
from trafilatura.settings import use_config

from model.rag.search import search_stock_news

__all__ = [
    "CUTOFF_DAYS",
    "collect_and_index",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "chunk_text",
    "extract_and_chunk",
    "fetch_candidates",
    "html_to_text",
    "is_recent",
    "parse_pub_date",
    "select_articles",
]

# 응답 없는 사이트에서 무한 대기하는 걸 방지
_TRAFILATURA_CFG = use_config()
_TRAFILATURA_CFG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "5")

# 주가 뉴스는 30일이 너무 길다. 실측에서 19일 전 기사가 선택됐다.
CUTOFF_DAYS = int(os.getenv("ERIC_NEWS_CUTOFF_DAYS", "5"))

# 직접 만든 모델의 block_size(384)가 작으므로 500자보다 조금 작게 자른다
CHUNK_SIZE = 358
CHUNK_OVERLAP = 50

# 후보를 요청 수의 몇 배로 받아올지. 제목 일치를 우선하도록 바꾼 뒤로는
# 통과하지 못하는 후보가 늘어 풀이 더 커야 한다(3 -> 8).
CANDIDATE_MULTIPLIER = int(os.getenv("ERIC_NEWS_CANDIDATE_MULTIPLIER", "8"))

_MAX_CONTENT_CHARS = 5000
_GENERIC_TERMS: frozenset[str] | None = None
_EXTRACT_WORKERS = 10


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


def parse_pub_date(item) -> datetime | None:
    """pubDate(RFC 822)를 파싱한다. 못 읽으면 None."""
    try:
        return parsedate_to_datetime(item.get("pubDate", ""))
    except Exception:
        return None


def is_recent(item, days=None):
    """최근 기사인지 확인한다.

    날짜를 못 읽는 기사는 통과시키되, `select_articles` 가 정렬에서 뒤로
    밀어낸다. 예전에는 파싱 실패를 무조건 통과(fail-open)시켜 오래된 기사가
    최신 기사보다 앞설 수 있었다.
    """
    if days is None:
        days = CUTOFF_DAYS
    pub = parse_pub_date(item)
    if pub is None:
        return True
    return pub > datetime.now(timezone.utc) - timedelta(days=days)


def extract_article_body(url: str) -> str:
    if not url:
        return ""

    try:
        downloaded = trafilatura.fetch_url(url, config=_TRAFILATURA_CFG)

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


def extract_news_body(original_link, naver_link, min_length=100):
    body = extract_article_body(original_link)

    if body and len(body) >= min_length:
        return body, original_link, "article_body"

    body = extract_article_body(naver_link)

    if body and len(body) >= min_length:
        return body, naver_link, "article_body"

    return "", original_link or naver_link, "naver_description"


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


def fetch_candidates(query: str, display: int, sort: str) -> list[dict]:
    """네이버 뉴스 검색으로 후보 기사를 받아온다. 필터링은 하지 않는다."""
    result = search_stock_news(query=query, display=display, sort=sort)
    return result["items"]


def _generic_market_terms() -> frozenset[str]:
    """종목을 가리지 못하는 일반 시장 용어.

    종목이 식별되지 않은 질문에서 느슨한 매칭을 할 때, "주가"·"실적" 같은
    용어로 아무 기사나 통과시키지 않기 위한 불용어다. 사전의
    ``stock_keywords`` / ``stock_intents`` / ``time_expressions`` 가 바로
    그 일반 어휘 목록이므로 따로 적지 않고 재사용한다.
    """
    global _GENERIC_TERMS
    if _GENERIC_TERMS is not None:
        return _GENERIC_TERMS

    terms = set()
    try:
        from model.graph.query_rewrite import stock_config

        for group in stock_config.get("stock_keywords", {}).values():
            terms.update(t.lower() for t in group)
        for intent in stock_config.get("stock_intents", {}).values():
            terms.update(t.lower() for t in intent.get("keywords", []))
        relative = stock_config.get("time_expressions", {}).get("relative", {})
        terms.update(t.lower() for t in relative)
    except Exception as e:  # 사전을 못 읽어도 수집은 계속돼야 한다
        print(f"[수집] 일반 용어 목록 로드 실패, 느슨한 매칭이 느슨해진다: {e}", flush=True)

    _GENERIC_TERMS = frozenset(terms)
    return _GENERIC_TERMS


def _distinctive_tokens(question: str | None) -> list[str]:
    """질문에서 종목을 가릴 수 있는 토큰만 남긴다."""
    if not question:
        return []
    generic = _generic_market_terms()
    tokens = []
    for raw in re.split(r"[^0-9A-Za-z가-힣]+", question):
        token = raw.strip().lower()
        if len(token) < 2 or token in generic:
            continue
        tokens.append(token)
    return tokens


def select_articles(
    items: list[dict],
    company: str | None,
    aliases: list[str],
    limit: int,
    question: str | None = None,
) -> list[dict]:
    """후보에서 최근 + 종목 일치 기사를 고른다. 네트워크를 타지 않는다.

    세 가지 fail-open 을 닫았다.

    1. ``company`` 가 없을 때 무조건 통과시키지 않는다. 질문의 변별력 있는
       토큰이 제목에 있으면 ``loose`` 로 살리고, 없으면 제외한다. 사전에
       없는 종목(실측: 카카오) 때문에 필터가 꺼져 "대전 대덕구 체납 징수"가
       수집되던 경로다.
    2. 제목 일치를 본문 일치보다 앞세운다. 본문에 종목명이 스쳐서 통과한
       기사(실측: 삼성전자 질문에 SK하이닉스·삼성E&A 기사)를 뒤로 밀어낸다.
       버리지는 않는다 — 제목 일치가 부족할 때 채운다.
    3. 결과가 0건이면 0건을 반환한다. 걸러지지 않은 원본으로 되돌아가지 않는다.

    고른 기사에는 ``metadata.match_kind`` (``title`` / ``loose`` /
    ``description``) 와 ``metadata.company_resolved`` 를 남겨 측정에서
    어떤 경로로 통과했는지 보이게 한다.
    """
    names = [n for n in [company, *(aliases or [])] if n]
    tokens = [] if names else _distinctive_tokens(question)

    def match_kind(item) -> str | None:
        title = html_to_text(item.get("title", "")).lower()
        if names:
            if any(n.lower() in title for n in names):
                return "title"
            if any(n.lower() in html_to_text(item.get("description", "")).lower()
                   for n in names):
                return "description"
            return None
        if tokens and any(t in title for t in tokens):
            return "loose"
        return None

    recent = [it for it in items if is_recent(it)]
    tagged = [(it, match_kind(it)) for it in recent]
    kept = [(it, kind) for it, kind in tagged if kind is not None]

    # 제목 일치로 limit 을 먼저 채우고, 부족하면 본문 전용 일치로 채운다.
    # 같은 등급 안에서는 날짜를 아는 최신 기사가 앞선다. 날짜를 모르는
    # 기사는 뒤로 밀어낸다 — 예전 fail-open 에서는 앞설 수 있었다.
    priority = {"title": 0, "loose": 1, "description": 2}
    _epoch = datetime.min.replace(tzinfo=timezone.utc)

    def sort_key(pair):
        item, kind = pair
        pub = parse_pub_date(item)
        return (priority[kind], pub is None, -(pub or _epoch).timestamp())

    kept.sort(key=sort_key)

    counts = Counter(kind for _, kind in kept)
    print(
        f"[수집] 전체 {len(items)}건 → 최근 {len(recent)}건 → "
        f"선택 가능 {len(kept)}건 (제목 {counts['title']} / "
        f"느슨 {counts['loose']} / 본문 {counts['description']})",
        flush=True,
    )

    selected = []
    for item, kind in kept[:limit]:
        enriched = dict(item)
        enriched["metadata"] = {
            **item.get("metadata", {}),
            "match_kind": kind,
            "company_resolved": company is not None,
        }
        selected.append(enriched)

    return selected


def _parse_news_item(item: dict) -> dict:
    title = html_to_text(item.get("title", ""))
    description = html_to_text(item.get("description", ""))

    # 원문 링크를 우선 사용하고, 없으면 네이버 링크 사용
    original_link = item.get("originallink", "")
    naver_link = item.get("link", "")
    pub_date = item.get("pubDate", "")

    article_body, article_link, content_type = extract_news_body(original_link, naver_link)
    content = article_body if article_body else description
    content = content[:_MAX_CONTENT_CHARS]

    return {
        # 아직 제목과 날짜를 합치지 않고 본문만 저장
        "content": content,
        "metadata": {
            # 호출자가 붙인 메타데이터를 먼저 깔고 여기서 만든 값으로 덮는다.
            # RSS 경로는 tag_companies 가 company/ticker 를 붙인 뒤 들어오므로,
            # 새로 만들면 종목 태그가 사라져 코퍼스 필터가 영구히 0건이 된다.
            **item.get("metadata", {}),
            "title": title,
            "link": article_link,
            "pubDate": pub_date,
            "source": item.get("metadata", {}).get("source", "naver_news"),
            "content_type": content_type,
        },
    }


def parse_news_items(items):
    with ThreadPoolExecutor(max_workers=_EXTRACT_WORKERS) as ex:
        return list(ex.map(_parse_news_item, items))


def extract_and_chunk(
    items: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """고른 기사의 본문을 받아 청킹한다. 원소는 ``{"text", "metadata"}``."""
    documents = parse_news_items(items)

    chunked_docs = []
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        for idx, chunk in enumerate(chunk_text(content, chunk_size=chunk_size, overlap=overlap)):
            # 모든 chunk에 제목과 작성일을 다시 붙인다
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
                },
            })

    return chunked_docs


def collect_and_index(
    store,
    query: str,
    display: int = 3,
    sort: str = "sim",
    company: str | None = None,
    aliases: list[str] | None = None,
) -> str:
    """질문 하나를 수집해 색인하고 ``question_id`` 를 돌려준다.

    ``store`` 는 ``save_question`` 과 ``index_chunks`` 를 가진 객체면 된다.
    여기서 저장소를 import하지 않는 이유는 수집과 저장을 분리해 두기 위한 것이다.
    """
    question_id = store.save_question(query)

    candidates = fetch_candidates(
        query=query,
        display=display * CANDIDATE_MULTIPLIER,
        sort=sort,
    )
    selected = select_articles(
        candidates, company, aliases or [], limit=display, question=query,
    )
    chunks = extract_and_chunk(selected)

    saved = store.index_chunks(question_id, chunks)

    print(f"질문 ID: {question_id}")
    print(f"{saved}개 chunk 저장 완료")

    return question_id
