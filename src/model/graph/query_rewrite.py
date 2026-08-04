import json
import re
from pathlib import Path
from typing import Any


# query_rewrite.py가 src/model/graph 안에 있는 경우
MODEL_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = MODEL_DIR / "data" / "stock" / "stock_query_config.json"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    stock_config = json.load(f)


companies = stock_config["companies"]
stock_intents = stock_config["stock_intents"]
stock_keywords = stock_config["stock_keywords"]
time_expressions = stock_config["time_expressions"]
query_rules = stock_config["query_rules"]


# --------------------------------------------------
# 기업 별칭 검색용 데이터 생성
# --------------------------------------------------

ALIAS_TO_COMPANY = {}

for company in companies:
    for alias in company.get("aliases", []):
        ALIAS_TO_COMPANY[alias.lower()] = company

# 긴 별칭부터 검사
# 예: "삼성전자"를 "삼성"보다 먼저 검사
SORTED_ALIASES = sorted(
    ALIAS_TO_COMPANY.keys(),
    key=len,
    reverse=True,
)

def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)

    return text

def extract_company(question: str) -> dict[str, Any]:
    """
    질문에서 구체적인 기업을 찾는다.

    반환 예:
    {
        "company": "SK하이닉스",
        "ticker": "000660",
        "aliases": [...],
        "candidates": []
    }
    """
    normalized = normalize_text(question)
    lowered = normalized.lower()

    # 구체적인 기업명을 먼저 검사
    for alias in SORTED_ALIASES:
        if alias in lowered:
            company = ALIAS_TO_COMPANY[alias]

            return {
                "company": company["canonical_name"],
                "ticker": company.get("ticker"),
                "aliases": company.get("aliases", []),
                "business_keywords": company.get("business_keywords", []),
                "candidates": [],
            }

    return {
        "company": None,
        "ticker": None,
        "aliases": [],
        "business_keywords": [],
        "candidates": [],
    }


def extract_time_expressions(question: str) -> list[str]:
    """최근, 오늘, 2분기 등의 시간 표현을 찾는다."""
    normalized = normalize_text(question)
    results = []

    relative_expressions = time_expressions.get("relative", {})

    for original, info in relative_expressions.items():
        if original in normalized:
            results.append(info.get("normalized", original))

    for period in time_expressions.get("periods", []):
        if period in normalized:
            results.append(period)

    for pattern in time_expressions.get("date_patterns", []):
        matches = re.findall(pattern, normalized)
        results.extend(matches)

    results = sorted(
        set(results),
        key=len,
        reverse=True,
    )

    filtered = []

    for result in results:
        if any(result in existing for existing in filtered):
            continue

        filtered.append(result)

    return filtered

def clean_question(question: str) -> str:
    cleaned = normalize_text(question)

    for stop_word in query_rules.get("stop_words", []):
        cleaned = cleaned.replace(stop_word, " ")

    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def extract_intents(question: str, max_intents: int = 2) -> list[dict[str, Any]]:
    """
    질문의 주식 관련 의도를 찾는다.

    예:
    '영업이익이 왜 줄었어?'
    → earnings, earnings_cause
    """
    normalized = normalize_text(question)
    lowered = normalized.lower()

    matched_intents = []

    earnings_terms = {
        "실적",
        "매출",
        "영업이익",
        "순이익",
        "적자",
        "흑자",
        "영업손실",
    }

    has_earnings_term = any(
        term in lowered
        for term in earnings_terms
    )

    for intent_name, intent_info in stock_intents.items():
        matched_keywords = [
            keyword
            for keyword in intent_info.get("keywords", [])
            if keyword.lower() in lowered
        ]

        if not matched_keywords:
            continue

        # 단순히 "왜"만 있다고 실적 원인으로 판단하지 않음
        if intent_name == "earnings_cause":
            specific_keywords = [
                keyword
                for keyword in matched_keywords
                if keyword not in {
                    "왜",
                    "이유",
                    "원인",
                    "배경",
                }
            ]

            if not specific_keywords and not has_earnings_term:
                continue

        score = sum(len(keyword) for keyword in matched_keywords)

        score += len(matched_keywords) * 5

        matched_intents.append({
            "name": intent_name,
            "score": score,
            "matched_keywords": matched_keywords,
            "query_terms": intent_info.get("query_terms", []),
            "sort": intent_info.get("sort", "sim"),
        })

    matched_intents.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return matched_intents[:max_intents]


def flatten_keyword_values(data: Any) -> list[str]:
    """중첩된 stock_keywords에서 문자열만 꺼낸다."""
    results = []

    if isinstance(data, str):
        results.append(data)

    elif isinstance(data, list):
        for item in data:
            results.extend(flatten_keyword_values(item))

    elif isinstance(data, dict):
        for value in data.values():
            results.extend(flatten_keyword_values(value))

    return results


ALL_STOCK_KEYWORDS = list(
    dict.fromkeys(
        flatten_keyword_values(
            stock_keywords
        )
    )
)

# HBM3E처럼 구체적인 긴 단어를 먼저 검사
ALL_STOCK_KEYWORDS.sort(
    key=len,
    reverse=True,
)


def extract_important_keywords(question: str, company_info: dict[str, Any], max_keywords: int = 4) -> list[str]:
    """
    질문에서 HBM, 영업이익, 배당, 수주 등의
    중요한 단어를 찾아낸다.
    """
    lowered = normalize_text(question).lower()

    candidates = (
        ALL_STOCK_KEYWORDS
        + company_info.get(
            "business_keywords",
            [],
        )
    )

    matched = []

    for keyword in candidates:
        if keyword.lower() in lowered:
            matched.append(keyword)

    # 긴 단어를 우선 사용
    matched.sort(
        key=len,
        reverse=True,
    )

    return list(
        dict.fromkeys(matched)
    )[:max_keywords]

def deduplicate_query_parts(parts: list[str]) -> list[str]:
    """
    완전히 같은 표현뿐 아니라 포함 관계에 있는 표현도 정리한다.

    예:
    2분기 + 분기           → 2분기
    실적 + 실적 원인       → 실적 원인
    수주 + 수주 공급계약   → 수주 공급계약
    최근 + 최근 뉴스       → 최근 뉴스
    """
    result = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        # 새로운 표현이 기존 표현에 포함되면 추가하지 않음
        if any(part in existing for existing in result):
            continue

        # 기존의 짧은 표현이 새로운 표현에 포함되면 제거
        result = [
            existing
            for existing in result
            if existing not in part
        ]

        result.append(part)

    return result

def rewrite_stock_query(question: str) -> dict[str, Any]:
    """
    사용자 질문을 네이버 뉴스 검색어로 변환한다.
    """
    question = normalize_text(question)

    company_info = extract_company(question)

    intents = extract_intents(question)
    times = extract_time_expressions(question)
    important_keywords = (extract_important_keywords(question, company_info))

    query_parts = []

    company = company_info["company"]

    if company:
        query_parts.append(company)

    query_parts.extend(times)
    query_parts.extend(important_keywords)

    for intent in intents:
        query_parts.extend(intent["query_terms"])

    # "SK하이닉스 주식"처럼 구체적인 주제가 없는 경우
    if company and not intents:
        query_parts.extend(
            query_rules.get("defaults", {}).get(
                "company_only_query_terms",
                ["최근", "실적", "주가 전망"],
            )
        )

    # 회사명이 없는 시장 전체 질문
    if not company:
        query_parts.insert(0, clean_question(question))

    query_parts = deduplicate_query_parts(query_parts)

    max_query_terms = (query_rules.get("defaults", {}).get("max_query_terms", 8))

    query_parts = query_parts[:max_query_terms]

    rewritten_query = " ".join(query_parts)

    # 최신 뉴스와 기간 질문은 날짜순
    use_date_sort = bool(times)

    if any(intent["sort"] == "date" for intent in intents):
        use_date_sort = True

    search_sort = ("date" if use_date_sort else query_rules.get("defaults", {}).get("default_sort", "sim"))

    return {
        "original_question": question,
        "company": company,
        "ticker": company_info["ticker"],
        "company_aliases": company_info["aliases"],
        "rewritten_query": rewritten_query,
        "sort": search_sort,
        "matched_intents": [intent["name"] for intent in intents],
        "important_keywords": important_keywords,
        "time_expressions": times,
    }


if __name__ == "__main__":
    test_questions = [
        "SK하이닉스 주식",
        "SK하이닉스 2분기 영업이익이 왜 줄었어?",
        "삼성전자 최근 HBM 수주 소식 알려줘",
        "LG 주식 알려줘",
        "현대차 배당 관련 뉴스 알려줘",
    ]

    for test_question in test_questions:
        print("=" * 60)
        print(
            rewrite_stock_query(
                test_question
            )
        )