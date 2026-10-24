"""뉴스 수집·검색 품질과 속도를 측정한다.

변경 전후를 같은 기준으로 비교하기 위한 하네스다. 실행할 때마다
`results/<label>.json` 으로 결과를 남기고, 두 결과를 `--compare` 로
나란히 비교한다.

사용:
    PYTHONPATH=src uv run python measurements/measure_retrieval.py --label before
    PYTHONPATH=src uv run python measurements/measure_retrieval.py --label after
    uv run python measurements/measure_retrieval.py --compare before after

측정 항목
    t_collect        수집 시간 (네이버 API + 본문 추출 + 청킹 + 색인)
    t_retrieve       검색 시간 (임베딩 + Chroma 질의)
    company_hit      검색된 청크가 대상 종목을 포함하는 비율
    company_resolved 질문에서 종목이 식별된 비율 (사전 커버리지)
    age_days         검색된 청크 기사의 경과 일수
    empty_context    컨텍스트가 0건인 질문 비율
    distance         Chroma 거리 분포
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 평가 질문. evaluate_rag.py 의 TEST_QUESTIONS 와 성격을 맞추되,
# 사전에 없는 종목(카카오)과 별칭 질문(삼전)을 의도적으로 포함한다.
QUESTIONS = [
    "삼성전자 주가 전망 알려줘",
    "SK하이닉스 주식 알려줘",
    "삼전 주가 전망",
    "네이버 관련 주식 뉴스 알려줘",
    "카카오 실적 어때?",
    "현대차 배당 알려줘",
    "기아 전기차 판매량 어때?",
    "셀트리온 신약 소식 있어?",
    "두산에너빌리티 원전 수주",
    "한화에어로스페이스 방산 수출",
]


def _age_days(pub_date: str):
    try:
        return (datetime.now(timezone.utc) - parsedate_to_datetime(pub_date)).days
    except Exception:
        return None


def _pub_from_chunk(text: str):
    m = re.search(r"작성일: (.+)", text)
    return m.group(1).strip() if m else None


def run(label: str) -> dict:
    from model.graph.query_rewrite import rewrite_stock_query
    from model.news.collect import collect_and_index
    from model.store.vector_store import ChromaNewsVectorStore

    store = ChromaNewsVectorStore()
    rows = []

    for q in QUESTIONS:
        rw = rewrite_stock_query(q)
        company = rw.get("company")
        aliases = [a for a in [company, *(rw.get("company_aliases") or [])] if a]

        t0 = time.perf_counter()
        qid = collect_and_index(
            store,
            query=rw.get("rewritten_query", q),
            display=3,
            sort=rw.get("sort", "sim"),
            company=company,
            aliases=rw.get("company_aliases", []),
        )
        t_collect = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = store.search_similar_news(
            question=q, question_id=qid, top_k=2, max_distance=1.25,
            # graph.retrieve_news_node 와 같게 넘긴다. 빼면 RSS 상시 코퍼스
            # 보충 경로가 측정에서 통째로 빠진다.
            company=company,
        )
        t_retrieve = time.perf_counter() - t0

        docs = res["documents"][0]
        dists = res["distances"][0]

        hits = [
            any(a.lower() in d.lower() for a in aliases) if aliases else False
            for d in docs
        ]
        ages = [_age_days(_pub_from_chunk(d) or "") for d in docs]

        rows.append(
            {
                "question": q,
                "company_resolved": company is not None,
                "company": company,
                "t_collect": round(t_collect, 3),
                "t_retrieve": round(t_retrieve, 3),
                "n_docs": len(docs),
                "company_hits": sum(hits),
                "ages_days": ages,
                "distances": [round(x, 3) for x in dists],
            }
        )
        print(
            f"  {q[:24]:26s} 수집 {t_collect:6.2f}s 검색 {t_retrieve:5.2f}s "
            f"문서 {len(docs)} 종목일치 {sum(hits)}/{len(docs)} "
            f"경과 {ages} dist {[round(x, 2) for x in dists]}",
            flush=True,
        )

    return {"label": label, "measured_at": datetime.now().isoformat(), "rows": rows}


def summarize(result: dict) -> dict:
    rows = result["rows"]
    tc = [r["t_collect"] for r in rows]
    tr = [r["t_retrieve"] for r in rows]
    n_docs = sum(r["n_docs"] for r in rows)
    n_hits = sum(r["company_hits"] for r in rows)
    ages = [a for r in rows for a in r["ages_days"] if a is not None]
    dists = [d for r in rows for d in r["distances"]]
    resolved = sum(1 for r in rows if r["company_resolved"])

    return {
        "questions": len(rows),
        "t_collect_median": round(statistics.median(tc), 3),
        "t_collect_max": round(max(tc), 3),
        "t_retrieve_median": round(statistics.median(tr), 3),
        "t_total_median": round(statistics.median([a + b for a, b in zip(tc, tr)]), 3),
        "company_resolved_rate": round(resolved / len(rows) * 100, 1),
        "company_hit_rate": round(n_hits / n_docs * 100, 1) if n_docs else None,
        "empty_context_rate": round(
            sum(1 for r in rows if r["n_docs"] == 0) / len(rows) * 100, 1
        ),
        "age_days_median": statistics.median(ages) if ages else None,
        "age_days_max": max(ages) if ages else None,
        "distance_median": round(statistics.median(dists), 3) if dists else None,
    }


FIELDS = [
    ("questions", "질문 수", ""),
    ("t_collect_median", "수집 시간 median", "s"),
    ("t_collect_max", "수집 시간 max", "s"),
    ("t_retrieve_median", "검색 시간 median", "s"),
    ("t_total_median", "전체 median", "s"),
    ("company_resolved_rate", "종목 식별률", "%"),
    ("company_hit_rate", "종목 일치율", "%"),
    ("empty_context_rate", "컨텍스트 0건", "%"),
    ("age_days_median", "기사 경과 median", "일"),
    ("age_days_max", "기사 경과 max", "일"),
    ("distance_median", "distance median", ""),
]


def print_summary(title: str, s: dict) -> None:
    print(f"\n{title}")
    for key, name, unit in FIELDS:
        v = s.get(key)
        print(f"  {name:20s} {('-' if v is None else v)}{unit}")


def compare(a_label: str, b_label: str) -> None:
    a = summarize(json.loads((RESULTS_DIR / f"{a_label}.json").read_text()))
    b = summarize(json.loads((RESULTS_DIR / f"{b_label}.json").read_text()))
    w = 20
    print(f"\n| {'지표':<{w}} | {a_label:>12} | {b_label:>12} | {'변화':>12} |")
    print(f"| {'-' * w} | {'-' * 12} | {'-' * 12} | {'-' * 12} |")
    for key, name, unit in FIELDS:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            delta = "-"
        elif isinstance(va, (int, float)) and va:
            delta = f"{(vb - va) / va * 100:+.0f}%"
        else:
            delta = f"{vb - va:+}" if isinstance(va, (int, float)) else "-"
        print(f"| {name:<{w}} | {str(va) + unit:>12} | {str(vb) + unit:>12} | {delta:>12} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="측정 결과를 저장할 이름 (예: before)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="두 결과 비교")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.compare:
        compare(*args.compare)
        return

    if not args.label:
        ap.error("--label 또는 --compare 중 하나가 필요하다")

    print(f"측정 시작: {args.label}")
    result = run(args.label)
    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(f"요약 [{args.label}]", summarize(result))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    sys.exit(main())
