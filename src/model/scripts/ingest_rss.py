"""RSS 피드를 받아 상시 코퍼스로 색인하고, 만료된 질문별 청크를 지운다.

주기 실행용이다. cron 이나 수동으로 돌린다:

    PYTHONPATH=src uv run python src/model/scripts/ingest_rss.py

RSS 는 키워드 질의가 불가능해 네이버 검색을 대체하지 않는다. 종목
메타데이터를 붙여 미리 색인해 두고, 질문 시 수집분이 부족하거나 오래됐을
때 보충원으로 쓴다.
"""

import argparse
import os
import sys
from pathlib import Path

try:  # 직접 실행 / -m 실행 양쪽 지원
    from _bootstrap import ensure_src_on_path
except ImportError:
    from model.scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from model.graph.query_rewrite import stock_config            # noqa: E402
from model.news.collect import extract_and_chunk              # noqa: E402
from model.news.rss import FEEDS, fetch_all_feeds, tag_companies  # noqa: E402
from model.store.vector_store import ChromaNewsVectorStore    # noqa: E402

# 질문별 청크를 얼마나 들고 있을지. 같은 종목을 연달아 묻는 패턴을 흡수하기엔
# 충분하고, 오래된 수집분이 쌓이지 않을 만큼 짧게 둔다.
DEFAULT_TTL_HOURS = int(os.getenv("ERIC_QUESTION_CHUNK_TTL_HOURS", "6"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="본문을 추출할 기사 수 상한 (기본: 전부)")
    parser.add_argument("--ttl-hours", type=int, default=DEFAULT_TTL_HOURS,
                        help="질문별 청크 보존 시간")
    parser.add_argument("--skip-purge", action="store_true",
                        help="TTL 삭제를 건너뛴다")
    parser.add_argument("--dry-run", action="store_true",
                        help="색인하지 않고 수집·태깅 결과만 센다")
    args = parser.parse_args()

    print(f"[RSS] 피드 {len(FEEDS)}개 수집", flush=True)
    items = fetch_all_feeds()
    if not items:
        print("[RSS] 수집된 기사가 없다. 피드가 모두 실패했는지 확인하라.", flush=True)
        return 1

    companies = stock_config["companies"]
    tagged = tag_companies(items, companies)
    if args.limit is not None:
        tagged = tagged[: args.limit]

    if args.dry_run:
        print(f"[RSS] dry-run: 색인 대상 {len(tagged)}건")
        return 0

    # 본문 추출은 기사마다 HTTP 요청이 든다. collect 쪽 병렬 추출을 재사용한다.
    chunks = extract_and_chunk(tagged)
    print(f"[RSS] 청크 {len(chunks)}건 생성", flush=True)

    store = ChromaNewsVectorStore()
    saved = store.index_corpus_chunks(chunks)
    print(f"[RSS] 코퍼스 색인 {saved}건", flush=True)

    if not args.skip_purge:
        removed = store.purge_expired_question_chunks(ttl_hours=args.ttl_hours)
        print(f"[TTL] {args.ttl_hours}시간 지난 질문별 청크 {removed}건 삭제", flush=True)

    total = store.news_collection.count()
    print(f"[RSS] 컬렉션 총 청크 {total}건", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
