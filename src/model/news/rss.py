"""RSS 상시 코퍼스 수집.

RSS는 **키워드 질의가 불가능하다.** 피드는 발행사·분야 단위라 "삼성전자
뉴스"를 물어볼 수 없고, 최근 N건이 종목 무관하게 온다. 그래서 네이버
키워드 검색을 대체하지 않는다.

대신 피드를 주기적으로 받아 종목 메타데이터를 붙여 미리 색인해 두면,
키워드 검색이 0건이거나 오래됐을 때 보충원이 된다. 네이버 API 쿼터에
의존하지 않는 상시 최신 코퍼스가 생긴다.

피드 하나가 죽어도 수집 전체와 요청 경로가 살아야 한다. 모든 네트워크·
파싱 예외를 삼키고 빈 리스트를 반환한다.
"""

import urllib.error
import urllib.request
from xml.etree import ElementTree

from model.graph.query_rewrite import _alias_in_text

__all__ = ["FEEDS", "fetch_feed", "fetch_all_feeds", "tag_companies"]

# 2026-10-23 실측으로 응답하는 피드만 둔다. 네이버 뉴스 RSS는 302(중단),
# 매경·서울경제·전자신문은 403/404였다.
FEEDS = [
    "https://www.yna.co.kr/rss/market.xml",
    "https://www.yna.co.kr/rss/economy.xml",
    "https://www.yna.co.kr/rss/industry.xml",
    "https://www.hankyung.com/feed/finance",
]

_USER_AGENT = "Mozilla/5.0 (compatible; eric-chatagent/1.0)"


def fetch_feed(url: str, timeout: float = 12.0) -> list[dict]:
    """피드 하나를 받아 기사 목록으로 돌려준다. 실패하면 빈 리스트.

    반환 원소는 네이버 검색 결과와 같은 키를 쓴다. 그래야
    :func:`model.news.collect.extract_and_chunk` 를 그대로 재사용할 수 있다.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except Exception as e:
        print(f"[RSS] 피드 수집 실패: {url} / {e}", flush=True)
        return []

    try:
        root = ElementTree.fromstring(raw)
    except Exception as e:
        print(f"[RSS] 피드 파싱 실패: {url} / {e}", flush=True)
        return []

    items = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "description": (node.findtext("description") or "").strip(),
            "pubDate": (node.findtext("pubDate") or "").strip(),
            "originallink": link,
            "link": link,
            "feed": url,
        })

    return items


def fetch_all_feeds(feeds: list[str] | None = None) -> list[dict]:
    """모든 피드를 받아 합친다. 죽은 피드는 건너뛰고, 같은 기사는 한 번만 남긴다.

    연합뉴스 market/economy/industry 는 같은 기사를 공유한다. 중복을 남기면
    코퍼스 id(링크 해시 기반)가 한 배치 안에서 충돌해 Chroma upsert 가
    터진다. 실제 수집에서 410건 중 31건이 충돌했다.
    """
    collected = []
    seen = set()
    duplicates = 0

    for url in feeds if feeds is not None else FEEDS:
        items = fetch_feed(url)
        print(f"[RSS] {url} -> {len(items)}건", flush=True)
        for item in items:
            key = item.get("link") or item.get("originallink") or item.get("title")
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            collected.append(item)

    if duplicates:
        print(f"[RSS] 피드 간 중복 {duplicates}건 제거", flush=True)
    return collected


def tag_companies(items: list[dict], companies: list[dict]) -> list[dict]:
    """제목에서 종목을 찾아 메타데이터로 붙인다. 못 찾은 기사는 버린다.

    상시 코퍼스는 종목 메타데이터로 1차 필터한 뒤 임베딩 검색을 하는 용도다.
    종목이 없는 기사("코스피 강보합 마감")는 그 필터에 걸리지 않으므로
    코퍼스만 키운다.

    별칭 매칭은 종목 식별과 같은 경계 규칙(:func:`_alias_in_text`)을 쓴다.
    ASCII 별칭에 단어경계를 요구하지 않으면 "earnings" 가 GS로 잡힌다.
    """
    # 긴 별칭을 먼저 보게 정렬한다. "삼성전자"가 "삼성"보다 앞서야 한다.
    indexed = sorted(
        ((alias, c) for c in companies for alias in c.get("aliases", [])),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    tagged = []
    for item in items:
        lowered = item.get("title", "").lower()
        for alias, company in indexed:
            if _alias_in_text(alias.lower(), lowered):
                enriched = dict(item)
                enriched["metadata"] = {
                    **item.get("metadata", {}),
                    "company": company["canonical_name"],
                    "ticker": company.get("ticker") or "",
                }
                tagged.append(enriched)
                break

    print(f"[RSS] 종목 태깅: 전체 {len(items)}건 → {len(tagged)}건", flush=True)
    return tagged
