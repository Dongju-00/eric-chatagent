"""서빙 시작 비용을 측정한다.

리소스 중복 적재를 없앤 효과를 수치로 남기기 위한 하네스다. 측정 항목:

    import_torch_load        ``import app.app`` 만으로 일어난 ``torch.load`` 호출 수
    import_persistent_client ``import app.app`` 만으로 만들어진 Chroma 클라이언트 수
    torch_load               ``build_resources()`` 가 일으킨 ``torch.load`` 호출 수
    persistent_client        ``build_resources()`` 가 만든 Chroma 클라이언트 수
    t_build_resources        ``build_resources()`` 소요 시간 (초)
    rss_*                    ``resource.getrusage`` 기준 최대 RSS (MB)

기준선(before)은 스펙에 기록된 변경 전 측정값이다. 변경 전에는 모듈 레벨에서
``baseline.py`` 와 ``graph.py`` 가 각자 스토어와 LLM을 만들었기 때문에
``import app.app`` 하나로 가중치 4벌과 Chroma 클라이언트 2개가 올라갔다.

RSS는 참고용으로만 본다. KoreanGPT 1벌은 fp32 파라미터만 159.5MB
(41,816,704개)지만, macOS의 메모리 압축과 할당자 재사용 때문에 측정된 RSS는
적재 벌수에 비례하지 않는다. 믿을 수 있는 수치는 적재 횟수다.

사용:
    LANGSMITH_TRACING=false PYTHONPATH=src uv run python measurements/measure_startup.py
    LANGSMITH_TRACING=false PYTHONPATH=src uv run python measurements/measure_startup.py --label startup_after

``LANGSMITH_TRACING=false`` 는 유효하지 않은 LangSmith 키 때문에 요청마다
403 로그가 쏟아지는 것을 막기 위한 것이다.
"""

import argparse
import json
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# 변경 전 기준선. 스펙의 측정값이다 (`import app.app` 한 번에 발생).
BEFORE = {
    "import_torch_load": 4,
    "import_persistent_client": 2,
    "note": (
        "모듈 레벨 전역 때문에 import만으로 KoreanGPT.pt / KoreanGPT_news_v2.pt / "
        "KoreanGPT.pt / KoreanGPT_smalltalk.pt 를 적재하고 같은 경로로 Chroma "
        "클라이언트를 2개 만들었다."
    ),
}


def _max_rss_mb() -> float:
    """최대 RSS를 MB로 돌려준다.

    ``ru_maxrss`` 단위는 플랫폼마다 다르다. macOS는 바이트, Linux는 킬로바이트다.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(raw / divisor, 1)


def run(label: str) -> dict:
    import chromadb
    import torch

    counts = {"torch_load": 0, "persistent_client": 0}
    real_load, real_client = torch.load, chromadb.PersistentClient

    def counting_load(*args, **kwargs):
        counts["torch_load"] += 1
        return real_load(*args, **kwargs)

    def counting_client(*args, **kwargs):
        counts["persistent_client"] += 1
        return real_client(*args, **kwargs)

    torch.load = counting_load
    chromadb.PersistentClient = counting_client

    # torch / chromadb 를 이미 적재한 상태의 기준점이다.
    rss_baseline = _max_rss_mb()

    # 1) import 비용. 리소스 생성이 lifespan으로 옮겨졌으므로 0이어야 한다.
    import app.app  # noqa: F401
    from model.resources import build_resources

    import_counts = dict(counts)
    rss_after_import = _max_rss_mb()

    # 2) 실제 리소스 생성 비용.
    counts["torch_load"] = 0
    counts["persistent_client"] = 0
    t0 = time.perf_counter()
    resources = build_resources()
    t_build = time.perf_counter() - t0
    rss_after_build = _max_rss_mb()

    torch.load, chromadb.PersistentClient = real_load, real_client

    return {
        "label": label,
        "measured_at": datetime.now().isoformat(),
        "platform": sys.platform,
        "before": BEFORE,
        "after": {
            "import_torch_load": import_counts["torch_load"],
            "import_persistent_client": import_counts["persistent_client"],
            "torch_load": counts["torch_load"],
            "persistent_client": counts["persistent_client"],
            "t_build_resources": round(t_build, 3),
            "rss_baseline_mb": rss_baseline,
            "rss_after_import_mb": rss_after_import,
            "rss_after_build_mb": rss_after_build,
            "fallback_llm": type(resources.fallback_llm).__name__,
        },
        "rss_note": (
            "RSS는 참고용이다. KoreanGPT 1벌의 fp32 파라미터는 159.5MB지만 "
            "macOS 메모리 압축과 할당자 재사용 때문에 측정값이 적재 벌수에 "
            "비례하지 않는다. 비교 가능한 수치는 적재 횟수다."
        ),
    }


def print_summary(result: dict) -> None:
    before, after = result["before"], result["after"]
    print(f"\n요약 [{result['label']}]")
    print(f"  import 시 torch.load        {before['import_torch_load']} -> {after['import_torch_load']}")
    print(
        f"  import 시 Chroma 클라이언트  {before['import_persistent_client']} -> "
        f"{after['import_persistent_client']}"
    )
    print(f"  build_resources torch.load  {after['torch_load']} (임베더 1 + LLM 2)")
    print(f"  build_resources Chroma      {after['persistent_client']}")
    print(f"  build_resources 소요        {after['t_build_resources']}s")
    print(
        f"  최대 RSS                    {after['rss_baseline_mb']} -> "
        f"{after['rss_after_import_mb']} -> {after['rss_after_build_mb']} MB "
        "(torch 적재 후 / app import 후 / 리소스 생성 후)"
    )


def tensor_bytes(module) -> int:
    """모듈이 들고 있는 파라미터+버퍼의 실제 바이트.

    RSS 와 달리 OS 메모리 회계에 의존하지 않는다. 파라미터 수 x 원소 크기라서
    같은 체크포인트면 어느 플랫폼에서도 같은 값이 나온다. 중복 적재 제거의
    효과를 주장할 수 있는 수치는 이것이다.
    """
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def measure_weight_memory() -> dict:
    """중복 적재 제거로 사라진 가중치 텐서 바이트를 센다."""
    from model.embed.embedder import KoreanGPTEmbedder
    from model.rag.baseline import build_llm

    embedder = tensor_bytes(KoreanGPTEmbedder().model)
    news = tensor_bytes(build_llm("stock_news").model)
    small = tensor_bytes(build_llm("small_talk").model)

    return {
        "embedder_mb": round(embedder / 1e6, 1),
        "news_llm_mb": round(news / 1e6, 1),
        "small_talk_llm_mb": round(small / 1e6, 1),
        # 변경 전에는 임베더가 2벌이었다 (baseline.py + graph.py 각각 1개)
        "before_total_mb": round((embedder * 2 + news + small) / 1e6, 1),
        "after_total_mb": round((embedder + news + small) / 1e6, 1),
        "removed_duplicate_mb": round(embedder / 1e6, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="startup_after", help="결과 파일 이름")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"측정 시작: {args.label}", flush=True)
    result = run(args.label)
    result["weight_memory"] = measure_weight_memory()

    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(result)

    w = result["weight_memory"]
    print()
    print("가중치 텐서 (RSS 와 달리 플랫폼 무관)")
    print(f"  임베더 1벌            {w['embedder_mb']}MB")
    print(f"  변경 전 적재 4벌 합계  {w['before_total_mb']}MB")
    print(f"  변경 후 적재 3벌 합계  {w['after_total_mb']}MB")
    print(f"  제거된 중복           {w['removed_duplicate_mb']}MB")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    sys.exit(main())
