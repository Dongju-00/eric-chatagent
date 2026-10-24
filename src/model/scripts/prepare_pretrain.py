from pathlib import Path

from datasets import load_dataset

try:  # 직접 실행 / -m 실행 양쪽 지원
    from _bootstrap import ensure_src_on_path
except ImportError:
    from model.scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from model.paths import DATA_DIR  # noqa: E402

# 사전학습을 위한 데이터셋 구성하기 위해
# Hugging Face에서 KOREAN-WEBTEXT + namuwiki dataset 가지고 와서 txt 파일로 변환 후 합침
_PROCESSED = DATA_DIR / "processed"
OUT1 = _PROCESSED / "pretrain1.txt"
OUT2 = _PROCESSED / "pretrain2.txt"
OUT_FINAL = _PROCESSED / "pretrain.txt"

WEBTEXT = 400_000_000
NAMUWIKI = 500_000_000


def write_dataset_to_txt(dataset_name: str, output_path: Path, max_chars: int):
    print("loading:", dataset_name)

    ds = load_dataset(dataset_name, split="train", streaming=True)

    total_chars = 0
    total_docs = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()

            if len(text) < 100:
                continue

            if title:
                f.write(title)
                f.write("\n")

            f.write(text)
            f.write("\n\n")

            total_chars += len(text)
            total_docs += 1

            if total_chars >= max_chars:
                break

    print("saved:", output_path)
    print("docs:", total_docs)
    print("chars:", total_chars)
    print()


def merge_pretrain_files():
    import random
    docs = []
    for p in (OUT1, OUT2,):
        text = p.read_text(encoding="utf-8")
        docs.extend(d for d in text.split("\n\n") if len(d.strip()) > 50)

    random.seed(42)
    random.shuffle(docs)          # 두 소스가 골고루 섞이게

    OUT_FINAL.write_text("\n\n".join(docs), encoding="utf-8")
    print(f"merged: {len(docs):,} docs, {OUT_FINAL.stat().st_size/1e9:.2f} GB")


def main():
    _PROCESSED.mkdir(parents=True, exist_ok=True)

    write_dataset_to_txt(
        dataset_name="HAERAE-HUB/KOREAN-WEBTEXT",
        output_path=OUT1,
        max_chars=WEBTEXT,
    )

    write_dataset_to_txt(
        dataset_name="heegyu/namuwiki-extracted",
        output_path=OUT2,
        max_chars=NAMUWIKI,
    )

    merge_pretrain_files()


if __name__ == "__main__":
    main()
