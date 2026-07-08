from pathlib import Path
from datasets import load_dataset

# 사전학습을 위한 데이터셋 구성하기 위해
# Hugging Face에서 KOREAN-WEBTEXT + namuwiki dataset 가지고 와서 txt 파일로 변환 후 합침
OUT1 = Path("data/processed/pretrain1.txt")
OUT2 = Path("data/processed/pretrain2.txt")
OUT_FINAL = Path("data/processed/pretrain.txt")

OUT1.parent.mkdir(parents=True, exist_ok=True)

MAX_CHARS = 35_000_000


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
    with OUT_FINAL.open("w", encoding="utf-8") as out:
        out.write(OUT1.read_text(encoding="utf-8"))
        out.write("\n\n")
        out.write(OUT2.read_text(encoding="utf-8"))

    print("merged:", OUT_FINAL)
    print("total chars:", len(OUT_FINAL.read_text(encoding="utf-8")))


write_dataset_to_txt(
    dataset_name="HAERAE-HUB/KOREAN-WEBTEXT",
    output_path=OUT1,
    max_chars=MAX_CHARS,
)

write_dataset_to_txt(
    dataset_name="heegyu/namuwiki-extracted",
    output_path=OUT2,
    max_chars=MAX_CHARS,
)

merge_pretrain_files()