import ast
import csv
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_hf"
OUT_DIR = ROOT / "data" / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SMALLTALK_PATH = RAW_DIR / "smalltalk" / "smalltalk_all.csv"
NEWS_TRAIN_PATH = RAW_DIR / "news" / "news_train.csv"
NEWS_VAL_PATH = RAW_DIR / "news" / "news_val.csv"

TRAIN_PATH = OUT_DIR / "qa_train.csv"
VAL_PATH = OUT_DIR / "qa_val.csv"

def clean_text(text):
    return str(text).replace("\n", " ").strip()


def load_smalltalk(rows):
    if not SMALLTALK_PATH.exists():
        print(f"skip: {SMALLTALK_PATH} 없음")
        return

    df = pd.read_csv(SMALLTALK_PATH)

    for _, row in df.iterrows():
        question = clean_text(row.get("question", ""))
        answer = clean_text(row.get("answer", ""))

        if question and answer:
            rows.append({
                "question": question,
                "answer": answer,
            })

    print(f"loaded smalltalk: {len(df):,}")

def load_news(rows):
    total = 0

    for path in [NEWS_TRAIN_PATH, NEWS_VAL_PATH]:
        if not path.exists():
            print(f"skip:{path} 없음")
            continue

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()

                if question and answer:
                    rows.append({
                        "question": question,
                        "answer": answer,
                    })
                    total += 1

    print(f"loaded rag pairs: {total:,}")

def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(rows)

# smalltalk과 stock_news 부분 따로 저장하기 위해서 나눔
def split_and_save(rows, prefix):
    """rows를 9:1로 나눠 {prefix}_train.csv / {prefix}_val.csv로 저장"""
    if not rows:
        print(f"skip: {prefix} 데이터 없음")
        return
    random.shuffle(rows)
    split_idx = int(len(rows) * 0.9)
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    train_path = OUT_DIR / f"{prefix}_train.csv"
    val_path = OUT_DIR / f"{prefix}_val.csv"
    save_csv(train_path, train_rows)
    save_csv(val_path, val_rows)
    print(f"{prefix}: train {len(train_rows):,}, val {len(val_rows):,}")


def main():
    general_rows = []
    news_rows = []

    load_smalltalk(general_rows)
    load_news(news_rows)

    random.seed(42)

    # 섞지 않고 각각 따로 저장
    split_and_save(general_rows, "qa_smalltalk")   # → qa_smalltalk_train.csv / _val.csv
    split_and_save(news_rows, "qa_news")            # → qa_news_train.csv / _val.csv

if __name__ == "__main__":
    main()