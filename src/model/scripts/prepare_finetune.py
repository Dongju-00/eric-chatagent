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
PERSONA_PATH = RAW_DIR / "nlpbada" / "korean-persona-chat-dataset-2.csv"
NEWS_TRAIN_PATH = RAW_DIR / "news" / "news_train.csv"
NEWS_VAL_PATH = RAW_DIR / "news" / "news_val.csv"

TRAIN_PATH = OUT_DIR / "qa_train.csv"
VAL_PATH = OUT_DIR / "qa_val.csv"

GENERAL_RATIO = 0.4
RAG_RATIO = 0.6

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


def load_persona(rows):
    if not PERSONA_PATH.exists():
        print(f"skip: {PERSONA_PATH} 없음")
        return

    df = pd.read_csv(PERSONA_PATH)
    count = 0

    for _, row in df.iterrows():
        dialog_text = row.get("session_dialog", "")

        try:
            dialog = ast.literal_eval(dialog_text)
        except Exception:
            continue

        if not isinstance(dialog, list):
            continue

        dialog = [clean_text(x) for x in dialog if clean_text(x)]

        for i in range(len(dialog) - 1):
            question = dialog[i]
            answer = dialog[i + 1]

            if question and answer:
                rows.append({
                    "question": question,
                    "answer": answer,
                })
                count += 1

    print(f"loaded persona pairs: {count:,}")

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


def main():
    general_rows = []
    news_rows = []

    load_smalltalk(general_rows)
    load_persona(general_rows)
    load_news(news_rows)

    random.seed(42)
    random.shuffle(general_rows)
    random.shuffle(news_rows)

    if news_rows:
        general_target = int(len(news_rows) * GENERAL_RATIO / RAG_RATIO)
        general_target = min(len(general_rows), general_target)

        mixed_rows = general_rows[:general_target] + news_rows
        print(f"mixed general: {general_target:,}")
        print(f"mixed rag: {len(news_rows):,}")
    else:
        mixed_rows = general_rows
        print("RAG 데이터가 없어서 일반 QA 데이터만 사용합니다.")

    split_idx = int(len(mixed_rows) * 0.9)
    train_rows = mixed_rows[:split_idx]
    val_rows = mixed_rows[split_idx:]

    save_csv(TRAIN_PATH, train_rows)
    save_csv(VAL_PATH, val_rows)

    print(f"saved train: {TRAIN_PATH} ({len(train_rows):,})")
    print(f"saved val: {VAL_PATH} ({len(val_rows):,})")


if __name__ == "__main__":
    main()