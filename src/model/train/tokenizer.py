import csv
import random
from pathlib import Path

from model.paths import DATA_DIR

def load_korean_chatbot_data(root_dir: str = None) -> str:
    base = Path(root_dir) / "data" if root_dir else DATA_DIR
    path = base / "processed" / "pretrain.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} 파일이 없습니다. scripts/prepare_pretrain.py를 먼저 실행하세요."
        )

    return path.read_text(encoding="utf-8")

# 파인튜닝 부분을 두 부분으로 나눠놨기 때문에 prefix로 확인
def load_qa_pairs(root_dir: str = None, split: str = "train", prefix: str = "qa"):
    if root_dir is None:
        path = DATA_DIR / "processed" / f"{prefix}_{split}.csv"
    else:
        path = Path(root_dir) / "processed" / f"{prefix}_{split}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} 파일이 없습니다. scripts/prepare_finetune.py를 먼저 실행하세요."
        )

    pairs = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()

            if question and answer:
                pairs.append((question, answer))

    return pairs
