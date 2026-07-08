import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

def load_korean_chatbot_data(root_dir: str = "data") -> str:
    path = Path(root_dir) / "processed" / "pretrain.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} 파일이 없습니다. scripts/prepare_pretrain.py를 먼저 실행하세요."
        )

    return path.read_text(encoding="utf-8")


def load_qa_pairs(root_dir: str = "korean_chatbot_RAG/src/model/data", split: str = "train"):
    if root_dir is None:
        path = DATA_DIR / "processed" / f"qa_{split}.csv"
    else:
        path = Path(root_dir) / "processed" / f"qa_{split}.csv"

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


def load_chatbot_qa_data(root_dir: str = None   , split: str = "train", shuffle: bool = True, seed: int = 42) -> str:
    pairs = load_qa_pairs(root_dir=root_dir, split=split)

    if shuffle:
        random.Random(seed).shuffle(pairs)

    return "".join(
        f"질문: {question}\n답변: {answer}\n\n"
        for question, answer in pairs
    )