"""기존 단일턴 뉴스 QA를 멀티턴으로 재구성한다. LLM을 쓰지 않는다.

`prepare_multiturn_news.py` 는 신규 기사를 수집하고 Gemini로 대화를
저술한다. 이 스크립트는 **이미 있는 QA만 재배열**하므로 비용이 0이다.

품질 상한이 낮다는 것을 먼저 알아야 한다.

- 컨텍스트 전문으로 묶으면 1,623 그룹이고 대부분 QA가 1개다.
  2턴 이상 만들 수 있는 것은 452개 대화 / 1,110턴이다.
- 제목으로 묶으면 180 그룹으로 커지지만, 같은 제목에 **발췌 본문이 최대
  6종** 섞여 있다(본문 1종인 제목은 14%뿐). 제목으로 묶어 이어붙이면 다른
  발췌의 답변이 선택한 컨텍스트에 없는 근거를 인용하게 된다. 설계 문서가
  실패 조건으로 둔 "근거 없는 생성 유입"이다. 그래서 **컨텍스트 전문으로만
  묶는다.**
- 기존 데이터는 단일턴용으로 만들어져 턴 간 연결이 설계되지 않았다. 같은
  질문이 반복되거나 답변이 서로 모순되는 경우가 있다.

따라서 이 산출물은 "멀티턴 학습이 먹히는지" 확인하는 용도다. 지시대명사
해소가 되지 않으면 `prepare_multiturn_news.py` 로 데이터를 저술해야 한다.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

try:  # 직접 실행 / -m 실행 양쪽 지원
    from _bootstrap import ensure_src_on_path
except ImportError:
    from model.scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from model.paths import DATA_DIR, SP_PREFIX          # noqa: E402
from model.train.model import block_size             # noqa: E402
from model.train.sp_tokenizer import load_sp         # noqa: E402

DEFAULT_SOURCE = DATA_DIR / "news/qa_news_train_merged.csv"
DEFAULT_OUT_DIR = DATA_DIR / "news"
MIN_TURNS = 2


def split_question(raw: str) -> tuple[str, str]:
    """`참고 뉴스:...질문: X` 를 (컨텍스트, 질문) 으로 나눈다."""
    text = str(raw)
    marker = text.rfind("\n질문:")
    if marker == -1:
        return "", text.strip()
    context = text[:marker]
    context = re.sub(r"^참고 뉴스:\s*", "", context).strip()
    return context, text[marker:].replace("\n질문:", "", 1).strip()


def build_conversations(rows, sp, max_tokens=block_size):
    """컨텍스트가 **완전히 같은** QA만 한 대화로 묶는다."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for context, question, answer in rows:
        if context and question and answer:
            grouped.setdefault(context, []).append((question, answer))

    conversations, skipped_single, skipped_dup = [], 0, 0
    for context, qas in grouped.items():
        head = f"참고 뉴스:\n{context}\n\n"

        # 같은 질문이 반복되는 행이 있다. 첫 번째만 남긴다.
        seen, unique = set(), []
        for q, a in qas:
            if q in seen:
                skipped_dup += 1
                continue
            seen.add(q)
            unique.append((q, a))

        kept = []
        for q, a in unique:
            trial = kept + [(q, a)]
            serialized = head + "".join(f"질문: {q}\n답변: {a}\n" for q, a in trial)
            if len(sp.encode(serialized, out_type=int)) > max_tokens:
                break
            kept = trial

        if len(kept) >= MIN_TURNS:
            conversations.append((context, kept))
        else:
            skipped_single += 1

    return conversations, skipped_single, skipped_dup


def write_csv(path: Path, conversations, id_offset: int = 0) -> int:
    """대화를 CSV 로 쓴다.

    ``id_offset`` 은 train/val 이 같은 ``conversation_id`` 를 쓰지 않게 하기
    위한 것이다. 파일마다 0부터 번호를 다시 매기면 두 파일에 같은 id 가 생겨
    노트북의 겹침 감사가 실패하고, 두 파일을 합쳤을 때 대화가 섞인다.
    """
    turns = 0
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conversation_id", "turn_index", "context", "question", "answer"]
        )
        writer.writeheader()
        for local_id, (context, qas) in enumerate(conversations):
            cid = id_offset + local_id
            for idx, (q, a) in enumerate(qas):
                writer.writerow({
                    "conversation_id": f"free_{cid:05d}",
                    "turn_index": idx,
                    # 첫 턴에만 컨텍스트를 둔다 (generate 산출물과 같은 모양)
                    "context": context if idx == 0 else "",
                    "question": q,
                    "answer": a,
                })
                turns += 1
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_path = args.output_dir / "qa_news_multiturn_free_train.csv"
    val_path = args.output_dir / "qa_news_multiturn_free_val.csv"
    for path in (train_path, val_path):
        if path.exists():
            raise FileExistsError(f"{path} 가 이미 있다. 덮어쓰지 않는다.")

    sp = load_sp(SP_PREFIX)

    rows = []
    with args.source.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            context, question = split_question(row.get("question", ""))
            rows.append((context, question, (row.get("answer") or "").strip()))

    conversations, skipped_single, skipped_dup = build_conversations(rows, sp)
    if not conversations:
        print("2턴 이상 묶을 수 있는 대화가 없다.")
        return 1

    # 컨텍스트 단위로 나눈다. 같은 컨텍스트가 train/val 양쪽에 가지 않는다.
    split = int(len(conversations) * (1 - args.val_ratio))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_train = write_csv(train_path, conversations[:split])
    # val 은 train 다음 번호부터 매긴다. 두 파일의 id 가 겹치면 안 된다.
    n_val = write_csv(val_path, conversations[split:], id_offset=split)

    print(f"원본 QA {len(rows):,}행")
    print(f"중복 질문으로 건너뜀 {skipped_dup:,}행")
    print(f"2턴을 못 채운 컨텍스트 {skipped_single:,}개")
    print()
    print(f"대화 {len(conversations):,}개 / 턴 {n_train + n_val:,}개")
    print(f"  train {split:,} 대화 / {n_train:,} 턴 -> {train_path}")
    print(f"  val   {len(conversations)-split:,} 대화 / {n_val:,} 턴 -> {val_path}")
    print()
    print("주의: 이 데이터는 앞 턴을 참조하는 질문이 설계되지 않았다.")
    print("지시대명사 해소가 되지 않으면 prepare_multiturn_news.py 로 저술해야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
