"""
KoreanGPT_news.pt 추가 파인튜닝

실행:
    uv run python finetune_news.py

주의:
    - 모델 내부 loss는 shift를 하지 않으므로, 여기서 직접 shift하여 손실을 계산한다.
    - block_size 등 하이퍼파라미터는 model.py의 전역값을 그대로 사용한다.
"""

import math
import random
import sys
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model.train.model import KoreanGPT, block_size   # noqa: E402

CONFIG = {
    "sp_model":  PROJECT_ROOT / "src/model/data/sp_korean.model",
    "ckpt_in":   PROJECT_ROOT / "src/model/checkpoints/KoreanGPT_news.pt",
    "ckpt_out":  PROJECT_ROOT / "src/model/checkpoints/KoreanGPT_news_v2.pt",
    "train_csv": PROJECT_ROOT / "src/model/data/news/qa_news_train_merged.csv",
    "val_csv":   PROJECT_ROOT / "src/model/data/news/qa_news_recent_val.csv",

    "batch_size":   8,
    "epochs":       3,
    "lr":           1e-5,     # 추가 파인튜닝이므로 낮게 (기존 지식 보존)
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "grad_clip":    1.0,
    "seed":         42,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


class QADataset(Dataset):
    """
    CSV의 question 컬럼에 '참고 뉴스:\\n...\\n\\n질문: ...' 이 통째로 들어있다.
    여기에 '\\n답변: ' + answer 를 이어붙여 학습 시퀀스를 만든다.
    손실은 답변 토큰에만 적용한다 (프롬프트는 -100 마스킹).
    """

    def __init__(self, csv_path, sp):
        df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(subset=["question", "answer"])
        self.rows = df.to_dict("records")
        self.sp = sp
        self.pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
        self.eos_id = sp.eos_id() if sp.eos_id() >= 0 else None
        self.n_truncated = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        prompt = f"{str(row['question']).strip()}\n답변: "
        answer = str(row["answer"]).strip()

        p_ids = self.sp.encode(prompt)
        a_ids = self.sp.encode(answer)
        if self.eos_id is not None:
            a_ids = a_ids + [self.eos_id]

        # 길이 초과 시 프롬프트 앞부분을 잘라 답변을 보존
        if len(p_ids) + len(a_ids) > block_size:
            over = len(p_ids) + len(a_ids) - block_size
            p_ids = p_ids[over:]
            self.n_truncated += 1

        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + list(a_ids)

        pad_len = block_size - len(ids)
        ids = ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return (
            torch.tensor(ids[:block_size], dtype=torch.long),
            torch.tensor(labels[:block_size], dtype=torch.long),
        )


def compute_loss(model, x, y):
    """
    모델 내부 loss는 shift를 하지 않으므로 여기서 직접 처리한다.
    위치 t의 logits로 t+1 토큰을 예측.
    """
    logits, _ = model(x)                      # targets 없이 호출 → logits만 사용
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = y[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        total += compute_loss(model, x.to(DEVICE), y.to(DEVICE)).item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    print(f"장치: {DEVICE} | block_size: {block_size}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(CONFIG["sp_model"]))
    vocab_size = sp.get_piece_size()
    print(f"어휘 크기: {vocab_size}")

    train_ds = QADataset(CONFIG["train_csv"], sp)
    val_ds = QADataset(CONFIG["val_csv"], sp)
    print(f"학습 {len(train_ds)}건 / 검증 {len(val_ds)}건")

    train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=CONFIG["batch_size"])

    # 모델 로드
    model = KoreanGPT(vocab_size)
    state = torch.load(CONFIG["ckpt_in"], map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE)
    model.train()
    print(f"파라미터: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    total_steps = len(train_dl) * CONFIG["epochs"]
    warmup = max(int(total_steps * CONFIG["warmup_ratio"]), 1)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n학습 전 검증 손실: {evaluate(model, val_dl):.4f}")
    best_val = float("inf")

    for epoch in range(1, CONFIG["epochs"] + 1):
        running = 0.0
        for i, (x, y) in enumerate(train_dl, 1):
            loss = compute_loss(model, x.to(DEVICE), y.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running += loss.item()
            if i % 20 == 0:
                print(f"  epoch {epoch} | {i}/{len(train_dl)} | "
                      f"loss {running/20:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                running = 0.0

        val_loss = evaluate(model, val_dl)
        print(f"[epoch {epoch}] 검증 손실: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CONFIG["ckpt_out"])
            print(f"  → 저장: {CONFIG['ckpt_out'].name}")
        else:
            print("  → 개선 없음, 저장 생략 (과적합 신호)")

    if train_ds.n_truncated:
        print(f"\n길이 초과로 프롬프트가 잘린 샘플: {train_ds.n_truncated}건")
    print(f"\n완료. 최종 검증 손실: {best_val:.4f}")


if __name__ == "__main__":
    main()
