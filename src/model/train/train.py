"""
진입점. 실행 모드 4가지:
  python train.py train      -> Stage 1 사전학습 (챗봇 데이터로 "이어쓰기" 학습)
  python train.py train_qa   -> Stage 2 파인튜닝 (챗봇 Q&A 포맷으로 추가 학습)
  python train.py chat       -> Stage 1 모델과 "이어쓰기" 대화
  python train.py chat_qa    -> Stage 2 모델과 "질문-답변" 대화
"""
import os
import sys
import torch
from pathlib import Path

from model.rag.generator import generate_reply_rag
from model.train.model import device, steps, lr, KoreanGPT
from model.train.tokenizer import load_korean_chatbot_data, load_qa_pairs
from model.train.sp_tokenizer import train_sp, load_sp, encode
from model.train.train_utils import make_batcher, train_loop, make_qa_batcher
from model.store.vector_store import ChromaNewsVectorStore

SP_VOCAB_SIZE = 16000
BASE_DIR = Path(__file__).resolve().parents[1]

SP_PREFIX = str(BASE_DIR / "data" / "sp_korean")    # data/sp_korean.model / .vocab 으로 생성됨

CKPT_DIR = BASE_DIR / "checkpoints"
GEN_CKPT = str(CKPT_DIR / "KoreanGPT.pt")      # Stage 1 가중치
# QA_CKPT = str(CKPT_DIR / "KoreanGPT_qa.pt")    # Stage 2 가중치  # 파인튜닝을 두 번 진행해서 최적화 된 모델 2개를 만듦
SMALLTALK_CKPT = str(CKPT_DIR / "KoreanGPT_smalltalk.pt")
NEWS_CKPT = str(CKPT_DIR / "KoreanGPT_news.pt")

EARLY_STOP_PATIENCE = 15     # 연속 10번 평가(=200 step) 동안 개선이 없으면 중단
EARLY_STOP_MIN_DELTA = 1e-4

FT_STEPS = 2000
FT_LR = 1e-4                 # Stage 1보다 낮은 lr로 미세조정 (급격한 망각 방지)

def pretrain_path() -> str:
    return str(BASE_DIR / "data" / "processed" / "pretrain.txt")

def build_pretrain_text() -> str:
    return load_korean_chatbot_data()


os.makedirs(BASE_DIR / "data", exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

if os.path.exists(f"{SP_PREFIX}.model"):
    sp = load_sp(SP_PREFIX)
else:
    sp = train_sp(pretrain_path(), SP_PREFIX, SP_VOCAB_SIZE)

vocab_size = sp.get_piece_size()


def train_stage1(model):
    token_cache = "model/data/processed/pretrain_tokens.pt"

    if os.path.exists(token_cache):
        print(f"loading token cache: {token_cache}", flush=True)
        data = torch.load(token_cache, map_location="cpu")
    else:
        print("loading pretrain text...", flush=True)
        text = build_pretrain_text()
        print(f"loaded text: {len(text):,} chars", flush=True)

        print("encoding pretrain text...", flush=True)
        ids = encode(text, sp)
        print(f"encoded tokens: {len(ids):,}", flush=True)

        data = torch.tensor(ids, dtype=torch.long)

        print(f"saving token cache: {token_cache}", flush=True)
        torch.save(data, token_cache)

    print(f"tokens: {len(data):,}, vocab_size: {vocab_size}, device: {device}", flush=True)

    n_train = int(0.9 * len(data))
    get_batch = make_batcher(data[:n_train], data[n_train:])

    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, device={device}", flush=True)

    train_loop(
        model,
        get_batch,
        steps,
        lr,
        GEN_CKPT,
        EARLY_STOP_PATIENCE,
        EARLY_STOP_MIN_DELTA,
        eval_interval=100,
        eval_iters=10,
    )

def _finetune(model, prefix, ckpt_path):
    train_pairs = load_qa_pairs(root_dir=None, split="train", prefix=prefix)
    val_pairs = load_qa_pairs(root_dir=None, split="val", prefix=prefix)
    print(f"{prefix} — train {len(train_pairs):,}, val {len(val_pairs):,}")

    get_batch = make_qa_batcher(train_pairs, val_pairs, sp)

    if not os.path.exists(GEN_CKPT):
        raise FileNotFoundError(f"{GEN_CKPT} 없음. Colab에서 사전학습한 것을 넣으세요.")
    model.load_state_dict(torch.load(GEN_CKPT, map_location=device))   # 기본에서 출발

    train_loop(model, get_batch, FT_STEPS, FT_LR, ckpt_path,
               EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
               eval_interval=50, eval_iters=10)

def train_smalltalk(model):
    _finetune(model, "qa_smalltalk", SMALLTALK_CKPT)

def train_news(model):
    _finetune(model, "qa_news", NEWS_CKPT)



MODES = {
    "train": train_stage1,
    "train_smalltalk": train_smalltalk,
    "train_news": train_news,
}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode not in MODES:
        sys.exit(f"알 수 없는 모드: {mode} (다음 중 하나를 쓰세요: {', '.join(MODES)})")

    model = KoreanGPT(vocab_size).to(device)
    MODES[mode](model)

