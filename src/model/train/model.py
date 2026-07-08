"""
Decoder-only GPT
Encoder-Decoder가 아니라, 토큰 시퀀스 하나를 받아서 "다음 토큰이 뭘까"를 계속
예측하는 구조입니다. 질문/답변 구분 없이 그냥 긴 텍스트를 이어쓰는 법을 배우고,
generator.py / app.py에서 "질문: ...\n답변: " 까지 입력으로 주고 그 뒤를 생성시키는
방식으로 씁니다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- hyperparameters ---
block_size = 256     # 컨텍스트 윈도우 (한 번에 볼 수 있는 최대 토큰 수)
n_embd = 384         # 임베딩 차원
n_head = 6           # Multi-Head Attention의 head 개수
n_layer = 6          # Transformer block(decoder layer) 개수
batch_size = 16
dropout = 0.2

steps = 50000
lr = 5e-4
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(1337)


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention: softmax(QK^T / sqrt(d) + mask) V."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // n_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / head_dim ** 0.5
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~causal, float("-inf"))  # 미래 토큰을 못 보게 마스킹
        att = self.drop(F.softmax(att, dim=-1))
        out = att @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))


class Block(nn.Module):
    """Transformer decoder block: causal self-attention + feed-forward."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class KoreanGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stop_tokens=None, temperature=1.0, top_k=None, repetition_penalty=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for token_id in set(idx[0, -block_size:].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
            if top_k is not None:
                kth_value = torch.topk(logits, top_k).values[:, -1:]
                logits[logits < kth_value] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_tokens is not None and next_id.item() in stop_tokens:
                break
        return idx
