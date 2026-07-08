import torch

from model.train.model import block_size, batch_size, device


def make_batcher(train_data, val_data):
    """train_data/val_data(1D LongTensor)에서 (x, y) 배치를 뽑는 get_batch 함수를 만든다."""
    def get_batch(split="train"):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size - 1, (batch_size,))
        x = torch.stack([d[i: i + block_size] for i in ix])
        y = torch.stack([d[i + 1: i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
    return get_batch


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters=10):
    model.eval()

    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def train_loop(model, get_batch, steps, lr, ckpt_path, patience, min_delta, eval_interval=100, eval_iters=10):
    """early stopping(patience/min_delta) 적용 학습 루프.

    val loss가 patience번 평가(=patience*20 step) 동안 min_delta 이상 줄지 않으면
    조기 종료하고, 지금까지 가장 좋았던 가중치를 ckpt_path에 저장한다.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0

    for step in range(steps):
        x, y = get_batch("train")
        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_interval == 0 or step == steps - 1:
            est = estimate_loss(model, get_batch, eval_iters)
            print(f"step {step:5d}  train {est['train']:.3f}  val {est['val']:.3f}", flush=True)

            if est["val"] < best_val - min_delta:
                best_val = est["val"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                torch.save(best_state, ckpt_path)
                print(f"saved best weights (val {best_val:.3f}) to {ckpt_path}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"early stopping at step {step} (best val {best_val:.3f})")
                    break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"loaded best weights (val {best_val:.3f}) from {ckpt_path}")
    return best_val


def make_qa_batcher(train_examples, val_examples, sp):
    """
    QA 파인튜닝용 배처.
    examples는 (question, answer) 문자열 쌍이다.
    여기서 SentencePiece로 토큰화한 뒤, 답변 부분에만 loss를 건다.
    """
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0

    def encode_example(question, answer):
        prompt = f"{question}\n답변: "
        full_text = f"{prompt}{answer}\n"

        prompt_ids = sp.encode(prompt, out_type=int)
        ids = sp.encode(full_text, out_type=int)

        return ids, len(prompt_ids)

    def build_batch(examples):
        ix = torch.randint(len(examples), (batch_size,))

        xs = []
        ys = []

        for i in ix:
            question, answer = examples[i.item()]
            ids, prompt_len = encode_example(question, answer)

            ids = ids[:block_size + 1]

            if len(ids) < 2:
                continue

            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)

            mask_until = min(prompt_len - 1, len(y))
            y[:mask_until] = -100

            pad_len = block_size - len(x)

            if pad_len > 0:
                x = torch.cat([
                    x,
                    torch.full((pad_len,), pad_id, dtype=torch.long)
                ])
                y = torch.cat([
                    y,
                    torch.full((pad_len,), -100, dtype=torch.long)
                ])

            xs.append(x)
            ys.append(y)

        return torch.stack(xs).to(device), torch.stack(ys).to(device)

    def get_batch(split="train"):
        examples = train_examples if split == "train" else val_examples
        return build_batch(examples)

    return get_batch