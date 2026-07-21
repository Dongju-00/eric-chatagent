import torch

from model.train.model import device, block_size


def _qa_stop_tokens(sp):
    """줄바꿈으로 끝나는 piece가 나오면 '답변: ...' 한 줄이 끝난 것으로 본다."""
    return {
        i for i in range(sp.get_piece_size())
        if sp.id_to_piece(i).endswith("\n")
    }


def _sentence_end_stop_tokens(sp):
    """'.', '?', '!'로 끝나는 piece가 나오면 한 문장이 끝난 것으로 본다."""
    return {
        i for i in range(sp.get_piece_size())
        if sp.id_to_piece(i) and sp.id_to_piece(i)[-1] in ".?!"
    }


@torch.no_grad()
def generate_reply_rag(model, sp, question: str, contexts: list[str], max_new_tokens: int = 120) -> str:
    stop_tokens = _qa_stop_tokens(sp)

    context_text = "\n\n".join(contexts)

    prompt = f"참고 뉴스: {context_text}\n질문: {question}\n답변: "

    ids = sp.encode(prompt, out_type=int)

    # 모델 block_size가 256이라 너무 길면 뒤쪽만 사용
    if len(ids) > block_size:
        ids = ids[-block_size:]

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens,
        stop_tokens=stop_tokens,
        temperature=0.2,
        top_k=1,
        repetition_penalty=1.2,
    )[0].tolist()

    return sp.decode(out[len(ids):]).strip()

