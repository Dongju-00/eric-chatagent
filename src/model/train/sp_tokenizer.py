"""
SentencePiece 학습/로드 래퍼.
train_sp(), load_sp()로 SentencePieceProcessor 객체 하나를 만들어두면
encode/decode를 전부 그 객체로 처리할 수 있습니다.
"""
import os

import sentencepiece as spm

# model.py의 KoreanGPT는 패딩을 쓰지 않는 통짜 텍스트 스트림 방식이라 PAD_ID가
# 실제로 학습에 쓰이진 않지만, 다른 토큰(UNK/BOS/EOS) ID와 안 겹치게 자리만 잡아둡니다.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train_sp(input_path : str, model_prefix : str, vocab_size : int=16000):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=["질문:", "답변:", "\n"],
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )
    print(f"SentencePiece 학습 완료 -> {model_prefix}.model")
    return load_sp(model_prefix)


def load_sp(model_prefix: str):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp


def encode(text: str, sp) -> list:
    return sp.encode(text, out_type=int)


def decode(ids, sp) -> str:
    return sp.decode(ids)
