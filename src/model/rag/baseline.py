"""직접 학습한 KoreanGPT를 LangChain LLM으로 감싼다.

이 모듈은 래퍼와 팩토리만 둔다. 모듈 레벨에서 인스턴스를 만들면 import만으로
가중치가 적재되므로 그렇게 하지 않는다. 실제 생성은 ``model.resources``의
``build_resources()``가, 프롬프트/체인 조립은 ``model.rag.prompts``가 맡는다.
"""

import os

import torch
from dotenv import load_dotenv
from langchain_core.language_models.llms import LLM
from langchain_google_genai import ChatGoogleGenerativeAI

from model.paths import CHECKPOINT_DIR, SP_PREFIX
from model.train.model import KoreanGPT, block_size, device
from model.train.sp_tokenizer import load_sp

load_dotenv()


class KoreanGPTLLM(LLM):
    """직접 학습한 KoreanGPT를 LangChain LLM으로 래핑"""

    model: object = None
    sp: object = None
    max_new_tokens: int = 80
    stop_ids: object = None

    def __init__(self, model_path, sp_prefix, **kwargs):
        super().__init__(**kwargs)
        self.sp = load_sp(sp_prefix)
        self.model = KoreanGPT(self.sp.get_piece_size()).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        self.stop_ids = {
            i for i in range(self.sp.get_piece_size())
            if self.sp.id_to_piece(i).endswith("\n")
        }

    def _encode_prompt(self, prompt: str) -> list[int]:
        ids = self.sp.encode(prompt, out_type=int)

        if len(ids) <= block_size:
            return ids

        question_marker = "\n질문:"
        marker_index = prompt.rfind(question_marker)

        # RAG 형식이 아닌 일반 프롬프트
        if marker_index == -1:
            return ids[-block_size:]

        # 뉴스 부분과 질문 부분 분리
        context_text = prompt[:marker_index]
        question_text = prompt[marker_index:]

        question_ids = self.sp.encode(
            question_text,
            out_type=int,
        )

        # 질문이 너무 길 경우 질문 뒷부분을 우선 보존
        if len(question_ids) >= block_size:
            return question_ids[-block_size:]

        # 질문을 제외하고 뉴스에 사용할 수 있는 토큰 수
        context_budget = block_size - len(question_ids)

        context_ids = self.sp.encode(
            context_text,
            out_type=int,
        )

        # 뉴스는 앞부분부터 보존
        context_ids = context_ids[:context_budget]

        return context_ids + question_ids

    @property
    def _llm_type(self) -> str:
        return "korean_gpt"

    @torch.no_grad()
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        ids = self._encode_prompt(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        out = self.model.generate(
            idx, self.max_new_tokens,
            stop_tokens=self.stop_ids,
            temperature=0.2,
            top_k=10,
            repetition_penalty=1.3,
        )[0].tolist()

        return self.sp.decode(out[len(ids):]).strip()

def build_llm(model="small_talk"):
    if model == "small_talk":
        return KoreanGPTLLM(
            model_path=CHECKPOINT_DIR / "KoreanGPT_smalltalk.pt",
            sp_prefix=str(SP_PREFIX),
        )
    elif model == "stock_news":
        return KoreanGPTLLM(
            model_path=CHECKPOINT_DIR / "KoreanGPT_news_v2.pt",
            sp_prefix=str(SP_PREFIX),
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
