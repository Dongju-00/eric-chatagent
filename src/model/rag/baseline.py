import os
from pathlib import Path
import torch
from dotenv import load_dotenv

from model.train.model import KoreanGPT, device, block_size
from model.train.sp_tokenizer import load_sp

from model.store.vector_store import ChromaNewsVectorStore

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.language_models.llms import LLM


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"

store = ChromaNewsVectorStore()

def retriever(question: str) -> str:
    result = store.search_similar_news(question, top_k=5)
    docs = result["documents"][0]
    seen, uniq = set(), []
    for d in docs:
        key = d[:60]                  # 앞부분으로 중복 판단
        if key not in seen:
            seen.add(key); uniq.append(d)
    return "\n\n".join(uniq[:3])

class KoreanGPTLLM(LLM):
    """직접 학습한 KoreanGPT를 LangChain LLM으로 래핑"""

    model: object = None
    sp: object = None
    max_new_tokens: int = 120
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
        # if len(ids) > block_size:
        #     ids = ids[-block_size:]        # block_size 256 초과 시 뒤쪽만

        idx = torch.tensor([ids], dtype=torch.long, device=device)

        out = self.model.generate(
            idx, self.max_new_tokens,
            stop_tokens=self.stop_ids,
            temperature=0.2,  # 0.2 → 0.8
            top_k=5,  # 1 → 40
            repetition_penalty=1.3,
        )[0].tolist()

        return self.sp.decode(out[len(ids):]).strip()

def build_llm(model="small_talk"):
    if model == "small_talk":
        return KoreanGPTLLM(
        model_path = MODEL_DIR / "KoreanGPT_smalltalk.pt",
        sp_prefix=str(DATA_DIR / "sp_korean"),
        )
    elif model == "stock_news":
        return KoreanGPTLLM(
            model_path = MODEL_DIR / "KoreanGPT_news.pt",
            sp_prefix=str(DATA_DIR / "sp_korean"),
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

llm = build_llm("stock_news")

prompt = PromptTemplate.from_template(
    "참고 뉴스: {context}\n질문: {question}\n답변: "
)

rag = (
    {"context" : RunnableLambda(retriever), "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    questions = [
        "카카오 주가 전망 어때?",
        "환율이 주식에 미치는 영향은?",
        "반도체 업황은 지금 어떤 상황이야?",
        "요즘 증시 분위기 어때?",
        "금리 인상이 주식에 어떤 영향을 줘?",
    ]
    for q in questions:
        store.build_news_vector_db(q)
        ctx = retriever(q)                    # 검색된 뉴스
        ans = rag.invoke(q)

        print(f"\n{'='*60}")
        print(f"[질문] {q}")
        print(f"[검색된 뉴스]\n{ctx[:500]}")   # 앞부분만
        print(f"[답변]\n{ans}")