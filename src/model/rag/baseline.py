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

# query = "안녕 반가워"
# query_id = store.build_news_vector_db(query)

# RAG
# print("RAG 파이프라인 시작")

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

    @property
    def _llm_type(self) -> str:
        return "korean_gpt"

    @torch.no_grad()
    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        ids = self.sp.encode(prompt, out_type=int)
        if len(ids) > block_size:
            ids = ids[-block_size:]        # block_size 256 초과 시 뒤쪽만

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = self.model.generate(
            idx, self.max_new_tokens,
            stop_tokens=self.stop_ids,
            temperature=0.8,  # 0.2 → 0.8
            top_k=40,  # 1 → 40
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

# graph.py에서 import 될 때 마다 실행돼서 토큰을 잡아먹음
"""
print(rag.invoke("투자 판단을 바로 내려도 돼?"))
print("RAG 파이프라인 완료")

# 평가
from langsmith.evaluation import evaluate
from langsmith import Client

client = Client()
DATASET_NAME = "korean-news-eval"

EVAL_QUESTIONS = [
    {
        "question": "투자 판단을 바로 내려도 돼?",
        "answer":   "참고 뉴스만으로 투자 판단을 바로 내리기는 어렵습니다. 뉴스에 나온 요인만 확인할 수 있으며, 실제 투자는 추가 정보와 리스크를 함께 검토해야 합니다."
                    "참고 뉴스: 제목: 네이버파이낸셜·두나무 주식 교환 일정 연기 내용: 네이버파이낸셜과 두나무의 포괄적 주식 교환 일정이 연말로 다시 연기되었다. 시장에서는 양사의 결합 시너지와 규제 변수에 대한 불확실성이 함께 거론되고 있다.",
    },
    {
        "question": "안녕 반가워,",
        "answer":  "안녕, 오늘 기분은 어때?",
    },
    {
        "question": "잘했다고 해줘",
        "answer":   "잘했어. 진짜 고생 많았어.",
    },
    {
        "question": "너는 뭐야?",
        "answer":   "나는 아직 배우는 중이지만, 편하게 이야기 나눌 수 있어.",
    },
    {
        "question": "리만 적분 가능한 함수는 어떤 특징을 가지나요?",
        "answer":   "리만 적분 가능한 함수는 구간 내에서 유한 개의 불연속점을 가진 함수로, 리만 합을 통해 적분값을 정의할 수 있습니다.",
    },
]

existing = [d for d in client.list_datasets(dataset_name=DATASET_NAME)]

inputs  = [{"question": ex["question"]} for ex in EVAL_QUESTIONS]
outputs = [{"answer":   ex["answer"]}   for ex in EVAL_QUESTIONS]

if existing:
    dataset = existing[0]
    print(f"기존 Dataset 사용: {dataset.id}")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="답변 평가 용",
    )
    print(f"새 Dataset 생성: {dataset.id}")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=inputs,
        outputs=outputs,
    )
    print(f"Example {len(EVAL_QUESTIONS)}건 추가 완료")

loaded = client.read_dataset(dataset_name=DATASET_NAME)

examples = list(client.list_examples(dataset_id=loaded.id))

for ex in examples[:3]:
    print("Q:", ex.inputs["question"])
    print("A:", ex.outputs["answer"] if ex.outputs else "(없음)")
    print()

def target(inputs):
    return {"answer" : rag.invoke(inputs["question"])}

def contains_expected_keyword(run, example):
    pred = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")

    keywords = [w for w in expected.split() if len(w) >= 2][:2]
    hit = all(k in pred for k in keywords)

    return {
        "key": "contains_expected_keyword",
        "score": 1 if hit else 0,
        "comment": f"필수 키워드 {keywords} 포함 여부",
    }

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "당신은 답변 품질을 평가하는 채점자입니다.\n"
     "아래 기대 답변(reference)과 모델 답변(prediction)을 비교하고,\n"
     "의미가 일치하면 1, 부분적으로만 일치하면 0.5, 무관하면 0을 점수로 매기세요.\n"
     "응답은 반드시 첫 줄에 0/0.5/1 중 하나의 숫자만, 둘째 줄부터 짧은 이유를 적으세요."),
    ("human",
     "질문: {question}\n\n"
     "기대 답변: {reference}\n\n"
     "모델 답변: {prediction}"),
])

judge_chain = JUDGE_PROMPT | llm | StrOutputParser()

def llm_judge(run, example):
    reply = judge_chain.invoke({
        "question": example.inputs["question"],
        "reference": example.outputs["answer"],
        "prediction": run.outputs["answer"],
    })

    first_line = reply.strip().splitlines()[0].strip()
    try:
        score = float(first_line)
    except ValueError:
        score = 0
    return {
        "key": "llm_judge_semantic_match",
        "score": score,
        "comment": reply,
    }

result = evaluate(
    target,
    data=DATASET_NAME,
    evaluators=[contains_expected_keyword, llm_judge],
    experiment_prefix="v1-baseline",
)

print(result)
"""

# 실행 부 app.py 완료하면 지워야 됨
# if __name__ == "__main__":
#     query = "카카오 주가"
#     store.build_news_vector_db(query)
#     print("RAG 파이프라인 시작")
#     print(rag.invoke("카카오 주가 전망 어때?"))
#     print("RAG 파이프라인 완료")
#
#     # 평가
#     from langsmith.evaluation import evaluate
#     from langsmith import Client
#
#     client = Client()
#     DATASET_NAME = "korean-news-eval"
#
#     EVAL_QUESTIONS = [
#     {"question": "카카오 주가 전망 어때?",
#      "answer": "주어진 자료만으로는 판단하기 어렵습니다. 추가 실적과 시장 상황을 함께 확인해야 합니다."},
#     {"question": "반도체 업황은 지금 어떤 상황이야?",
#      "answer": "뉴스에 언급된 요인은 확인할 수 있으나, 전반적 업황 판단에는 추가 자료가 필요합니다."},
#     {"question": "환율이 주식에 미치는 영향은?",
#      "answer": "환율 변동은 수출입 기업 실적에 영향을 줄 수 있으나, 개별 종목 판단은 별도 확인이 필요합니다."},
# ]
#     existing = list(client.list_datasets(dataset_name=DATASET_NAME))
#
#     inputs = [{"question": ex["question"]} for ex in EVAL_QUESTIONS]
#     outputs = [{"answer": ex["answer"]} for ex in EVAL_QUESTIONS]
#
#     if existing:
#         client.delete_dataset(dataset_id=existing[0].id)
#         print("기존 Dataset 삭제")
#     dataset = client.create_dataset(
#         dataset_name=DATASET_NAME,
#         description="답변 평가 용",
#     )
#     client.create_examples(dataset_id=dataset.id, inputs=inputs, outputs=outputs)
#     print(f"새 Dataset 생성: {dataset.id}, Example {len(EVAL_QUESTIONS)}건")
#
#     loaded = client.read_dataset(dataset_name=DATASET_NAME)
#
#     examples = list(client.list_examples(dataset_id=dataset.id))
#
#     for ex in examples[:3]:
#         print("Q:", ex.inputs["question"])
#         print("A:", ex.outputs["answer"] if ex.outputs else "(없음)")
#         print()
#
#
#     def target(inputs):
#         store.build_news_vector_db(inputs["question"])
#         return {"answer": rag.invoke(inputs["question"])}
#
#
#     def contains_expected_keyword(run, example):
#         pred = run.outputs.get("answer", "")
#         expected = example.outputs.get("answer", "")
#
#         keywords = [w for w in expected.split() if len(w) >= 2][:2]
#         hit = all(k in pred for k in keywords)
#
#         return {
#             "key": "contains_expected_keyword",
#             "score": 1 if hit else 0,
#             "comment": f"필수 키워드 {keywords} 포함 여부",
#         }
#
#
#     JUDGE_PROMPT = ChatPromptTemplate.from_messages([
#         ("system",
#          "당신은 답변 품질을 평가하는 채점자입니다.\n"
#          "아래 기대 답변(reference)과 모델 답변(prediction)을 비교하고,\n"
#          "의미가 일치하면 1, 부분적으로만 일치하면 0.5, 무관하면 0을 점수로 매기세요.\n"
#          "응답은 반드시 첫 줄에 0/0.5/1 중 하나의 숫자만, 둘째 줄부터 짧은 이유를 적으세요."),
#         ("human",
#          "질문: {question}\n\n"
#          "기대 답변: {reference}\n\n"
#          "모델 답변: {prediction}"),
#     ])
#
#     judge_llm = build_llm("gemini")
#     judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()
#
#
#     def llm_judge(run, example):
#         reply = judge_chain.invoke({
#             "question": example.inputs["question"],
#             "reference": example.outputs["answer"],
#             "prediction": run.outputs["answer"],
#         })
#
#         first_line = reply.strip().splitlines()[0].strip()
#         try:
#             score = float(first_line)
#         except ValueError:
#             score = 0
#         return {
#             "key": "llm_judge_semantic_match",
#             "score": score,
#             "comment": reply,
#         }
#
#
#     result = evaluate(
#         target,
#         data=DATASET_NAME,
#         evaluators=[contains_expected_keyword, llm_judge],
#         experiment_prefix="v1-baseline",
#     )
#
#     print(result)

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