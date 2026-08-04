import operator
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, Annotated
from typing import Literal, Any

from model.store.vector_store import ChromaNewsVectorStore
from model.rag.baseline import build_llm, prompt, KoreanGPTLLM
from model.graph.query_rewrite import rewrite_stock_query

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

store = ChromaNewsVectorStore()

PROJECT_DIR = Path(__file__).resolve().parents[1]

# graph에 적용할 llm들 빌드해놓기
small_talk_llm = build_llm("small_talk")
stock_news_llm = build_llm("stock_news")

# 공개 가중치 모델 로드
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
#
# _tok = None
# _model = None
#
# def load_model():
#     global _tok, _model
#     if _model is None:
#         _tok = AutoTokenizer.from_pretrained(MODEL_ID)
#         _model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,)
#         _model.eval()
#     return _tok, _model


route_llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

ROUTER_PROMPT = """사용자의 질문을 아래 세 가지 중 하나로 분류하세요.

smalltalk:
- 인사, 감사, 잡담, 감정, 일상 대화
- 예: 안녕, 반가워, 고마워, 오늘 기분 어때?, 뭐 하고 있어?

stock_rag:
- 주식, 기업, 종목, 주가, 실적, 배당, 투자 관련 질문
- 예: 삼성전자 주가 알려줘, SK하이닉스 실적은 어때?

fallback:
- smalltalk과 stock_rag에 해당하지 않는 질문
- 예: 파이썬 리스트 사용법 알려줘, 서울 날씨 알려줘

반드시 다음 중 하나만 출력하세요.
smalltalk
stock_rag
fallback

질문: {question}
분류:"""

FALLBACK_PROMPT = PromptTemplate.from_template(
    """당신은 한국 주식 정보를 안내하는 챗봇입니다.
사용자가 주식과 무관한 질문을 했습니다. 간단히 답하되, 3문장 이내로 짧게 작성하세요.
답변 마지막에 주식 관련 질문을 안내해 주세요.

질문: {question}
답변:"""
)

# 공개 가중치 모델로 라우팅
# VALID = {"smalltalk", "stock_rag", "fallback"}
#
# def classify(question: str) -> str:
#     tok, model = load_model()
#     messages = [{"role": "user", "content": ROUTER_PROMPT.format(question=question)}]
#     text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tok(text, return_tensors="pt")
#
#     with torch.no_grad():
#         out = model.generate(**inputs, max_new_tokens=8, do_sample=False,
#                              pad_token_id=tok.eos_token_id)
#
#     raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#     print(f"[라우터 원본 출력] {raw!r}", flush=True)      # ← 디버깅용
#
#     cleaned = raw.strip().lower().replace("`", "").replace("_", "_")
#
#     # 부분 매칭 (긴 라벨부터 검사)
#     for label in ["stock_rag", "smalltalk", "fallback"]:
#         if label in cleaned:
#             return label
#     # 느슨한 매칭
#     if "stock" in cleaned:
#         return "stock_rag"
#     if "small" in cleaned or "talk" in cleaned:
#         return "smalltalk"
#     return "fallback"

router_prompt = PromptTemplate.from_template(ROUTER_PROMPT)
router_chain = ROUTER_PROMPT | route_llm | StrOutputParser()

fallback_chain = FALLBACK_PROMPT | route_llm | StrOutputParser()

class AgentState(TypedDict):
    question : str
    answer : str

    question_id : str
    retrieved_docs: list[dict[str, Any]]
    contexts : list[str]

    rewritten_query : str
    ticker : str
    search_sort : str
    company : str
    company_aliases : list[str]

    fallback : str
    trace: Annotated[list[str], operator.add]

def build_graph():
    # 노드 :  route_node, small_talk_node, search_news_node, retrieval_news_node, generate_answer_node
    def router_node(state: AgentState) -> dict:

        route = router_chain.invoke({"question": state["question"]}).strip().lower()

        if "stock" in route:
            route = "stock_rag"
        elif "small" in route or "talk" in route:
            route = "smalltalk"
        else :
            route = "fallback"

        return {"route" : route, "trace" : ["router_node"]}


    def small_talk_node(state: AgentState) -> dict:
        question= f"질문: {state["question"]}\n 답변: "
        answer = small_talk_llm.invoke(question)
        return {"answer" : answer, "trace" : ["small_talk_node"]}

    def rewrite_query_node(state: AgentState) -> dict:
        result = rewrite_stock_query(state["question"])

        print("원래 질문:", state["question"])
        print("재작성 검색어:", result["rewritten_query"])
        print("기업:", result["company"])
        print("기업 별칭:", result["company_aliases"])
        print("질문 의도:", result["matched_intents"])

        return {
            "company": result["company"],
            "ticker": result["ticker"],
            "company_aliases": result["company_aliases"],
            "rewritten_query": result["rewritten_query"],
            "search_sort": result["sort"],
            "trace": ["rewrite_query_node"],
        }

    def search_news_node(state: AgentState) -> dict:
        question_id = store.build_news_vector_db(
            query=state.get("rewritten_query", state["question"]),
            display=10,
            sort=state.get("search_sort", "sim"),
            company=state.get("company"),
            aliases=state.get("company_aliases", []),
        )
        return {"question_id": question_id, "trace": ["search_news_node"]}
    # def retrieve_news_node(state: AgentState) -> dict:
    #     result = store.search_similar_news(state["question"], question_id=state["question_id"], top_k=3)
    #     return {"contexts" : result["documents"][0], "retrieved_docs" : result["metadatas"][0], "trace" : ["retrieve_news_node"]}

    def retrieve_news_node(state: AgentState) -> dict:
        result = store.search_similar_news(question=state["question"], question_id=state["question_id"], top_k=2, max_distance=1.25,)

        contexts = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        print("검색 결과 거리 확인")

        for metadata, distance in zip(metadatas, distances,):
            print(metadata.get("title", ""), "/ distance:", distance)

        return {"contexts": contexts, "retrieved_docs": metadatas, "trace": ["retrieve_news_node"]}

    def generate_node(state: AgentState) -> dict:
        contexts = state["contexts"]
        if not contexts:
            return {"answer": ("질문과 관련성이 충분한 뉴스를 \n찾지 못했습니다."), "trace": ["generate_node"]}
        context_text = "\n\n".join(contexts)
        filled = f"참고 뉴스:\n{context_text}\n\n질문: {state['question']}\n답변: "
        answer = stock_news_llm.invoke(filled)
        return {"answer": answer, "trace": ["generate_node"]}

    def fallback_node(state: AgentState) -> dict:
        answer = route_llm.invoke(state["question"])
        return {"answer" : "주어진 자료로는 판단할 수 없습니다", "trace" : ["fallback_node"]}

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("router",router_node)
    graph_builder.add_node("small_talk",small_talk_node)
    graph_builder.add_node("rewrite_query",rewrite_query_node)
    graph_builder.add_node("search_news",search_news_node)
    graph_builder.add_node("retrieve_news",retrieve_news_node)
    graph_builder.add_node("generate",generate_node)
    graph_builder.add_node("fallback",fallback_node)

    graph_builder.add_edge(START,"router")

    graph_builder.add_conditional_edges(
        "router",
        lambda state:state["route"],
        {
            "smalltalk" : "small_talk",
            "stock_rag" : "rewrite_query",
            "fallback" : "fallback",
        }
    )

    # RAG 파이프라인
    graph_builder.add_edge("rewrite_query","search_news")
    graph_builder.add_edge("search_news","retrieve_news")
    graph_builder.add_edge("retrieve_news","generate")

    graph_builder.add_edge("small_talk", END)
    graph_builder.add_edge("generate",END)
    graph_builder.add_edge("fallback",END)

    return graph_builder.compile(checkpointer=MemorySaver())
