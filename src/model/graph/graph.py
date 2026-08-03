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

load_dotenv()

store = ChromaNewsVectorStore()

PROJECT_DIR = Path(__file__).resolve().parents[1]

# graph에 적용할 llm들 빌드해놓기
small_talk_llm = build_llm("small_talk")
stock_news_llm = build_llm("stock_news")
route_llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "질문을 아래 네 템플릿 중 하나로 분류하세요.\n"
     "- smalltalk: 인사, 잡담, 감정표현\n"
     "- stock_rag: 주식, 뉴스, 경제 관련 질문\n"
     "- fallback: 그 외 질문들"),
    ("human", "{question}")
])

router_chain = ROUTER_PROMPT | route_llm | StrOutputParser()

RouteType = Literal["smalltalk", "stock_rag", "fallback"]

class AgentState(TypedDict):
    question : str
    answer : str

    question_id : str
    retrieved_docs: list[dict[str, Any]]
    contexts : list[str]

    route : RouteType
    rewritten_query : str
    ticker : str
    search_sort : str


    fallback : str
    trace: Annotated[list[str], operator.add]

def build_graph():
    # 노드 :  route_node, small_talk_node, search_news_node, retrieval_news_node, generate_answer_node
    def router_node(state: AgentState) -> dict:

        route = router_chain.invoke({"question" : state["question"]}).strip()

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
        print("질문 의도:", result["matched_intents"])

        return {
            "company": result["company"],
            "ticker": result["ticker"],
            "rewritten_query": result["rewritten_query"],
            "search_sort": result["sort"],
            "trace": ["rewrite_query_node"],
        }

    def search_news_node(state: AgentState) -> dict:
        query = state.get("rewritten_query", state["question"])
        sort = state.get("search_sort", "sim")
        question_id = store.build_news_vector_db(query=query, display=10, sort=sort)
        return {"question_id" : question_id ,"trace" : ["search_news_node"]}

    # def retrieve_news_node(state: AgentState) -> dict:
    #     result = store.search_similar_news(state["question"], question_id=state["question_id"], top_k=3)
    #     return {"contexts" : result["documents"][0], "retrieved_docs" : result["metadatas"][0], "trace" : ["retrieve_news_node"]}

    def retrieve_news_node(state: AgentState) -> dict:
        result = store.search_similar_news(question=state["question"], question_id=state["question_id"], top_k=3, max_distance=None,)

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
        filled = f"참고 뉴스: {context_text}\n질문: {state['question']}\n답변: "
        answer = stock_news_llm.invoke(filled)
        return {"answer": answer, "trace": ["generate_node"]}

    def fallback_node(state: AgentState) -> dict:
        return {"answer" : "주어진 자료로는 판단할 수 없습니다", "trace" : ["fallback_node"]}

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("router",router_node)
    graph_builder.add_node("small_talk",small_talk_node)
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
            "stock_rag" : "search_news",
            "fallback" : "fallback",
        }
    )

    # RAG 파이프라인
    graph_builder.add_edge("search_news","retrieve_news")
    graph_builder.add_edge("retrieve_news","generate")

    graph_builder.add_edge("small_talk", END)
    graph_builder.add_edge("generate",END)
    graph_builder.add_edge("fallback",END)

    return graph_builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    graph = build_graph()
    for q in ["안녕 반가워", "삼성전자 주가 어때?", "리만 적분이 뭐야?"]:
        result = graph.invoke(
            {"question": q},
            config={"configurable": {"thread_id": "t1"}},  # MemorySaver 쓸 때만
        )
        print(f"\n[{q}]")
        print("route:", result["route"])
        print("trace:", result["trace"])
        print("answer:", result["answer"])