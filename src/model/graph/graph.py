import operator
import os
import sys

from langchain_google_genai import GoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from typing import Literal, Any

from model.train.model import KoreanGPT
from model.store.vector_store import ChromaNewsVectorStore

# 노드 : search_news_node, retrieval_news_node, generate_answer_node

RouteTpye = Literal[
    "smalltalk",
    "stock_rag",
    "search",
    "fallback",
]

class AgentState(TypedDict):
    question : str
    answer : str

    question_id : str
    answer_id : int
    retrieved_docs: list[dict[str, Any]]
    contexts : list[str]

    route : RouteTpye
    route_reason : str
    rewritten_query : str

    error : str
    trace: Annotated[list[str], operator.add]

def search_news_node(state : AgentState) -> AgentState:
    return {
        "answer" : "검색 기반 답변",
        "trace" : ["search_news_node"],
    }

def retrieve_news_node(state : AgentState) -> AgentState:
    return {
        "question" : "검색 결과 기반 답변",
        "trace" : ["retrieve_news_node"],
    }



