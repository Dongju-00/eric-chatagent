import os
from pathlib import Path
import hashlib

from dotenv import load_dotenv
from model.embed.embedder import KoreanGPTEmbedder
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import trafilatura

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

embeddings = KoreanGPTEmbedder()

vectorstore = Chroma(,embeddings)

# RAG
print("RAG 파이프라인 시작")
retriever = vectorstore.as_retriever(search_kwargs["k":3])

prompt = ChatPromptTemplate.from_message([
    ("system",
     "다음 문서를 근거로 사용자 질문에 답하세요. "
     "근거가 부족하면 '주어진 자료에서는 확인할 수 없습니다.'라고 답하세요.\n\n"
     "{context}"),
    ("human", "{question}"),
])

def build_llm() :
    return ChatGoogleGenerativeAI(
        model = "models/gemini-2.5-flash",
        google_api_key = os.getenv("GOOGLE_API_KEY")
    )

llm = build_llm()

