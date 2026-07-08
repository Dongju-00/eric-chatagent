import os
import csv
from dotenv import load_dotenv
from google import genai

from model.store.vector_store import ChromaNewsVectorStore
from pathlib import Path

load_dotenv()

MODEL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MODEL_DIR / "data"

DEFAULT_INPUT_PATH = DATA_DIR / "processed" / "eval_questions.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "processed" / "rag_eval_results.csv"

def evaluate_retrieval_with_gemini(question, contexts, metadatas):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY 환경변수가 없습니다.")

    client = genai.Client(api_key=api_key)

    context_text = ""

    for i, (context, metadata) in enumerate(zip(contexts, metadatas), start=1):
        title = metadata.get("title", "")
        link = metadata.get("link", "")

        context_text += f"""
[검색 결과 {i}]
제목: {title}
내용:
{context}
링크: {link}
""".strip()
        context_text += "\n\n"

    prompt = f"""
너는 RAG 검색 품질 평가자입니다.

아래 사용자 질문과 검색된 뉴스 chunk들을 보고 평가하세요.
답변 생성 품질이 아니라, 검색된 문서가 질문에 적절한지만 평가합니다.

평가 기준:
1. retrieval_relevance: 검색 결과가 질문과 의미적으로 관련 있는가? 1~5점
2. keyword_match: 질문의 핵심 키워드가 검색 결과에 잘 반영되었는가? 1~5점
3. context_usefulness: 이 검색 결과를 바탕으로 답변을 만들 수 있는가? 1~5점
4. pass: 검색 결과가 RAG 답변 생성에 사용할 만하면 true, 아니면 false
5. reason: 평가 이유를 한국어로 짧게 설명

반드시 JSON 형식으로만 답변하세요.

사용자 질문:
{question}

검색된 뉴스 chunk:
{context_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


def evaluate_questions():
    store = ChromaNewsVectorStore()

    questions = [
        "삼성전자 주가 전망 알려줘",
        "네이버 관련 주식 뉴스 알려줘",
        "국내 주식 시장 분위기 알려줘",
    ]

    for question in questions:
        print("=" * 60)
        print("질문:", question)

        # 평가용으로 해당 질문에 대한 뉴스도 새로 색인
        question_id = store.build_news_vector_db(question, display=10)
        print("저장된 질문 ID:", question_id)

        result = store.search_similar_news(question, top_k=3)

        contexts = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        print("검색된 chunk:")
        for metadata, distance in zip(metadatas, distances):
            print("-", metadata.get("title", ""), "/ distance:", distance)

        evaluation = evaluate_retrieval_with_gemini(
            question=question,
            contexts=contexts,
            metadatas=metadatas,
        )

        print("Gemini 평가:")
        print(evaluation)


if __name__ == "__main__":
    evaluate_questions()