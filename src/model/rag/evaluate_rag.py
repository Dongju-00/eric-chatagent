import csv
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from model.store.vector_store import ChromaNewsVectorStore
from model.rag.baseline import stock_news_chain, build_llm  # 실제 서빙에 쓰는 LLM 재사용
from model.graph.query_rewrite import rewrite_stock_query

load_dotenv()

MODEL_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = MODEL_DIR / "data"

DEFAULT_OUTPUT_PATH = DATA_DIR / "eval" / "eval_results.csv"

stock_news_llm = build_llm("stock_news")

TEST_QUESTIONS = [
    "삼성전자 주가 전망 알려줘",
    "SK하이닉스 주식 알려줘",
    "삼전 주가 전망",
    "네이버 관련 주식 뉴스 알려줘",
    "국내 주식 시장 분위기 알려줘",
]

# 검색 품질 평가 (Gemini)
def evaluate_retrieval_with_gemini(question, contexts, metadatas):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY 환경변수가 없습니다.")

    client = genai.Client(api_key=api_key)

    context_text = ""
    for i, (context, metadata) in enumerate(zip(contexts, metadatas), start=1):
        title = metadata.get("title", "")
        link = metadata.get("link", "")
        context_text += f"""[검색 결과 {i}]
제목: {title}
내용:
{context}
링크: {link}""".strip()
        context_text += "\n\n"

    prompt = f"""너는 RAG 검색 품질 평가자입니다.

아래 사용자 질문과 검색된 뉴스 chunk들을 보고 평가하세요.
답변 생성 품질이 아니라, 검색된 문서가 질문에 적절한지만 평가합니다.

평가 기준:
1. keyword_search: 질문 내용이 검색하는 내용과 잘 맞는가? 1~5점
2. retrieval_relevance: 검색 결과가 질문과 의미적으로 관련 있는가? 1~5점
3. keyword_match: 질문의 핵심 키워드가 검색 결과에 잘 반영되었는가? 1~5점
4. context_usefulness: 이 검색 결과를 바탕으로 답변을 만들 수 있는가? 1~5점
5. pass: 검색 결과가 RAG 답변 생성에 사용할 만하면 true, 아니면 false
6. reason: 평가 이유를 한국어로 짧게 설명

반드시 JSON 형식으로만 답변하세요. 마크다운 코드블록(```) 없이 순수 JSON만 출력하세요.

사용자 질문:
{question}

검색된 뉴스 chunk:
{context_text}"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    raw = response.text.strip()
    # 혹시 코드블록으로 감싸져 오면 제거
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [경고] Gemini 응답 JSON 파싱 실패: {raw[:200]}", flush=True)
        return {
            "keyword_search": None, "retrieval_relevance": None,
            "keyword_match": None, "context_usefulness": None,
            "pass": None, "reason": "JSON 파싱 실패",
        }



# 생성 품질 평가 (규칙 기반)
def compute_repetition_ratio(answer: str) -> float:

    tokens = answer.split()
    if len(tokens) < 5:
        return 1.0
    return len(set(tokens)) / len(tokens)

# 길이 n 이상의 동일 어절 시퀀스가 몇 번 반복되는지 카운트
def compute_ngram_repetition(answer: str, n: int = 4) -> int:

    tokens = answer.split()
    if len(tokens) < n * 2:
        return 0
    seen = {}
    repeats = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] == 2:  # 두 번째 등장부터 반복으로 카운트
            repeats += 1
    return repeats

# 컨텍스트에 실제로 존재하는 비율. 환각의 근사 지표.
def compute_groundedness(answer: str, contexts: list[str]) -> float:

    ctx_text = " ".join(contexts)
    candidates = re.findall(r"[가-힣A-Za-z0-9]{2,}", answer)
    if not candidates:
        return 1.0
    grounded = sum(1 for w in candidates if w in ctx_text)
    return grounded / len(candidates)

# 실제 서빙 LLM으로 답변을 생성하고 규칙 기반 지표를 계산.
def evaluate_generation(question: str, contexts: list[str]) -> dict:

    if not contexts:
        return {
            "answer": "",
            "repetition_ratio": None,
            "ngram_repeats": None,
            "groundedness": None,
            "generation_pass": None,
        }

    context_text = "\n\n".join(contexts)
    filled = f"참고 뉴스:\n{context_text}\n\n질문: {question}\n답변: "
    # answer = stock_news_llm.invoke(filled)
    answer = stock_news_chain.invoke({"context_text": context_text, "question": question})

    repetition_ratio = compute_repetition_ratio(answer)
    ngram_repeats = compute_ngram_repetition(answer)
    groundedness = compute_groundedness(answer, contexts)

    # 통과 기준: 어휘 다양성 0.4 이상, 4-gram 반복 0회, 근거성 0.3 이상, 형식 문제 없음
    generation_pass = (
        repetition_ratio >= 0.4
        and ngram_repeats <= 1
        and groundedness >= 0.2
    )

    return {
        "answer": answer,
        "repetition_ratio": round(repetition_ratio, 3),
        "ngram_repeats": ngram_repeats,
        "groundedness": round(groundedness, 3),
        "generation_pass": generation_pass,
    }

# 전체 평가 실행 + CSV 저장
def evaluate_questions(questions=None, output_path=DEFAULT_OUTPUT_PATH):
    questions = questions or TEST_QUESTIONS
    store = ChromaNewsVectorStore()
    rows = []

    for question in questions:
        print("=" * 60)
        print("질문:", question)
        rw = rewrite_stock_query(question)

        question_id = store.build_news_vector_db(query=rw["rewritten_query"], display=10, sort=rw["sort"], company=rw["company"], aliases=rw["company_aliases"])
        print("저장된 질문 ID:", question_id)

        result = store.search_similar_news(question, question_id=question_id, top_k=3)

        contexts = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        print("검색된 chunk:")
        for metadata, distance in zip(metadatas, distances):
            print("-", metadata.get("title", ""), "/ distance:", distance)

        # 검색 품질
        retrieval_eval = evaluate_retrieval_with_gemini(question, contexts, metadatas)
        print("검색 평가:", retrieval_eval)

        # 생성 품질
        gen_eval = evaluate_generation(question, contexts)
        print("생성 평가:", {k: v for k, v in gen_eval.items() if k != "answer"})
        print("답변:", gen_eval["answer"][:150])

        rows.append({
            "question": question,
            "question_id": question_id,
            "num_contexts": len(contexts),
            "avg_distance": round(sum(distances) / len(distances), 4) if distances else None,
            # 검색 품질
            "retrieval_keyword_search": retrieval_eval.get("keyword_search"),
            "retrieval_relevance": retrieval_eval.get("retrieval_relevance"),
            "retrieval_keyword_match": retrieval_eval.get("keyword_match"),
            "retrieval_context_usefulness": retrieval_eval.get("context_usefulness"),
            "retrieval_pass": retrieval_eval.get("pass"),
            "retrieval_reason": retrieval_eval.get("reason"),
            # 생성 품질
            "answer": gen_eval["answer"],
            "repetition_ratio": gen_eval["repetition_ratio"],
            "ngram_repeats": gen_eval["ngram_repeats"],
            "groundedness": gen_eval["groundedness"],
            "generation_pass": gen_eval["generation_pass"],
        })

    _write_csv(rows, output_path)
    _print_summary(rows)
    return rows


def _write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n결과 저장: {output_path}")


def _print_summary(rows):
    n = len(rows)
    retrieval_pass = sum(1 for r in rows if r["retrieval_pass"] is True)
    gen_pass = sum(1 for r in rows if r["generation_pass"] is True)
    avg_repetition = sum(r["repetition_ratio"] or 0 for r in rows) / n
    avg_groundedness = sum(r["groundedness"] or 0 for r in rows) / n
    total_ngram_repeats = sum(r["ngram_repeats"] or 0 for r in rows)

    print("\n" + "=" * 60)
    print("요약")
    print(f"  검색 통과율     : {retrieval_pass}/{n}")
    print(f"  생성 통과율     : {gen_pass}/{n}")
    print(f"  평균 어휘 다양성 : {avg_repetition:.3f} (1.0에 가까울수록 반복 없음)")
    print(f"  평균 근거성     : {avg_groundedness:.3f}")
    print(f"  4-gram 반복 총합 : {total_ngram_repeats}")


if __name__ == "__main__":
    evaluate_questions()