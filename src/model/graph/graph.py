import functools
import os
import httpx

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from typing import TYPE_CHECKING, Literal, Any

from model.graph.query_rewrite import rewrite_stock_query
from model.news.collect import collect_and_index
from model.llm.openrouter import FALLBACK_MESSAGE, FALLBACK_SYSTEM_PROMPT
from model.paths import SP_PREFIX
from model.train.sp_tokenizer import load_sp
from model.rag.prompts import build_stock_news_chain

# 타입 힌트 전용 import다. 런타임에 `graph -> resources` 의존을 만들지 않는다.
# `resources.py` 가 나중에 이 모듈의 VLLM_BASE_URL / MODEL_ID / ROUTER_PROMPT 중
# 하나라도 필요해지면(Task 5의 fallback 교체) 모듈 레벨 import는 순환이 된다.
if TYPE_CHECKING:
    from model.resources import AppResources


load_dotenv()

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")

# vLLM 서버에서 사용하는 모델명
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

ROUTER_PROMPT = """사용자의 질문을 아래 세 가지 중 하나로 분류하세요.

smalltalk:
- 인사, 감사, 잡담, 감정, 일상 대화
- 예: 안녕, 반가워, 고마워, 오늘 기분 어때?, 뭐 하고 있어?

stock_rag:
- 주식, 기업, 종목, 주가, 실적, 배당, 투자 관련 질문
- 예: 삼성전자 주가 알려줘, SK하이닉스 실적은 어때?

fallback:
- smalltalk과 stock_rag에 해당하지 않는 질문
- 예: 파이썬 리스트 사용법 알려줘, 서울 날씨 알려줘, 선형대수 설명해줘

반드시 다음 중 하나만 출력하세요.
smalltalk
stock_rag
fallback

질문: {question}
분류:"""

def _parse_label(raw: str) -> str:
    cleaned = raw.strip().lower().replace("`", "")
    for label in ["stock_rag", "smalltalk", "fallback"]:
        if label in cleaned:
            return label
    if "stock" in cleaned:
        return "stock_rag"
    if "small" in cleaned or "talk" in cleaned:
        return "smalltalk"
    return "fallback"


def classify(question: str) -> str:
    try:
        resp = httpx.post(
            f"{VLLM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        print(f"[vLLM 라우터 출력] {raw!r}", flush=True)
        return _parse_label(raw)
    except Exception as e:
        print(f"[vLLM 호출 실패] {e}", flush=True)
        # 분류 실패 시 fallback 경로로 보낸다. 자유 답변은 그 경로에서
        # OpenRouter 가 맡는다(fallback_node). "로컬 Transformers 폴백"은
        # 구현된 적이 없다 — README 에도 그렇게 적혀 있었으나 사실과 달랐다.
        return "fallback"

class AgentState(TypedDict):
    question : str
    route: Literal["smalltalk", "stock_rag", "fallback"]
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

    # 리듀서를 쓰지 않는다. operator.add 를 붙이면 LangGraph 가 입력값에도
    # 리듀서를 적용해 턴 시작 초기화가 불가능하고, 같은 thread_id 로
    # 호출할수록 trace 가 무한히 늘어난다(실측 2 -> 4 -> 6).
    # 대신 각 노드가 state 의 기존 리스트에 명시적으로 덧붙인다.
    trace: list[str]

    # 멀티턴 대화 이력. trace 와 목적이 다르므로 별개 필드로 둔다.
    # trace 는 디버깅용 경로 기록이고 이력은 모델에 넘기는 문맥이다.
    history: list[dict[str, str]]

# 이력을 프롬프트에 넣을 때의 토큰 상한. block_size 가 384 이고 뉴스
# 컨텍스트(median 61 토큰)와 질문·답변 자리를 남겨야 하므로 넉넉히 두지 않는다.
HISTORY_MAX_TOKENS = int(os.getenv("ERIC_HISTORY_MAX_TOKENS", "120"))


def _advance(state: "AgentState", node: str) -> list[str]:
    """이번 턴의 trace 에 노드를 덧붙인다.

    리듀서가 없으므로 노드 반환값이 덮어쓰기가 된다. 들어온 리스트에
    직접 덧붙여야 턴 안의 경로가 남는다.
    """
    return [*state.get("trace", []), node]


def render_history(history: list[dict[str, str]]) -> str:
    """이력을 프롬프트에 끼울 문자열로 만든다.

    `{history}` 슬롯은 **호출자가 머리말과 끝 빈 줄까지 넘기는** 계약이다.
    끝 빈 줄이 없으면 바로 뒤의 `참고 뉴스:` 가 이력 마지막 줄에 붙는다.
    이력이 없으면 빈 문자열이어야 하고, 그때 프롬프트는 변경 전과
    글자 단위로 같다.
    """
    if not history:
        return ""

    lines = ["이전 대화:"]
    for turn in history:
        lines.append(f"질문: {turn.get('question', '')}")
        lines.append(f"답변: {turn.get('answer', '')}")
    return "\n".join(lines) + "\n\n"


def trim_history(history: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    """토큰 상한에 맞게 오래된 턴부터 버린다.

    최신 턴은 그것만으로 상한을 넘더라도 남긴다. 직전 질문을 버리면
    지시대명사를 해소할 근거가 사라져 멀티턴의 의미가 없어진다.
    """
    if not history:
        return []

    sp = load_sp(SP_PREFIX)

    def cost(turns) -> int:
        text = "".join(
            f"질문: {t.get('question', '')}\n답변: {t.get('answer', '')}\n"
            for t in turns
        )
        return len(sp.encode(text, out_type=int))

    kept = list(history)
    while len(kept) > 1 and cost(kept) > max_tokens:
        kept.pop(0)
    return kept


def fallback_node(state: "AgentState", llm=None) -> dict:
    """주식과 무관한 질문에 자유 답변한다. OpenRouter 를 쓴다.

    `llm` 을 인자로 받는 이유는 테스트가 바꿔 끼울 수 있어야 하기 때문이다.
    그래프 등록 시 `functools.partial` 로 주입된 클라이언트를 묶는다.

    어떤 경우에도 예외를 밖으로 내지 않는다. 지금 vLLM 라우터가 503이라
    이 경로가 상시 경로이고, 여기서 터지면 모든 질문이 500이 된다.
    키나 오류 본문을 응답에 섞지 않는다.
    """
    if llm is None:
        return {"answer": FALLBACK_MESSAGE, "trace": _advance(state, "fallback_node")}

    try:
        # OpenRouter 모델은 멀티턴이 기본이므로 이력을 그대로 넘긴다.
        # KoreanGPT 경로와 달리 재학습이 필요하지 않다.
        messages = [SystemMessage(content=FALLBACK_SYSTEM_PROMPT)]
        for turn in trim_history(state.get("history", []), HISTORY_MAX_TOKENS):
            messages.append(HumanMessage(content=turn.get("question", "")))
            messages.append(AIMessage(content=turn.get("answer", "")))
        messages.append(HumanMessage(content=state["question"]))

        response = llm.invoke(messages)
        answer = (getattr(response, "content", "") or "").strip()
        if not answer:
            print("[fallback] OpenRouter 응답이 비어 고정 문구로 내려간다", flush=True)
            return {"answer": FALLBACK_MESSAGE, "trace": _advance(state, "fallback_node")}

        print(f"[fallback OpenRouter 응답] {answer[:50]!r}", flush=True)
        return {"answer": answer, "trace": _advance(state, "fallback_node")}

    except Exception as e:
        # 예외 메시지에 키가 섞여 들어올 수 있어 종류만 남긴다
        print(f"[fallback OpenRouter 실패] {type(e).__name__}", flush=True)
        return {"answer": FALLBACK_MESSAGE, "trace": _advance(state, "fallback_node")}


def build_graph(resources: "AppResources"):
    """노드 클로저에 리소스를 주입해 그래프를 조립한다.

    ``resources``는 필수 인자다. 모듈 레벨 전역을 읽지 않으므로 import만으로
    가중치가 적재되지 않는다.
    """
    store = resources.store
    small_talk_llm = resources.small_talk_llm
    stock_news_chain = build_stock_news_chain(resources.news_llm)

    # 노드 :  route_node, small_talk_node, search_news_node, retrieval_news_node, generate_answer_node
    def router_node(state: AgentState) -> dict:

        # 공개 가중치 모델 Qwen2.5
        route = classify(state["question"])

        if "stock" in route:
            route = "stock_rag"
        elif "small" in route or "talk" in route:
            route = "smalltalk"
        else :
            route = "fallback"

        # 턴 시작 초기화. 리듀서가 없으므로 trace 는 여기서 새로 시작하고,
        # 지난 턴의 근거(question_id/contexts/retrieved_docs)를 비운다.
        # 비우지 않으면 스몰톡 답변에 직전 주식 턴의 뉴스가 근거처럼 따라붙는다.
        return {
            "route": route,
            "trace": ["router_node"],
            "question_id": "",
            "contexts": [],
            "retrieved_docs": [],
        }


    def small_talk_node(state: AgentState) -> dict:
        question= f"질문: {state["question"]}\n 답변: "
        answer = small_talk_llm.invoke(question)
        return {"answer" : answer, "trace": _advance(state, "small_talk_node")}

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
            "trace": _advance(state, "rewrite_query_node"),
        }

    def search_news_node(state: AgentState) -> dict:
        question_id = collect_and_index(
            store,
            query=state.get("rewritten_query", state["question"]),
            display=3,
            sort=state.get("search_sort", "sim"),
            company=state.get("company"),
            aliases=state.get("company_aliases", []),
        )
        return {"question_id": question_id, "trace": _advance(state, "search_news_node")}

    def retrieve_news_node(state: AgentState) -> dict:
        result = store.search_similar_news(
            question=state["question"],
            question_id=state["question_id"],
            top_k=2,
            max_distance=1.25,
            # 질문분이 부족하면 상시 RSS 코퍼스에서 보충한다
            company=state.get("company"),
        )

        contexts = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        print("검색 결과 거리 확인")

        for metadata, distance in zip(metadatas, distances,):
            print(metadata.get("title", ""), "/ distance:", distance)

        return {"contexts": contexts, "retrieved_docs": metadatas, "trace": _advance(state, "retrieve_news_node")}

    def generate_node(state: AgentState) -> dict:
        contexts = state["contexts"]
        if not contexts:
            return {"answer": ("질문과 관련성이 충분한 뉴스를 \n찾지 못했습니다."), "trace": _advance(state, "generate_node")}
        context_text = "\n\n".join(contexts)
        # history 는 프롬프트의 partial_variables 기본값("")을 그대로 쓴다.
        # KoreanGPT 뉴스 모델은 아직 멀티턴 학습 전이라 이력을 넣으면 무시한다
        # (실측: "그 회사 어때?" -> "응, 잘 지냈어"). 별도 계획의 재학습이 끝나면
        # 여기에 "history": render_history(state.get("history", [])) 를 넘긴다.
        answer = stock_news_chain.invoke({
            "context_text": context_text,
            "question": state["question"],
        })
        return {"answer": answer, "trace": _advance(state, "generate_node")}

    # 공개 가중치 모델 Qwen을 라우터로 사용할 때
    def record_history_node(state: AgentState) -> dict:
        """이번 턴의 질문·답변을 이력에 더하고 토큰 상한을 적용한다.

        모든 경로의 끝에 둔다. trace 와 달리 이력은 세션에 누적되는 것이
        의도이고, 상한은 `trim_history` 가 맡는다.
        """
        turn = {"question": state["question"], "answer": state.get("answer", "")}
        history = [*state.get("history", []), turn]
        return {
            "history": trim_history(history, HISTORY_MAX_TOKENS),
            "trace": _advance(state, "record_history_node"),
        }

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("router",router_node)
    graph_builder.add_node("small_talk",small_talk_node)
    graph_builder.add_node("rewrite_query",rewrite_query_node)
    graph_builder.add_node("search_news",search_news_node)
    graph_builder.add_node("retrieve_news",retrieve_news_node)
    graph_builder.add_node("generate",generate_node)
    graph_builder.add_node(
        "fallback",
        functools.partial(fallback_node, llm=resources.fallback_llm),
    )
    graph_builder.add_node("record_history", record_history_node)

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

    # 모든 경로가 이력 기록을 거쳐 끝난다
    graph_builder.add_edge("small_talk", "record_history")
    graph_builder.add_edge("generate", "record_history")
    graph_builder.add_edge("fallback", "record_history")
    graph_builder.add_edge("record_history", END)

    return graph_builder.compile(checkpointer=MemorySaver())
