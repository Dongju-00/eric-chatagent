"""OpenRouter 클라이언트 — fallback 자유 답변용.

배포된 vLLM 라우터(Modal)가 503을 반환하면 라우터 분류가 실패해 모든
질문이 fallback으로 간다. 자유 답변까지 같은 서버에 묶여 있으면 고정
문구만 나오므로, 자유 답변을 외부 LLM으로 옮겨 라우터 장애와 분리한다.

라우터 분류는 vLLM에 남긴다. 여기서 다루지 않는다.

새 의존성을 추가하지 않는다. 이미 선언된 ``langchain-openai`` 의
``ChatOpenAI`` 를 OpenRouter 엔드포인트로 돌려 쓴다.
"""

import os

from langchain_openai import ChatOpenAI

__all__ = [
    "FALLBACK_MESSAGE",
    "FALLBACK_SYSTEM_PROMPT",
    "OPENROUTER_BASE_URL",
    "REASONING_OFF",
    "build_openrouter_llm",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

API_KEY_ENV = "ERIC_CHATAGENT_OPENROUTER_KEY"
MODEL_ENV = "ERIC_CHATAGENT_OPENROUTER_MODEL"

# fallback 은 주식과 무관한 질문에 3문장으로 답하는 자리라 큰 모델이 필요없다.
# OpenRouter 무료 티어의 채팅 모델은 전부 추론 모델이므로 추론을 끈다
# (아래 REASONING_OFF 참고).
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"

# 추론을 끄지 않으면 토큰 예산이 생각 단계에서 전부 소모돼 content 가 빈다.
# 실측(nemotron-3.5-lightning:free, 같은 프롬프트):
#   추론 비활성  0.80s  출력 24토큰(추론 0)   -> 정상 답변
#   effort=low   3.80s  출력 400토큰(추론 400) -> content 빈 문자열
#   예산 1500   48.66s  출력 1193토큰(추론 1143) -> 답변은 오지만 날씨를 지어냈다
REASONING_OFF = {"reasoning": {"enabled": False}}

# OpenRouter 가 없거나 호출이 실패했을 때의 최후 응답. 기존 문구를 유지한다.
FALLBACK_MESSAGE = "주식 관련 질문에 답변드릴 수 있습니다. 종목명과 함께 질문해 주세요."

# 주식과 무관한 질문에 쓰는 시스템 프롬프트. 기존 vLLM 경로의 문구를 그대로 쓴다.
FALLBACK_SYSTEM_PROMPT = (
    "당신은 한국 주식 정보를 안내하는 챗봇입니다. "
    "사용자가 주식과 무관한 질문을 했습니다. "
    "3문장 이내로 간단히 답하세요."
)


def build_openrouter_llm() -> ChatOpenAI | None:
    """OpenRouter 클라이언트를 만든다. 키가 없으면 ``None``.

    키가 없는 환경에서도 서버가 떠야 하므로 예외를 던지지 않는다.
    ``fallback_node`` 가 ``None`` 을 받으면 고정 문구로 답한다.
    """
    api_key = (os.getenv(API_KEY_ENV) or "").strip()
    if not api_key:
        print(f"[OpenRouter] {API_KEY_ENV} 가 없어 fallback 자유 답변을 끈다", flush=True)
        return None

    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        model=os.getenv(MODEL_ENV, DEFAULT_MODEL),
        max_tokens=200,
        temperature=0.7,
        timeout=30,
        extra_body=REASONING_OFF,
    )
