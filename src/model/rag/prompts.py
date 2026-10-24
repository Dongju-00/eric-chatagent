"""주식 뉴스 답변 프롬프트와 체인 조립.

프롬프트와 체인은 LLM 인스턴스를 인자로 받아 **호출 시점에** 조립한다.
모듈 레벨에서 체인을 만들면 import만으로 가중치가 적재되므로 그렇게 하지
않는다.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

# ``{history}``는 이전 대화 이력 슬롯이다. KoreanGPT는 아직 멀티턴 학습 전이므로
# 기본값은 빈 문자열이고, 그때 프롬프트는 이력 슬롯이 없던 기존 형태와 글자
# 단위로 같다 (tests/test_resources.py 의 StockNewsPromptTests 가 단정한다).
#
# 슬롯 위치가 중요하다. 이력은 **증거 블록 바깥**, `참고 뉴스:` 머리말보다 앞에
# 둔다. 증거 블록 안에 넣으면 `아래 참고 뉴스만 근거로 답변하세요` 지시 아래로
# 이력이 들어가 모델이 이력을 검색된 뉴스로 읽는다. 근거 없는 생성이 유입되는
# 경로이므로 `참고 뉴스:`와 `{context_text}` 사이에는 아무것도 끼우지 않는다.
#
# 이력을 넣는 호출자가 머리말과 끝 줄바꿈까지 포함해 넘긴다
# (예: ``"이전 대화:\n질문: ...\n답변: ...\n\n"``).
STOCK_NEWS_PROMPT = PromptTemplate.from_template(
    """당신은 한국 주식 뉴스를 요약하는 챗봇입니다.
아래 참고 뉴스만 근거로 답변하세요. 뉴스에 없는 내용은 추측하지 마세요.
답변은 5문장 이내로 간결하게 작성하세요.
투자 판단(매수/매도)을 할만하다, 할만하지 않다 정도만 전달하고, 정보만 전달하세요.
그리고 지금 주가 흐름도 설명하시오.

{history}참고 뉴스:
{context_text}

질문: {question}

답변:""",
    partial_variables={"history": ""},
)


def build_stock_news_chain(llm) -> Runnable:
    """``{"context_text", "question", "history"}``를 받는 체인을 만든다.

    ``history``는 생략할 수 있고, 생략하면 빈 문자열로 채워진다.
    """
    return STOCK_NEWS_PROMPT | llm | StrOutputParser()
