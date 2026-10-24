import contextlib
import hashlib
from unittest import mock
"""테스트용 가짜 리소스.

가중치나 Chroma 없이 `build_graph` / `lifespan` 의 주입 경로만 검증하기 위한
것이다. `news_llm` 은 체인(`PROMPT | llm | StrOutputParser()`)에 들어가므로
Runnable 이어야 해서 실제 LangChain LLM 을 상속한다. `store` 는 노드 클로저가
메서드만 부르므로 평범한 객체로 충분하다.
"""

from langchain_core.language_models.llms import LLM


class FakeLLM(LLM):
    """고정 문자열을 돌려주는 LLM. 받은 프롬프트를 기록한다."""

    answer: str = "FAKE-ANSWER"
    prompts: list = None

    def __init__(self, answer="FAKE-ANSWER", **kwargs):
        super().__init__(**kwargs)
        self.answer = answer
        self.prompts = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeEmbedder:
    """KoreanGPT.pt(160MB) 적재 없이 결정적인 임베딩을 돌려준다.

    실제 Chroma를 임시 디렉터리에 띄워 검색 범위와 TTL 삭제를 검증할 때 쓴다.
    텍스트 해시를 각 차원에 흘려넣어 같은 텍스트면 같은 벡터가 나오게 한다.
    """

    DIM = 8

    def embed_text(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [h[i] / 255.0 for i in range(self.DIM)]

    def embed_texts(self, texts):
        return [self.embed_text(t) for t in texts]


class FakeStore:
    """`ChromaNewsVectorStore` 중 그래프 경로가 쓰는 메서드만 흉내낸다.

    수집 오케스트레이션이 `collect_and_index` 로 나갔으므로 저장소 fake 만으로는
    네트워크를 막을 수 없다. 그래프 주입 테스트는 `fetch_candidates` /
    `extract_and_chunk` 를 함께 patch 한다.
    """

    def __init__(self, documents=("가짜 뉴스 본문",)):
        self.documents = list(documents)
        self.build_calls = []
        self.index_calls = []
        self.search_calls = []

    def save_question(self, question):
        self.build_calls.append({"query": question})
        return "FAKE-QID"

    def index_chunks(self, question_id, chunks):
        self.index_calls.append((question_id, chunks))
        return len(chunks)

    def search_similar_news(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "documents": [self.documents],
            "metadatas": [[{"title": f"제목 {i}"} for i in range(len(self.documents))]],
            "distances": [[0.5] * len(self.documents)],
        }


def fake_resources(news_answer="NEWS-FAKE", small_talk_answer="SMALLTALK-FAKE",
                   documents=("가짜 뉴스 본문",)):
    """가짜 객체로 채운 `AppResources`.

    `AppResources` 는 타입 강제를 하지 않는 dataclass 라서 그대로 쓸 수 있다.
    """
    from model.resources import AppResources

    return AppResources(
        store=FakeStore(documents=documents),
        news_llm=FakeLLM(answer=news_answer),
        small_talk_llm=FakeLLM(answer=small_talk_answer),
        fallback_llm=None,
    )


@contextlib.contextmanager
def no_network_collect(chunks=None):
    """수집 단계를 가짜로 바꿔 네트워크를 타지 않게 한다.

    `collect_and_index` 가 `fetch_candidates`(네이버 API)와
    `extract_and_chunk`(기사 본문 HTTP)를 호출하므로, 저장소 fake 만으로는
    단위 테스트가 네트워크를 벗어나지 못한다.
    """
    if chunks is None:
        chunks = [{
            "text": "제목: 가짜 제목\n작성일: Thu, 23 Oct 2026 00:00:00 +0900\n내용: 가짜 기사 본문",
            "metadata": {"title": "가짜 제목", "chunk_id": 0},
        }]
    with mock.patch("model.news.collect.fetch_candidates", return_value=[{"title": "가짜 제목"}]), \
         mock.patch("model.news.collect.extract_and_chunk", return_value=chunks):
        yield
