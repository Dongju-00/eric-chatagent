"""서빙 리소스를 한곳에서 만들어 주입한다.

예전에는 ``baseline.py``와 ``graph.py``가 모듈 레벨에서 각자
``ChromaNewsVectorStore()``와 LLM을 만들었다. 그래서 import만으로 가중치가
적재되고, ``KoreanGPT.pt``(임베더)가 두 벌 올라갔다.

지금은 ``build_resources()``만 리소스를 만든다. 호출하면
``torch.load``가 3회(임베더 1 + LLM 2), ``chromadb.PersistentClient``가 1회
일어난다. 서버에서는 ``app.py``의 ``lifespan``이 한 번 호출하고,
``build_graph(resources)``로 주입한다.
"""

from dataclasses import dataclass
from typing import Any

from model.rag.baseline import build_llm
from model.llm.openrouter import build_openrouter_llm
from model.store.vector_store import ChromaNewsVectorStore

__all__ = ["AppResources", "build_resources"]


@dataclass
class AppResources:
    """프로세스당 한 벌만 존재해야 하는 리소스 묶음."""

    store: ChromaNewsVectorStore
    news_llm: Any
    small_talk_llm: Any
    fallback_llm: Any = None


def build_resources() -> AppResources:
    """공유 리소스를 만든다. 서버 시작 시 1회만 호출된다.

    스토어는 1개만 만든다. ``ChromaNewsVectorStore`` 가 내부에서
    ``KoreanGPTEmbedder`` 를 만들고 그 임베더가 ``KoreanGPT.pt``(160MB)를
    적재하므로, 2개를 만들면 같은 가중치가 2벌 올라간다.

    ``news_llm`` 과 ``small_talk_llm`` 은 서로 다른 파인튜닝 체크포인트라
    공유 대상이 아니다. 따라서 ``torch.load`` 는 3회가 정상이다.

    ``fallback_llm`` 은 OpenRouter 클라이언트다. 키가 없으면 ``None`` 이고,
    그때 ``fallback_node`` 는 고정 문구로 답한다. 가중치를 적재하지 않으므로
    위 3회에 영향이 없다.
    """
    return AppResources(
        store=ChromaNewsVectorStore(),
        news_llm=build_llm("stock_news"),
        small_talk_llm=build_llm("small_talk"),
        fallback_llm=build_openrouter_llm(),
    )
