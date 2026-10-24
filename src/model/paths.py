"""프로젝트 경로를 한곳에서 해석한다.

디렉터리 깊이를 숫자로 가정하는 ``parents[N]`` 방식은 파일을 옮기면 조용히
엉뚱한 경로를 가리키기 때문에 쓰지 않는다. 대신:

- ``MODEL_ROOT``는 이 파일이 ``src/model/`` 안에 있다는 사실만으로 바로 얻는다.
- ``PROJECT_ROOT``는 ``pyproject.toml``이 있는 가장 가까운 상위 디렉터리를 찾는다.

모든 경로는 환경변수로 덮어쓸 수 있다:

===========================  ======================================
환경변수                      기본값
===========================  ======================================
``ERIC_PROJECT_ROOT``        ``pyproject.toml``이 있는 상위 디렉터리
``ERIC_MODEL_ROOT``          ``<project>/src/model``
``ERIC_DATA_DIR``            ``<model>/data``
``ERIC_CHECKPOINT_DIR``      ``<model>/checkpoints``
``ERIC_CHROMA_PATH``         ``<project>/storage/chroma``
``ERIC_SP_PREFIX``           ``<data>/sp_korean`` (확장자 없는 prefix)
===========================  ======================================
"""

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "MODEL_ROOT",
    "DATA_DIR",
    "CHECKPOINT_DIR",
    "CHROMA_PATH",
    "SP_PREFIX",
]

ROOT_MARKER = "pyproject.toml"


def find_project_root(start: Path) -> Path:
    """``start``에서 위로 올라가며 ROOT_MARKER가 있는 디렉터리를 찾는다."""
    for candidate in (start, *start.parents):
        if (candidate / ROOT_MARKER).is_file():
            return candidate

    # 마커를 못 찾는 배치(예: 패키지만 복사된 환경)에서는 src/의 부모로 되돌린다.
    # 이 경우에도 ERIC_PROJECT_ROOT로 명시해 덮어쓸 수 있다.
    return start.parent.parent


def _env_path(name: str, default: Path) -> Path:
    """환경변수가 있으면 그 값을, 없으면 default를 Path로 돌려준다."""
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser().resolve()


# 이 파일은 src/model/paths.py이므로, 부모 디렉터리가 곧 모델 패키지 루트다.
_MODEL_ROOT_DEFAULT = Path(__file__).resolve().parent

PROJECT_ROOT = _env_path("ERIC_PROJECT_ROOT", find_project_root(_MODEL_ROOT_DEFAULT))
MODEL_ROOT = _env_path("ERIC_MODEL_ROOT", _MODEL_ROOT_DEFAULT)

DATA_DIR = _env_path("ERIC_DATA_DIR", MODEL_ROOT / "data")
CHECKPOINT_DIR = _env_path("ERIC_CHECKPOINT_DIR", MODEL_ROOT / "checkpoints")

# Chroma는 저장소 루트의 storage/를 쓴다. docker-compose.yml이 ./storage를
# /app/storage로 마운트하므로 컨테이너에서도 같은 위치로 해석된다.
CHROMA_PATH = _env_path("ERIC_CHROMA_PATH", PROJECT_ROOT / "storage" / "chroma")

# SentencePiece는 확장자 없는 prefix를 받는다 (<prefix>.model / <prefix>.vocab).
SP_PREFIX = _env_path("ERIC_SP_PREFIX", DATA_DIR / "sp_korean")
