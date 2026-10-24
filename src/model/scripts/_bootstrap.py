"""스크립트를 직접 실행할 때 ``src/``를 ``sys.path``에 올린다.

``src/model/scripts/*.py``는 ``python src/model/scripts/train_xxx.py`` 처럼
직접 실행되므로 ``model`` 패키지가 import 경로에 없다. 여기서 저장소 루트를
찾아 ``src/``를 넣어준 뒤에야 ``model.paths``를 import할 수 있다.

경로 자체의 기본값과 환경변수 처리는 모두 :mod:`model.paths`에 있다.
이 모듈은 ``src/``를 찾는 일만 한다.
"""

import os
import sys
from pathlib import Path

ROOT_MARKER = "pyproject.toml"


def find_project_root() -> Path:
    """ERIC_PROJECT_ROOT가 있으면 그 값을, 없으면 ROOT_MARKER를 찾아 올라간다."""
    env = os.getenv("ERIC_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ROOT_MARKER).is_file():
            return candidate

    raise RuntimeError(
        f"{ROOT_MARKER}을 찾지 못했습니다. ERIC_PROJECT_ROOT를 설정하세요."
    )


def ensure_src_on_path() -> Path:
    """저장소의 ``src/``를 sys.path 맨 앞에 넣고 저장소 루트를 돌려준다."""
    root = find_project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root
