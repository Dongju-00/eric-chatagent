import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """FastAPI 서버 실행."""

    uvicorn.run(
        "app.app:app",
        host=host,
        port=port,
        reload=reload,
        app_dir=str(SRC),      # reload 모드에서도 src 기준으로 import
    )

if __name__ == "__main__":
    serve()