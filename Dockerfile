FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first so this layer is cached while source changes
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev

RUN uv run python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Dongju-00/eric-chatagent', local_dir='.')"

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run","python", "main.py"]