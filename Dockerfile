FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src ./src
RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["uvicorn", "token_daddy.main:app", "--host", "0.0.0.0", "--port", "8000"]
