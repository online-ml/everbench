# syntax=docker/dockerfile:1

FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts
COPY src ./src
COPY tasks ./tasks
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable \
    && useradd --create-home --uid 10001 everbench \
    && mkdir -p /data/archives \
    && chown -R everbench:everbench /app /data

USER everbench

EXPOSE 8000
VOLUME ["/data"]

CMD ["sh", "scripts/container-start.sh"]
