# Multi-stage build: resolve dependencies with uv, ship a slim runtime image.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer: cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Project layer.
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

RUN groupadd --system arb && useradd --system --gid arb --create-home arb

WORKDIR /app
COPY --from=builder --chown=arb:arb /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER arb

ENTRYPOINT ["arb"]
CMD ["info"]
