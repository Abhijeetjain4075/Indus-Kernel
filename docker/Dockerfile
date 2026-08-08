# Indus Kernel — production Dockerfile.
# Multi-stage build for a slim runtime image.

# ---- Stage 1: build ----
FROM python:3.12-slim AS builder

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Layer cache: deps first
COPY pyproject.toml uv.lock* ./
COPY packages/ ./packages/
RUN uv sync --frozen --no-install-project --no-dev

# Now install the project
COPY . .
RUN uv sync --frozen --no-dev

# ---- Stage 2: runtime ----
FROM python:3.12-slim AS runtime

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash indus

WORKDIR /app

# Copy the venv from builder
COPY --from=builder --chown=indus:indus /app /app

USER indus

EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# Default: start the kernel
CMD ["uv", "run", "ik-kernel", "dev"]
