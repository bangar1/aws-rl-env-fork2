# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=aws_rl_env

# Copy environment code (always at root of build context)
COPY . /app/env

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies using uv sync
# If uv.lock exists, use it; otherwise resolve on the fly
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --extra dev --no-install-project --no-editable; \
    else \
    uv sync --extra dev --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
    uv sync --frozen --extra dev --no-editable; \
    else \
    uv sync --extra dev --no-editable; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the uv-managed Python interpreter from builder
COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Install AWS CLI
RUN apt-get update && \
    apt-get install -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*

# Configure AWS CLI to point to aws_infra (MiniStack) and use dummy credentials
RUN mkdir -p /root/.aws && \
    printf '[default]\nregion = us-east-1\noutput = json\n' > /root/.aws/config && \
    printf '[default]\naws_access_key_id = test\naws_secret_access_key = test\n' > /root/.aws/credentials
ENV AWS_ENDPOINT_URL=http://localhost:4566

# Enable the web interface for OpenEnv (if applicable)
# ENV ENABLE_WEB_INTERFACE=true

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"


# DEV_MODE=1 enables live reload via --reload flag
ENV DEV_MODE=1

ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Entrypoint: start aws_infra in background, then run the FastAPI server
CMD ["sh", "-c", "aws_infra -d & sleep 2 && uvicorn server.app:app --host 0.0.0.0 --port 8000 $([ \"$DEV_MODE\" = '1' ] && echo '--reload --reload-dir /app/env')"]