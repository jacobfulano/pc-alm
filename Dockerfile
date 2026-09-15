FROM us-docker.pkg.dev/flourish-gpu/flourish/flourish-base:base-3

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-group cpu --group gpu --extra test

ENV PATH=/app/.venv/bin:$PATH \
    UV_NO_SYNC=1 \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

COPY pcalm/ ./pcalm/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY train.py LICENSE README.md ./

RUN if [ -n "${UV_PYTHON:-}" ]; then \
      echo "error: UV_PYTHON must be empty in a deploy runtime image" >&2; \
      exit 1; \
    fi
