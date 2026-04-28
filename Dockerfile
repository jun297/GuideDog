FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV UV_LINK_MODE=copy

# System dependencies (ffmpeg, PyAV, decord, opencv, general build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential cmake pkg-config \
    ffmpeg \
    libavcodec-dev libavformat-dev libswscale-dev libavdevice-dev \
    libavfilter-dev libavutil-dev libswresample-dev \
    libsm6 libxext6 libxrender-dev \
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up Python 3.11 via uv
RUN uv python install 3.11

WORKDIR /workspace

# Create venv
RUN uv venv --python 3.11 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install torch first (CUDA 12.1 wheels)
RUN uv pip install torch>=2.1.0 torchvision>=0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy dependency files for layer caching
COPY pyproject.toml setup.py ./
COPY lmms_eval/__init__.py lmms_eval/__init__.py
COPY distribution_analysis/requirements.txt distribution_analysis/requirements.txt

# Install project in editable mode
RUN uv pip install -e ".[metrics,gemini,qwen]"

# Additional pins from requirements.txt
RUN uv pip install \
    absl-py rouge_score bert_score \
    qwen-vl-utils google-generativeai \
    "transformers>=4.39.2,<4.49"

# Distribution analysis deps
RUN uv pip install -r distribution_analysis/requirements.txt

# Copy full project (overridden by volume mount in dev)
COPY . .

# Re-install in editable mode with full source
RUN uv pip install -e . --no-deps

CMD ["bash"]
