FROM docker.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8

# GH200 = Hopper SM90a
ENV TORCH_CUDA_ARCH_LIST="9.0a" \
    MAX_JOBS="${MAX_JOBS}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
    # \
    # # If fa3-fwd isn't present, force safe fallback for Omni diffusion attention
    # DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA

# Keep host NVIDIA driver libraries first to avoid CUDA Error 803
# from loading cuda-compat stubs in /usr/local/cuda/compat.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash bash-completion readline-common less vim-tiny\
 && rm -rf /var/lib/apt/lists/*
#  RUN apt-get update 
# ---- OS deps ----
RUN apt-get update &&  apt-get install -y --no-install-recommends \
    bash readline-common less \
    git curl ca-certificates \
    python3 python3-pip python3-dev python3-venv \
    build-essential ninja-build cmake pkgconf \
    sox libsox-fmt-all ffmpeg \
    libnuma1 libnuma-dev \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update 

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Create isolated Python env
RUN python3 -m venv /opt/venv
ENV PATH="/root/.local/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"

RUN which python && which python3 && which pip && python --version
# Install tooling INSIDE venv (not system Python)
RUN /opt/venv/bin/python -m pip install --upgrade pip wheel "setuptools<=80"
# ---- Python tooling ----
# RUN python3 -m pip install -U pip wheel "setuptools<=80"


RUN /opt/venv/bin/python -m pip install \
  torch==2.7.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

ENV VLLM_PRECOMPILED_WHEEL_COMMIT=b834b4cbf1d5094affdf231df2be86920610d83e
RUN git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git /tmp/vllm-src && \
    /opt/venv/bin/python -m pip install -r /tmp/vllm-src/requirements/build.txt && \
    /opt/venv/bin/python -m pip install -r /tmp/vllm-src/requirements/cuda.txt && \
    VLLM_TARGET_DEVICE=cuda /opt/venv/bin/python -m pip install /tmp/vllm-src -v --no-build-isolation && \
    rm -rf /tmp/vllm-src && \
    /opt/venv/bin/python -m pip install git+https://github.com/huggingface/transformers && \
    /opt/venv/bin/python -m pip install accelerate && \
    /opt/venv/bin/python -m pip install qwen-omni-utils -U
RUN /opt/venv/bin/python -m pip install transformers==4.57.0
RUN /opt/venv/bin/python -m pip install -U flash-attn --no-build-isolation
