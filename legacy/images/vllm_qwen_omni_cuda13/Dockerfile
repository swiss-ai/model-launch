FROM docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG TARGET_PLATFORM=aarch64

ARG VLLM_VERSION=0.18.0

# GH200 = Hopper SM90a
ENV TORCH_CUDA_ARCH_LIST="9.0a" \
    MAX_JOBS="${MAX_JOBS}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Keep host NVIDIA driver libraries first to avoid CUDA Error 803
# from loading cuda-compat stubs in /usr/local/cuda/compat.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib

WORKDIR /opt

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash bash-completion readline-common less vim-tiny \
 && rm -rf /var/lib/apt/lists/*

# ---- OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash readline-common less \
    git curl ca-certificates \
    python3 python3-pip python3-dev python3-venv \
    build-essential ninja-build cmake pkgconf \
    sox libsox-fmt-all ffmpeg \
    libnuma1 libnuma-dev \
 && rm -rf /var/lib/apt/lists/*

# Install miniconda (same CUDA13 pattern as vllm_cuda13 image).
ENV PATH=/opt/conda/bin:${PATH}
ADD "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh" /root/miniconda.sh
RUN chmod +x /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /opt/conda && \
    rm /root/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython git compilers make && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create isolated Python env
RUN python3 -m venv /opt/venv
ENV PATH="/root/.local/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"

RUN which python && which python3 && which pip && python --version
# Install tooling INSIDE venv (not system Python)
RUN /opt/venv/bin/python -m pip install --upgrade pip wheel "setuptools<70.0.0"

RUN /opt/venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130

# Ensure ray is available in the same runtime env as vllm.
RUN /opt/venv/bin/python -m pip install "ray[default]"

# CUDA 13 compatibility: expose cuDNN/NCCL headers where extension builds can find them.
RUN CUDNN_DIR=$(dirname $(find /usr -name 'cudnn.h' -print -quit)) && \
    ln -sf ${CUDNN_DIR}/cudnn*.h /opt/venv/include/ && \
    ln -sf $(find /opt/venv/lib/python${PYTHON_VERSION}/site-packages -path '*/nvidia/nccl/include/nccl.h' -print -quit) /opt/venv/include/nccl.h


RUN /opt/venv/bin/python -m pip install \
  https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu130-cp38-abi3-manylinux_2_35_aarch64.whl \
  --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH}

RUN git clone https://github.com/vllm-project/vllm-omni.git && \
    cd vllm-omni && \
    pip install -e . && \
    cd .. 

RUN  /opt/venv/bin/python -m pip install git+https://github.com/huggingface/transformers && \
    /opt/venv/bin/python -m pip install accelerate && \
    /opt/venv/bin/python -m pip install qwen-omni-utils -U

RUN /opt/venv/bin/python -m pip install -U flash-attn --no-build-isolation

# CUDA 13 toolkit provides cuBLAS; pip's nvidia-cublas can conflict with it.
RUN /opt/venv/bin/python -m pip uninstall -y nvidia-cublas || true
