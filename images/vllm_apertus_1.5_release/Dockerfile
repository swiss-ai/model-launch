FROM docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG VLLM_MAIN_CUDA_VERSION=13.0
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX"
ARG VLLM_REPO_COMMIT=ae6170f874a7c66512bd24aa714b3a02f8a61838
ARG SWISS_AI_VLLM_COMMIT=f2225709c6a85d40ca5c0435625bb5856ed40e9d
ARG TRANFORMERS_COMMIT=0a2293153f548ae0fe8001ee89134660392b2f25
ARG TARGETARCH

ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION}" \
    VLLM_PRECOMPILED_WHEEL_VARIANT="${CUDA_VERSION_PATH}"

# Keep host NVIDIA driver libraries first to avoid CUDA Error 803
# from loading cuda-compat stubs in /usr/local/cuda/compat.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash bash-completion ca-certificates ccache curl git less vim-tiny \
    build-essential cmake ninja-build pkgconf \
    libibverbs-dev libnuma1 libnuma-dev \
    python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir "uv==0.11.30"
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN python --version && python -m pip --version && uv --version

# hadolint ignore=DL3013
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade \
    pip \
    wheel \
    "setuptools>=77.0.3,<81.0.0"

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        "packaging>=24.2" \
        ninja \
        "cmake>=3.26.1" \
        "setuptools-scm>=8" \
        "jinja2>=3.1.6" \
        setuptools_rust \
        regex \
        build \
        "protobuf>=5.29.6,!=6.30.*,!=6.31.*,!=6.32.*,!=6.33.0.*,!=6.33.1.*,!=6.33.2.*,!=6.33.3.*,!=6.33.4.*" \
        "ray[default]==2.56.0" && \
    python -c "import torch; print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda)"

# CUDA 13 compatibility: expose cuDNN/NCCL headers where extension builds can find them.
RUN CUDNN_H="$(find /usr -name cudnn.h -print -quit)" && \
    [ -n "${CUDNN_H}" ] || { echo "cudnn.h not found" >&2; exit 1; } && \
    CUDNN_DIR="$(dirname "${CUDNN_H}")" && \
    find "${CUDNN_DIR}" -maxdepth 1 -name 'cudnn*.h' -exec ln -sf {} /opt/venv/include/ \; && \
    NCCL_H=$(find /opt/venv/lib/python${PYTHON_VERSION}/site-packages -path '*/nvidia/nccl/include/nccl.h' -print -quit) && \
    if [ -n "${NCCL_H}" ]; then ln -sf "${NCCL_H}" /opt/venv/include/nccl.h; fi

WORKDIR /workspace

RUN git clone --branch apertus-1-5 https://github.com/swiss-ai/vllm.git /workspace/vllm && \
    git -C /workspace/vllm checkout ${SWISS_AI_VLLM_COMMIT}

RUN git clone --branch feature/apertus_1p5_pipeline https://github.com/RaphaelKreft/transformers.git /workspace/transformers && \
    git -C /workspace/transformers checkout ${TRANSFORMERS_COMMIT}

ENV VLLM_REPO=/workspace/vllm \
    PYTHONPATH=/workspace/vllm:${PYTHONPATH} \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib

WORKDIR /workspace/transformers

RUN uv pip install \
    --python /opt/venv/bin/python \
    --no-deps .

WORKDIR /workspace/vllm

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    set -eu; \
    export VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-cu130}"; \
    export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-13.0}"; \
    ARCH="${TARGETARCH:-$(uname -m)}"; \
    case "$ARCH" in \
        amd64|x86_64) WHEEL_ARCH="x86_64" ;; \
        arm64|aarch64) WHEEL_ARCH="aarch64" ;; \
        *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;; \
    esac; \
    export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://wheels.vllm.ai/${VLLM_REPO_COMMIT}/vllm-0.23.1rc1.dev977%2Bgae6170f87-cp38-abi3-manylinux_2_28_${WHEEL_ARCH}.whl}"; \
    export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-${VLLM_REPO_COMMIT}}"; \
    VLLM_USE_PRECOMPILED=1 uv pip install \
    --python /opt/venv/bin/python \
    --reinstall-package vllm \
    --no-build-isolation ".[audio]" \
    --torch-backend=auto;

RUN uv pip install --python /opt/venv/bin/python scipy ruff tblib pytest  "fastapi[standard]>=0.115.0,<0.137.0"
