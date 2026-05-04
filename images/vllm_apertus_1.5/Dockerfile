FROM docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG VLLM_MAIN_CUDA_VERSION=13.0

ENV TORCH_CUDA_ARCH_LIST="9.0a" \
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash bash-completion ca-certificates ccache curl git less vim-tiny \
    build-essential cmake ninja-build pkgconf \
    libibverbs-dev libnuma1 libnuma-dev \
    python3 python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN python --version && python -m pip --version && uv --version

# hadolint ignore=DL3013
RUN python -m pip install --upgrade \
    pip \
    wheel \
    "setuptools>=77.0.3,<81.0.0"

RUN uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 && \
    uv pip install --python /opt/venv/bin/python \
        "packaging>=24.2" \
        ninja \
        "cmake>=3.26.1" \
        "setuptools-scm>=8" \
        "jinja2>=3.1.6" \
        regex \
        build \
        "protobuf>=5.29.6,!=6.30.*,!=6.31.*,!=6.32.*,!=6.33.0.*,!=6.33.1.*,!=6.33.2.*,!=6.33.3.*,!=6.33.4.*" \
        "ray[default]" && \
    python -c "import torch; print('Torch:', torch.__version__); print('Torch CUDA:', torch.version.cuda)"

# CUDA 13 compatibility: expose cuDNN/NCCL headers where extension builds can find them.
RUN CUDNN_DIR=$(dirname "$(find /usr -name cudnn.h -print -quit)") && \
    find "${CUDNN_DIR}" -maxdepth 1 -name 'cudnn*.h' -exec ln -sf {} /opt/venv/include/ \; && \
    NCCL_H=$(find /opt/venv/lib/python${PYTHON_VERSION}/site-packages -path '*/nvidia/nccl/include/nccl.h' -print -quit) && \
    if [ -n "${NCCL_H}" ]; then ln -sf "${NCCL_H}" /opt/venv/include/nccl.h; fi

WORKDIR /workspace

RUN git clone --branch apertus_integration https://github.com/swiss-ai/vllm.git /workspace/vllm && \
    git clone https://github.com/baaivision/Emu3.5.git /workspace/Emu3.5 && \
    git clone https://github.com/swiss-ai/benchmark-audio-tokenizer.git /workspace/benchmark-audio-tokenizer && \
    git -C /workspace/benchmark-audio-tokenizer submodule update --init --recursive

ENV VLLM_REPO=/workspace/vllm \
    EMU35=/workspace/Emu3.5 \
    ec=/workspace/benchmark-audio-tokenizer \
    PYTHONPATH=/workspace/vllm:${PYTHONPATH} \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/venv/lib/python3.12/site-packages/torch/lib \
    VLLM_APERTUS_EMU35_CODEBASE=/workspace/Emu3.5

WORKDIR /workspace/vllm

RUN set -eu; \
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"; \
    mkdir -p "${UV_CACHE_DIR}"; \
    export VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-cu130}"; \
    export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-13.0}"; \
    export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_aarch64.whl}"; \
    if [ -z "${VLLM_PRECOMPILED_WHEEL_COMMIT:-}" ] && VLLM_GIT_COMMIT="$(git -C "${VLLM_REPO}" rev-parse HEAD 2>/dev/null)"; then \
    export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_GIT_COMMIT}"; \
    fi; \
    VLLM_USE_PRECOMPILED=1 uv pip install \
    --python /opt/venv/bin/python \
    --reinstall-package vllm \
    --no-build-isolation \
    --editable . \
    --torch-backend=auto;

RUN <<'EOF'
python - <<'PY'
import torch

print(f"Torch runtime: {torch.__version__} (CUDA {torch.version.cuda})")
try:
    import vllm._C  # noqa: F401
except Exception as exc:
    if "libcuda.so.1" in str(exc):
        print("Skipping strict vllm._C import check during image build: libcuda.so.1 is unavailable in build environment.")
    else:
        raise
else:
    print("vLLM native extension import check passed.")
PY
uv pip install --python /opt/venv/bin/python scipy ruff tblib pytest
printf '%s\n' \
    'Production CUDA 13 image with local vLLM source built inline in Dockerfile.' \
    'Included paths:' \
    '  /workspace/vllm' \
    '  /workspace/Emu3.5 (EMU35)' \
    '  /workspace/benchmark-audio-tokenizer (VLLM_APERTUS_AUDIO_TOKENIZER_CODEBASE)' \
    > /etc/motd
EOF
