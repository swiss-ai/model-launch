FROM docker.io/nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MAX_JOBS=8
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG RUST_TOOLCHAIN=1.95.0
ARG VLLM_MAIN_CUDA_VERSION=13.0
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX"
ARG TARGETARCH

# Build vLLM from the swiss-ai mirror of the Infomaniak fork (apertus-1-5
# branch): the Infomaniak tip plus Apertus multimodal runtime fixes.
# https://github.com/swiss-ai/infomaniak-vllm/tree/apertus-1-5
ARG VLLM_REPO_URL=https://github.com/swiss-ai/infomaniak-vllm.git
ARG VLLM_REPO_BRANCH=apertus-1-5
ARG VLLM_REPO_COMMIT=aa64d5e6dca80cb4a9e3cb6ba4b3591a49d268a8

ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION}"

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

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Rust toolchain — the apertus-1-5 branch builds a Rust extension via setuptools-rust
# (rust-toolchain.toml pins channel 1.95).
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"
ENV PATH="/opt/cargo/bin:${PATH}"

# --seed installs pip/setuptools/wheel so the later `python -m pip` step and
# in-container debugging keep working (a bare uv venv ships without pip).
# Unlike python3 -m venv, uv does not create include/ — the cuDNN/NCCL
# header symlink step below needs it.
RUN /root/.local/bin/uv venv --seed --python /usr/bin/python3 /opt/venv && \
    mkdir -p /opt/venv/include
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

RUN python --version && python -m pip --version && uv --version && rustc --version

# hadolint ignore=DL3013
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade \
    pip \
    wheel \
    "setuptools>=77.0.3,<81.0.0"

# Build-time dependencies. Torch is pinned to 2.11.0 to match the branch's pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==2.11.0 \
        torchvision==0.26.0 \
        torchaudio==2.11.0 \
        "packaging>=24.2" \
        ninja \
        "cmake>=3.26.1" \
        "setuptools-scm>=8" \
        "setuptools-rust>=1.9.0" \
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

# Clone the branch and pin the exact commit for reproducibility.
RUN git clone --branch "${VLLM_REPO_BRANCH}" --single-branch "${VLLM_REPO_URL}" /workspace/vllm && \
    git -C /workspace/vllm checkout "${VLLM_REPO_COMMIT}" && \
    git -C /workspace/vllm rev-parse HEAD

# No PYTHONPATH here: the editable install already exposes the package. Forcing
# /workspace/vllm onto sys.path shadows the editable-install finder and hides
# the compiled extensions (vllm._C) built by setup.py.
ENV VLLM_REPO=/workspace/vllm \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib

WORKDIR /workspace/vllm

# Build and install vLLM from source (compiles CUDA + Rust extensions).
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    --mount=type=cache,target=/root/.cache/ccache,sharing=locked \
    uv pip install \
    --python /opt/venv/bin/python \
    --reinstall-package vllm \
    --no-build-isolation \
    --editable .

RUN uv pip install --python /opt/venv/bin/python scipy ruff tblib pytest "fastapi[standard]>=0.115.0,<0.137.0"

# set -e is required: without it a failing python check is swallowed and the
# image commits anyway (the trailing printf exits 0). The bash shebang keeps
# this working even if a builder ignores SHELL (OCI image format).
RUN <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo "Compiled extensions in source tree:"
find /workspace/vllm/vllm -maxdepth 1 -name '*.so' -print
python - <<'PY'
import torch

print(f"Torch runtime: {torch.__version__} (CUDA {torch.version.cuda})")

import importlib
import importlib.util

import vllm

print(f"vllm resolved from: {vllm.__file__}")

# This branch builds against the libtorch stable ABI, so the CUDA kernels are
# vllm._C_stable_libtorch / vllm._moe_C_stable_libtorch (see
# vllm/platforms/cuda.py import_kernels), not vllm._C.
REQUIRED = ["vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"]
missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Compiled vLLM extensions missing from image: {missing}")
try:
    for name in REQUIRED:
        importlib.import_module(name)
except ImportError as exc:
    # A missing .so must fail the build; only tolerate the driver being
    # absent from the build environment.
    if "libcuda.so.1" in str(exc):
        print("Extensions present; skipping strict import check: libcuda.so.1 is unavailable in build environment.")
    else:
        raise
else:
    print("vLLM native extension import check passed.")
PY
printf '%s\n' \
    'Production CUDA 13 image with Infomaniak vLLM (apertus-1-5 branch) built inline in Dockerfile.' \
    'Source: https://github.com/Infomaniak/vllm/tree/apertus-1-5' \
    'Included path: /workspace/vllm' \
    > /etc/motd
EOF
