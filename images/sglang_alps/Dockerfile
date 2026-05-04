# SGLang on the CSCS Alps extended NGC PyTorch base.
#
# Base image courtesy of CSCS — see https://github.com/eth-cscs/alps-extended-images
# for the Slingshot-tuned NGC PyTorch builds (NCCL, NVSHMEM, OpenMPI, libfabric/CXI).

# Declared globally so BuildKit can resolve them in FROM across both stages.
ARG FA3_BUILDER_BASE=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.01-py3-alps4

# Build FA3 on CUDA 13.0 — nvcc 13.1 (in 26.01-alps4) breaks FA3's hopper
# kernels on aarch64. Binary built against 13.0 runs fine on the 13.1 runtime.
FROM ${FA3_BUILDER_BASE} AS fa3-builder

ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION_PATH=cu130
ARG FLASH_ATTN_3_SHA="92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
# Default sized for GH-hosted ubuntu-24.04-arm (4 cores / 16 GB) — each FA3
# hopper .cu peaks ~4-8 GB during nvcc. Override for beefier build hosts.
ARG FA3_MAX_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y \
        python3.12 python3.12-dev python3-pip git ninja-build \
        build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3013
RUN pip install --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1 && \
    pip install packaging ninja wheel

WORKDIR /tmp/build
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/Dao-AILab/flash-attention.git && \
    git -C flash-attention fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention checkout ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention submodule update --init --depth 1

# Build FA3 and restructure the egg in one layer so the intermediate egg
# directory doesn't bloat the builder stage's exported artifact.
WORKDIR /tmp/build/flash-attention/hopper
RUN FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
        MAX_JOBS=${FA3_MAX_JOBS} python3.12 setup.py install && \
    SP=$(python3.12 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    mkdir -p /opt/flash_attn_3 && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_3/* /opt/flash_attn_3/ && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_interface.py /opt/flash_attn_3/ && \
    touch /opt/flash_attn_3/__init__.py


FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION_PATH=cu130

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1

# Two-step install: pin cu130 torch first via --torch-backend, then install
# sglang + router + cu130 sgl-kernel. The second resolve doesn't disturb the
# already-installed torch (its torch==2.9.1 constraint matches the cu130
# local-version wheel), and sglang's transitive torchao dep resolves from
# PyPI normally — unlike the single-resolve form, where --torch-backend's
# scope confuses torchao resolution.
# Drop pip NCCL/cuBLAS in the same layer so torch dlopens the alps-built
# libnccl from /usr and the CUDA 13 toolkit's cuBLAS.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1 \
        --torch-backend=${CUDA_VERSION_PATH} && \
    uv pip install --system --prerelease=allow \
        "sgl-kernel @ https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp312-abi3-manylinux2014_aarch64.whl" \
        "sglang[all]==0.5.10.post1" \
        sglang-router==0.3.2 && \
    uv pip uninstall --system nvidia-nccl-cu13 nvidia-nccl-cu12 \
                              nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas

# Drop in flash_attn_3 prebuilt against CUDA 13.0 in the fa3-builder stage.
COPY --from=fa3-builder /opt/flash_attn_3 /tmp/flash_attn_3_built
RUN SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    rm -rf "${SP}/flash_attn_3" && \
    mv /tmp/flash_attn_3_built "${SP}/flash_attn_3"

# Verify curl (for router health checks) and assert torch is a CUDA build.
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import sys, torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
if torch.version.cuda is None:
    sys.exit(f"expected a CUDA torch build, got {torch.__version__}")
print(f"nccl: {torch.cuda.nccl.version()}")
PY
