# SGLang on the CSCS Alps extended NGC PyTorch base.
#
# Base image courtesy of CSCS — see https://github.com/eth-cscs/alps-extended-images
# for the Slingshot-tuned NGC PyTorch builds (NCCL, NVSHMEM, OpenMPI, libfabric/CXI).

ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.01-py3-alps4

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION_PATH=cu130
ARG SGLANG_KERNEL_VERSION=0.4.1
ARG SGLANG_VERSION=0.5.10.post1
ARG SGLANG_ROUTER_VERSION=0.3.2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1

# Pin cu130 torch first via --torch-backend, then install the cu130 sglang-kernel
# wheel from sgl-project/whl (PyPI only ships generic manylinux wheels — the
# cu130 local-version variants are GitHub-only). flash-attn-4 is a pure-Python
# wheel that JITs hopper/blackwell kernels via nvidia-cutlass-dsl at runtime,
# so no from-source FA3 build stage is needed.
# Drop pip NCCL/cuBLAS in the same layer so torch dlopens the alps-built
# libnccl from /usr and the CUDA 13 toolkit's cuBLAS.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1 \
        --torch-backend=${CUDA_VERSION_PATH} && \
    uv pip install --system --no-deps --force-reinstall \
        "sglang-kernel @ https://github.com/sgl-project/whl/releases/download/v${SGLANG_KERNEL_VERSION}/sglang_kernel-${SGLANG_KERNEL_VERSION}+${CUDA_VERSION_PATH}-cp310-abi3-manylinux2014_aarch64.whl" && \
    uv pip install --system --prerelease=allow \
        "sglang[all]==${SGLANG_VERSION}" \
        "sglang-router==${SGLANG_ROUTER_VERSION}" \
        "flash-attn-4>=4.0.0b4" \
        "nvidia-cutlass-dsl>=4.4.1" && \
    uv pip uninstall --system nvidia-nccl-cu13 nvidia-nccl-cu12 \
                              nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas

# Verify curl (for router health checks), assert torch is a CUDA build, and
# confirm sgl_kernel imports cleanly (the failure mode this image fixes).
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import sys, torch, sgl_kernel
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"sgl_kernel ok: {sgl_kernel.__file__}")
if torch.version.cuda is None:
    sys.exit(f"expected a CUDA torch build, got {torch.__version__}")
print(f"nccl: {torch.cuda.nccl.version()}")
PY
