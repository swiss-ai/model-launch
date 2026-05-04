# vLLM on the CSCS Alps extended NGC PyTorch base.
#
# Base image courtesy of CSCS — see https://github.com/eth-cscs/alps-extended-images
# for the Slingshot-tuned NGC PyTorch builds (NCCL, NVSHMEM, OpenMPI, libfabric/CXI).

ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.02-py3-alps4
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1

# Pin --torch-backend=cu130 explicitly: `auto` falls back to a CPU wheel when
# nvcc/nvidia-smi aren't visible on the build host (e.g. GH-hosted runners).
# Drop pip NCCL/cuBLAS in the same layer so torch dlopens the alps-built
# libnccl from /usr and the CUDA 13 toolkit's cuBLAS, and the wheels don't
# linger as dead weight in an earlier layer.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system -U vllm "ray[default]==2.55.0" \
        --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
        --torch-backend=cu130 && \
    uv pip uninstall --system nvidia-nccl-cu13 nvidia-nccl-cu12 \
                              nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas

# Verify curl (for router health checks) and assert torch is a CUDA build.
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import sys, torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
if torch.version.cuda is None:
    sys.exit(f"expected a CUDA torch build, got {torch.__version__}")
print(f"nccl: {torch.cuda.nccl.version()}")
PY
