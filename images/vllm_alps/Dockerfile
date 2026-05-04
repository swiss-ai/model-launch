# vLLM on the CSCS Alps extended NGC PyTorch base.
#
# The base ships an NGC PyTorch image with the Slingshot HPC stack baked in
# (libfabric/CXI, NCCL 2.28.7, NVSHMEM, OpenMPI 5, aws-ofi-nccl, GDRCopy/XPMEM)
# and an /opt/alps/env/alps-runtime.env that configures Slingshot defaults.
# See: https://github.com/eth-cscs/alps-extended-images
#
# CSCS internal mirror (faster on Alps): jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps4
# Override via: --build-arg BASE_IMAGE=...
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

# vLLM nightly cu130 wheel. NGC PyTorch 26.02 already provides torch 2.10
# linked against the alps-built NCCL on /usr; --torch-backend=auto adapts.
# 26.02-py3-alps4's Python 3.12 is Debian-marked externally-managed
# (PEP 668); the BREAK_SYSTEM_PACKAGES envs above let pip/uv write into it.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system -U vllm \
        --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
        --torch-backend=auto

# Drop pip-installed NCCL wheels so torch dlopens the alps-built libnccl
# on the system path (avoids version-skew with the slingshot-tuned build).
RUN pip uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 || true

# nvidia-cublas pip wheel conflicts with the CUDA 13 toolkit's cuBLAS in the
# NGC base; the toolkit version is already correct, so drop the wheel.
RUN pip uninstall -y nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas || true

RUN pip install --no-cache-dir "ray[default]==2.55.0"

# Verify the NGC base provides curl (used by router health checks at runtime)
# and print the loaded NCCL version (expect 2.28.7 from the alps build).
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"nccl: {torch.cuda.nccl.version()}")
PY

WORKDIR /opt
