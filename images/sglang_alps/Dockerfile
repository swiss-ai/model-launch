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
ARG FA3_MAX_JOBS=32

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# hadolint ignore=DL3008
RUN apt-get update && apt-get install --no-install-recommends -y \
        python3.12 python3.12-dev python3-pip git ninja-build \
        build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

WORKDIR /tmp/build
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/Dao-AILab/flash-attention.git && \
    git -C flash-attention fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention checkout ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention submodule update --init --depth 1

WORKDIR /tmp/build/flash-attention/hopper
RUN FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
        MAX_JOBS=${FA3_MAX_JOBS} python3.12 setup.py install

# Restructure the egg so the runtime stage can drop it into site-packages.
RUN SP=$(python3.12 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    mkdir -p /opt/flash_attn_3 && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_3/* /opt/flash_attn_3/ && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_interface.py /opt/flash_attn_3/ && \
    touch /opt/flash_attn_3/__init__.py


FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION_PATH=cu130

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Pin cu130 torch up front so the FA3 .so loads against the final torch ABI.
RUN pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

# Drop in flash_attn_3 prebuilt against CUDA 13.0 in the fa3-builder stage.
COPY --from=fa3-builder /opt/flash_attn_3 /tmp/flash_attn_3_built
RUN SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    rm -rf "${SP}/flash_attn_3" && \
    mv /tmp/flash_attn_3_built "${SP}/flash_attn_3"

# Install sglang then immediately re-pin cu130 torch (its deps drag cu12 wheels).
RUN pip install --no-cache-dir "sglang[all]==0.5.10.post1" && \
    pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

# Replace sgl-kernel with the cu130 wheel; install router.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
        "sgl-kernel @ https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp312-abi3-manylinux2014_aarch64.whl" && \
    pip install --no-cache-dir sglang-router==0.3.2

# Drop pip NCCL/cuBLAS so torch dlopens the alps-built libs from /usr.
RUN pip uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 \
                     nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas || true

# Verify curl (for router health checks) and print the loaded NCCL version.
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"nccl: {torch.cuda.nccl.version()}")
PY

WORKDIR /opt
