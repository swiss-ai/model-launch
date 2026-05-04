# SGLang on the CSCS Alps extended NGC PyTorch base.
#
# Uses the 26.01-py3-alps4 base because SGLang 0.5.10.post1 strict-pins
# torch==2.9.1; NGC PyTorch 26.01 ships torch 2.9.x. We still re-install
# torch 2.9.1 cu130 from pytorch.org because sglang's transitive deps drag
# in cu12 wheels — the dance preserves cu130 alignment.
# See: https://github.com/eth-cscs/alps-extended-images
#
# CSCS internal mirror: jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps4

# Flash-Attention 3 builder stage.
#
# The 26.01-py3-alps4 base ships the CUDA 13.1 toolkit, and nvcc 13.1's
# cudafe++ emits host-side stubs (__cudaLaunchPrologue, __cudaSetupArg,
# __cudaLaunch, __cudaRegisterEntry, ...) that the matching cuda_runtime.h
# no longer declares on aarch64. FA3's heavily-templated hopper kernels
# trigger this — the build dies with thousands of "was not declared in
# this scope" errors per .cu file. CUDA 13.0 is unaffected (proven by
# images/sglang_cuda13), so build FA3 in a side stage on a CUDA 13.0
# toolkit and copy the compiled package into the runtime stage. ABI is
# safe: both stages use cp312 + torch 2.9.1+cu130, and a binary built
# against CUDA 13.0 runs fine on the 13.1 runtime.
ARG FA3_BUILDER_BASE=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04
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

# Restructure the egg into a clean flash_attn_3/ package under /opt so the
# runtime stage can drop it straight into site-packages with one COPY.
RUN SP=$(python3.12 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    mkdir -p /opt/flash_attn_3 && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_3/* /opt/flash_attn_3/ && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_interface.py /opt/flash_attn_3/ && \
    touch /opt/flash_attn_3/__init__.py


# Runtime stage.
ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.01-py3-alps4
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION_PATH=cu130

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Pin torch to the cu130 wheel up front so the flash-attn 3 .so we drop in
# below loads against the final torch ABI; sglang install will not displace
# this torch (its deps drag cu12 wheels — re-pinned again after that step).
RUN pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

# Drop in flash_attn_3 prebuilt against CUDA 13.0 in the fa3-builder stage.
COPY --from=fa3-builder /opt/flash_attn_3 /tmp/flash_attn_3_built
RUN SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    rm -rf "${SP}/flash_attn_3" && \
    mv /tmp/flash_attn_3_built "${SP}/flash_attn_3"

# Install sglang then immediately restore cu130 torch (its deps drag in
# cu12 wheels). Single RUN keeps the cu13 ABI consistent in one layer.
RUN pip install --no-cache-dir "sglang[all]==0.5.10.post1" && \
    pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

# Replace sgl-kernel with the cu130 wheel; install sglang-router.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
        "sgl-kernel @ https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp312-abi3-manylinux2014_aarch64.whl" && \
    pip install --no-cache-dir sglang-router==0.3.2

# Drop pip NCCL/cuBLAS wheels: alps-built libnccl on /usr and the CUDA 13
# toolkit's cuBLAS in the NGC base are the ones we want torch to dlopen.
RUN pip uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 \
                     nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas || true

# Verify the NGC base provides curl (used by router health checks at runtime)
# and print the loaded NCCL version (expect 2.28.7 from the alps build).
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"nccl: {torch.cuda.nccl.version()}")
PY

WORKDIR /opt
