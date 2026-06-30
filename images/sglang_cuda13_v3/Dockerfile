ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}


FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install miniconda, Python, and Python build dependencies. The build runs
# `podman build` natively per arch and passes no build args, so the Miniforge
# arch is detected from `uname -m` (aarch64 / x86_64) rather than hardcoded.
ARG PYTHON_VERSION
ARG MINIFORGE_VERSION=26.3.2-3
ENV PATH=/opt/conda/bin:$PATH
# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-Linux-$(uname -m).sh" -o /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /opt/conda && \
    rm /root/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake pyyaml numpy git compilers make && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# PyTorch (cu130) version pins — single source of truth for the whole stack.
ARG CUDA_VERSION_PATH
ARG INSTALL_CHANNEL=whl
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0

# Install the cu130 PyTorch stack up front: flash-attention is built against this
# exact ABI, and sglang (below) reuses these already-satisfied versions rather than
# resolving its own — which is why no force-reinstall step is needed afterwards.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}

RUN CUDNN_DIR=$(dirname "$(find /usr -name 'cudnn.h' -print -quit)") && \
    ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/ && \
    ln -sf "$(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit)" /opt/conda/include/nccl.h

ARG FLASH_ATTN_3_SHA="e79261e4fe23135c55923782707f297a957fa124"
RUN git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git && \
    git -C flash-attention fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention checkout ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention submodule update --init --depth 1

WORKDIR /app/build/flash-attention/hopper
ARG FA3_MAX_JOBS=32
RUN FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=${FA3_MAX_JOBS} python setup.py install

WORKDIR /app/build
RUN rm -rf flash-attention

WORKDIR /opt/conda/lib/python${PYTHON_VERSION}/site-packages
RUN mkdir -p flash_attn_3 && \
    mv flash_attn_3-*.egg/flash_attn_3/* flash_attn_3/ && \
    mv flash_attn_3-*.egg/flash_attn_interface.py flash_attn_3/ && \
    touch flash_attn_3/__init__.py && \
    rm -rf flash_attn_3-*

WORKDIR /app/build

# Install sglang in a single layer, no reinstall dance. As of torch 2.11 the default
# PyPI wheels are CUDA 13, and the cu130 torch stack above already satisfies sglang's
# torch==2.11.0 pin, so sglang reuses it instead of pulling a conflicting build. Only
# sglang-kernel still needs an explicit cu130 wheel (PyPI ships a single untagged
# build), pinned here by URL for determinism. nvidia-cublas (cu13, which torch no
# longer bundles) and sglang-router complete the runtime; curl is for health checks.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
        "sglang[all]==0.5.14" \
        "sglang-kernel @ https://github.com/sgl-project/whl/releases/download/v0.4.4/sglang_kernel-0.4.4+cu130-cp310-abi3-manylinux2014_$(uname -m).whl" && \
    pip install --no-cache-dir nvidia-cublas==13.6.0.2 && \
    pip install --no-cache-dir sglang-router==0.3.2 && \
    conda install -y curl && conda clean -ya

WORKDIR /opt
