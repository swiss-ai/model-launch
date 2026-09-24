ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=docker.io/library/ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}


FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install miniconda, Python, and Python build dependencies (arch-aware: the same
# Dockerfile builds on the GH200 (aarch64) and x86_64 CI runners).
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "$(uname -m)" in \
      x86_64) MINIFORGE_ARCH="x86_64" ;; \
      aarch64) MINIFORGE_ARCH="aarch64" ;; \
      *) echo "Unsupported architecture: $(uname -m)" && exit 1 ;; \
    esac; \
    curl -L "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" -o /root/miniconda.sh; \
    chmod +x /root/miniconda.sh; \
    bash /root/miniconda.sh -b -p /opt/conda; \
    rm /root/miniconda.sh; \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython git compilers make; \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja; \
    /opt/conda/bin/conda clean -ya

# Versions. SGLang pins its torch and kernel package exactly (python/pyproject.toml of the
# release); the factory (model-launch-factory) renders these from the release's metadata.
ARG SGLANG_VERSION=0.5.20
ARG TORCH_VERSION=2.13.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG SGLANG_KERNEL_VERSION=0.4.7
ARG CUDA_VERSION_PATH
ARG INSTALL_CHANNEL=whl

# PyTorch cu130 first, so sglang's resolver keeps it (the PyPI torch wheel is cu12).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchvision torchaudio==${TORCHAUDIO_VERSION}

# Symlink cuDNN and NCCL headers into conda include path (flashinfer / FA4 JIT).
RUN CUDNN_DIR=$(dirname "$(find /usr -name 'cudnn.h' -print -quit)") && \
    ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/ && \
    ln -sf "$(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit)" /opt/conda/include/nccl.h

# SGLang with all dependencies. No kernel is compiled from source here: FA3 comes with
# sglang-kernel where its wheel carries it (x86_64), FA4 is flash-attn-4 (CuTe DSL, JIT at
# first use — the attention backend to use on GH200: --attention-backend fa4), flashinfer
# JIT-compiles into its cache. The hand-built FlashAttention 3 of the 0.5.10 image is gone:
# 0.5.20 no longer imports it, and its nvcc compile broke with the current toolchain.
RUN pip install --no-cache-dir "sglang[all]==${SGLANG_VERSION}"

# Re-install PyTorch cu130 in case a dependency pulled the cu12 wheel back in.
RUN pip install --no-cache-dir --force-reinstall --no-deps --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchvision torchaudio==${TORCHAUDIO_VERSION}

# sglang-kernel: the cu130 wheel from sgl-project/whl (PyPI's is cu12), per arch.
RUN set -eux; \
    case "$(uname -m)" in \
      x86_64) WHEEL_ARCH="x86_64" ;; \
      aarch64) WHEEL_ARCH="aarch64" ;; \
    esac; \
    pip install --no-cache-dir --no-deps --force-reinstall \
      "sglang-kernel @ https://github.com/sgl-project/whl/releases/download/v${SGLANG_KERNEL_VERSION}/sglang_kernel-${SGLANG_KERNEL_VERSION}+${CUDA_VERSION_PATH}-cp310-abi3-manylinux2014_${WHEEL_ARCH}.whl"

# nvidia-cublas for CUDA 13, curl for router health checks, sglang-router.
RUN pip install --no-cache-dir nvidia-cublas==13.4.0.1 && \
    conda install -y curl && conda clean -ya && \
    pip install --no-cache-dir sglang-router==0.3.2

WORKDIR /opt
