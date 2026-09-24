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
# hadolint ignore=DL3008
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
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml ipython git compilers make; \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja; \
    /opt/conda/bin/conda clean -ya

# Install vllm cu130 nightly (brings its own torch 2.10.0+cu130).
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly/cu130

# Ensure ray is installed and on PATH.
RUN pip install --no-cache-dir "ray[default]==2.55.0"

# Symlink cuDNN and NCCL headers into conda include path (tolerant: the x86_64 CUDA image
# keeps them elsewhere and the JIT paths find them through the wheels' own copies).
RUN set -eux; \
    CUDNN_HEADER="$(find /usr /opt/conda -name 'cudnn.h' -print -quit)"; \
    if [ -n "${CUDNN_HEADER}" ]; then \
        ln -sf "$(dirname "${CUDNN_HEADER}")"/cudnn*.h /opt/conda/include/; \
    else \
        echo "cudnn.h not found; skipping cuDNN header symlink"; \
    fi; \
    NCCL_HEADER="$(find /usr /opt/conda -path '*/include/nccl.h' -print -quit)"; \
    if [ -n "${NCCL_HEADER}" ]; then \
        ln -sf "${NCCL_HEADER}" /opt/conda/include/nccl.h; \
    else \
        echo "nccl.h not found; skipping NCCL header symlink"; \
    fi

# Remove nvidia-cublas (pulled as transitive dep) — conflicts with CUDA 13 toolkit's cuBLAS.
RUN pip uninstall -y nvidia-cublas 2>/dev/null; true

# Install curl for router health checks.
RUN conda install -y curl && conda clean -ya

WORKDIR /opt
