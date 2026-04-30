ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=aarch64
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=docker.io/library/ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

FROM ${DEVEL_BASE_IMAGE} AS build

# NOTE: libnuma1 is mounted from the host via the toml file.

WORKDIR /app/build

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
ADD "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh" /root/miniconda.sh
RUN chmod +x /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /opt/conda && \
    rm /root/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython git compilers make && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# Install vllm cu130 nightly (brings its own torch 2.10.0+cu130).
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly/cu130

# Ensure ray is installed and on PATH.
RUN pip install --no-cache-dir "ray[default]==2.55.0"

# Symlink cuDNN and NCCL headers into conda include path.
RUN CUDNN_DIR=$(dirname "$(find /usr -name 'cudnn.h' -print -quit)") && \
    ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/ && \
    ln -sf "$(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit)" /opt/conda/include/nccl.h

# Remove nvidia-cublas (pulled as transitive dep) — conflicts with CUDA 13 toolkit's cuBLAS.
RUN pip uninstall -y nvidia-cublas 2>/dev/null; true

# Install curl for router health checks.
RUN conda install -y curl && conda clean -ya

WORKDIR /opt
