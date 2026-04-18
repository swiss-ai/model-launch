ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=aarch64
ARG CUDA_VERSION=13.0.0
ARG PYTHON_VERSION=3.12
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}


FROM ${DEVEL_BASE_IMAGE} AS build

# NOTE: libnuma1 is mounted from the host via the toml file.

WORKDIR /app/build

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh" -o /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /opt/conda && \
    rm /root/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython git compilers make curl && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging setuptools ninja && \
    /opt/conda/bin/conda clean -ya

# Install vllm cu130 stable (brings its own torch 2.10.0+cu130).
# nvidia-cublas is uninstalled in the same layer so it never lands in the image.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system "vllm==0.19.1" --torch-backend=auto --extra-index-url https://wheels.vllm.ai/0.19.1/cu130/ && \
    pip uninstall -y nvidia-cublas 2>/dev/null; true

# Ensure ray is installed and on PATH.
RUN pip install --no-cache-dir "ray[default]==2.55.0"

# Symlink cuDNN and NCCL headers into conda include path.
RUN CUDNN_DIR=$(dirname "$(find /usr -name 'cudnn.h' -print -quit)") && \
    ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/ && \
    ln -sf "$(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit)" /opt/conda/include/nccl.h

WORKDIR /opt
