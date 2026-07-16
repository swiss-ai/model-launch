ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=docker.io/library/ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}


FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install miniconda, Python, and Python build dependencies.
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH

# hadolint ignore=DL3008
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    ARCH="$(uname -m)"; \
    case "${ARCH}" in \
      x86_64) MINIFORGE_ARCH="x86_64" ;; \
      aarch64) MINIFORGE_ARCH="aarch64" ;; \
      *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac; \
    curl -L "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" -o /root/miniconda.sh; \
    chmod +x /root/miniconda.sh; \
    bash /root/miniconda.sh -b -p /opt/conda; \
    rm /root/miniconda.sh; \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml ipython git compilers make; \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja; \
    /opt/conda/bin/conda clean -ya

# Install vLLM CUDA13 stack compatible with vLLM-Omni 0.20.0.
# Use --python explicitly so uv targets the conda Python, not /usr Python.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --python /opt/conda/bin/python --no-cache torch==2.11.0 --torch-backend=auto && \
    uv pip install --python /opt/conda/bin/python --no-cache vllm==0.20.2 --torch-backend=auto && \
    rm -rf /root/.cache/uv

# Ensure ray is installed and on PATH.
RUN /opt/conda/bin/python -m pip install --no-cache-dir "ray[default]==2.55.0"

# Symlink cuDNN and NCCL headers into conda include path when present.
RUN set -eux; \
    CUDNN_HEADER="$(find /usr /opt/conda -name 'cudnn.h' -print -quit)"; \
    if [ -n "${CUDNN_HEADER}" ]; then \
        CUDNN_DIR="$(dirname "${CUDNN_HEADER}")"; \
        ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/; \
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
RUN /opt/conda/bin/python -m pip uninstall -y nvidia-cublas 2>/dev/null; true

# Install curl for router health checks.
RUN conda install -y curl && conda clean -ya

# Install audio system dependencies for Qwen3-TTS & vLLM-Omni.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies for Qwen3-TTS & vLLM-Omni.
# vLLM-Omni is installed without dependencies to avoid replacing torch/vLLM/CUDA wheels.
RUN uv pip install --python /opt/conda/bin/python --no-cache --no-deps \
    vllm-omni==0.20.0 && \
    uv pip install --python /opt/conda/bin/python --no-cache \
    av==16.0.1 \
    omegaconf==2.3.0 \
    diffusers==0.36.0 \
    accelerate==1.12.0 \
    soundfile==0.13.1 \
    cache-dit==1.3.0 \
    tqdm==4.67.1 \
    torchsde==0.2.6 \
    openai-whisper==20250625 \
    "imageio[ffmpeg]==2.37.2" \
    x-transformers==2.12.2 \
    einops==0.8.1 \
    prettytable==3.17.0 \
    aenum==3.1.16 \
    pyzmq==27.1.0 \
    janus==2.0.0 \
    pydub==0.25.1 \
    onnxruntime==1.23.2 \
    fa3-fwd==0.0.3 \
    transformers==5.8.0 \
    gradio==6.14.0 \
    gradio-client==2.5.0 \
    librosa==0.11.0 \
    requests==2.34.2 \
    numpy==2.3.5 && \
    rm -rf /root/.cache/uv

WORKDIR /opt