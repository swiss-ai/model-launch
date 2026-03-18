ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=aarch64
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=docker.io/library/ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

#########################################################################
# Build image
#########################################################################

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

# Install PyTorch for CUDA 13.
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.9.1
ARG INSTALL_CHANNEL=whl
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchvision torchaudio

# Symlink cuDNN and NCCL headers into conda include path.
RUN CUDNN_DIR=$(dirname $(find /usr -name 'cudnn.h' -print -quit)) && \
    ln -sf ${CUDNN_DIR}/cudnn*.h /opt/conda/include/ && \
    ln -sf $(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit) /opt/conda/include/nccl.h

# Install flash-attn 3.
ARG FLASH_ATTN_3_SHA="92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
ARG FA3_MAX_JOBS=32
RUN git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
    git checkout ${FLASH_ATTN_3_SHA} && \
    git submodule update --init --depth 1 && \
    cd hopper && \
    FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE MAX_JOBS=${FA3_MAX_JOBS} python setup.py install && \
    cd /app/build && \
    rm -rf flash-attention

# Fix flash_attn_3 package structure for imports.
RUN cd /opt/conda/lib/python${PYTHON_VERSION}/site-packages/ && \
    mkdir -p flash_attn_3 && \
    mv flash_attn_3-*.egg/flash_attn_3/* flash_attn_3/ && \
    mv flash_attn_3-*.egg/flash_attn_interface.py flash_attn_3/ && \
    touch flash_attn_3/__init__.py && \
    rm -rf flash_attn_3-*

# Install sglang with all dependencies.
# NOTE: sglang pins torch to cuda12 wheels, so we use --no-deps first,
# then install its non-torch dependencies, then re-force cu130 torch.
RUN pip install --no-cache-dir "sglang[all]"

# Re-install PyTorch cu130 — sglang downgrades to cuda12 torch via its deps.
ARG CUDA_VERSION_PATH2=${CUDA_VERSION_PATH}
RUN pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH2}/ \
    torch==${TORCH_VERSION} torchvision torchaudio

# Re-install sgl-kernel with cu130 wheels (default pulls cuda12 version).
# Download wheel directly to avoid hash mismatch from the index.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    "sgl-kernel @ https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp312-abi3-manylinux2014_aarch64.whl"

# Install nvidia-cublas for torch compile compatibility.
RUN pip install --no-cache-dir nvidia-cublas

# Install curl for router health checks and sglang-router for load balancing.
RUN conda install -y curl && conda clean -ya
RUN pip install --no-cache-dir sglang-router

WORKDIR /opt
