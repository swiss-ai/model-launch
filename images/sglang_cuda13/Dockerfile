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

ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.9.1
ARG INSTALL_CHANNEL=whl
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

RUN CUDNN_DIR=$(dirname "$(find /usr -name 'cudnn.h' -print -quit)") && \
    ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/ && \
    ln -sf "$(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit)" /opt/conda/include/nccl.h

ARG FLASH_ATTN_3_SHA="92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
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

# Install sglang with all dependencies.
# NOTE: sglang pins torch to cuda12 wheels, so we use --no-deps first,
# then install its non-torch dependencies, then re-force cu130 torch.
RUN pip install --no-cache-dir "sglang[all]==0.5.10.post1"

# Re-install PyTorch cu130 — sglang downgrades to cuda12 torch via its deps.
ARG CUDA_VERSION_PATH2=${CUDA_VERSION_PATH}
RUN pip install --no-cache-dir --force-reinstall --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH2}/ \
    torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

# Re-install sgl-kernel with cu130 wheels, install nvidia-cublas, curl, and sglang-router.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
    "sgl-kernel @ https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp312-abi3-manylinux2014_aarch64.whl" && \
    pip install --no-cache-dir nvidia-cublas==13.4.0.1 && \
    conda install -y curl && conda clean -ya && \
    pip install --no-cache-dir sglang-router==0.3.2

WORKDIR /opt
