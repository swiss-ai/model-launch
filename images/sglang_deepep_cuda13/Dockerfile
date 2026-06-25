ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=aarch64
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=docker.io/library/ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}


FROM ${DEVEL_BASE_IMAGE} AS build

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

# ---------------------------------------------------------------------------
# DeepEP (DeepSeek expert-parallel all-to-all) + NVSHMEM.
#
# Enables `--moe-a2a-backend deepep` for MoE models like GLM-5.2. DeepEP has no
# PyPI wheel, so it is built from source against the nvidia-nvshmem-cu13 wheel
# (which ships an aarch64 build and is auto-detected by DeepEP's setup.py).
# Pinned commit + the CUDA-13 CCCL include patch follow SGLang's own
# docker/Dockerfile so the build matches the sglang 0.5.10.post1 / cu130 stack.
#
# NOTE: DeepEP's *inter-node* path relies on NVSHMEM IBGDA over InfiniBand/RoCE
# and does NOT work over Slingshot/CXI. On Clariden expect intranode (NVLink)
# expert parallelism to work; multi-node deepep may fail at NVSHMEM init.
# ---------------------------------------------------------------------------
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/opt/conda/lib/python${PYTHON_VERSION}/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH}

ARG DEEPEP_COMMIT=9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee
ARG DEEPEP_MAX_JOBS=32

# NVSHMEM (>=3.3.9 required by DeepEP). cu13 wheel ships an aarch64 build and is
# auto-discovered by DeepEP via importlib find_spec("nvidia.nvshmem").
RUN pip install --no-cache-dir "nvidia-nvshmem-cu13>=3.3.9"

# InfiniBand verbs headers. DeepEP's csrc/kernels/configs.cuh unconditionally
# #includes <infiniband/mlx5dv.h> (even for the intranode kernels), so the
# Mellanox direct-verbs header must be present at compile time. libibverbs-dev
# provides it (and pulls in the runtime libibverbs/libmlx5 it links against).
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

# Build + install DeepEP from source for Hopper/SM90 (GH200).
#
# GDRCopy / IBGDA are intentionally omitted: DeepEP's inter-node path needs
# NVSHMEM IBGDA over InfiniBand/RoCE, which is unsupported on Clariden's
# Slingshot/CXI fabric, so only intranode (NVLink) EP is usable here and that
# path doesn't need them. The GitHub tarball is used (not the zip) so no extra
# apt packages (unzip) are required — tar is in the base image.
# hadolint ignore=DL3003
RUN cd /tmp && curl -fsSL -o deepep.tar.gz \
        https://github.com/deepseek-ai/DeepEP/archive/${DEEPEP_COMMIT}.tar.gz && \
    tar -xzf deepep.tar.gz && mv DeepEP-${DEEPEP_COMMIT} DeepEP && cd DeepEP && \
    sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh && \
    sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh && \
    sed -i "/^    include_dirs = \['csrc\/'\]/a\    include_dirs.append('${CUDA_HOME}/include/cccl')" setup.py && \
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=${DEEPEP_MAX_JOBS} \
        python setup.py bdist_wheel -d /tmp/deepep-wheels && \
    pip install --no-cache-dir /tmp/deepep-wheels/*.whl && \
    cd / && rm -rf /tmp/DeepEP /tmp/deepep.tar.gz /tmp/deepep-wheels

WORKDIR /opt
