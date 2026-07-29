ARG UBUNTU_VERSION=24.04
ARG TARGET_PLATFORM=aarch64
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

FROM ${DEVEL_BASE_IMAGE}

ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:${PATH} \
    SGLANG_BUILD_RUST_EXTS=none

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        bzip2 \
    && rm -rf /var/lib/apt/lists/*

ADD "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh" /root/miniforge.sh

RUN chmod +x /root/miniforge.sh && \
    bash /root/miniforge.sh -b -p /opt/conda && \
    rm /root/miniforge.sh && \
    conda install -y \
        python=${PYTHON_VERSION} && \
    python -m pip install --upgrade --no-cache-dir \
        pip "setuptools<70" && \
    conda clean -ya

# Must match the current fork's python/pyproject.toml.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG INSTALL_CHANNEL=whl
ARG CUDA_VERSION_PATH

ARG SGLANG_APERTUS_REPOSITORY=https://github.com/swiss-ai/sglang-apertus.git
ARG SGLANG_APERTUS_BRANCH=apertus-1-5
ARG SGLANG_APERTUS_COMMIT=925bfcf03b757fa2761a086610e64d4c536ee651
ARG TRANSFORMERS_REPOSITORY=https://github.com/swiss-ai/transformers.git
ARG TRANSFORMERS_BRANCH=feature/apertus_1p5_pipeline
ARG TRANSFORMERS_COMMIT=e66e45540f5b8d1ab4a2621061cf257f99e621a3

RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
        torch==${TORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION}

RUN CUDNN_DIR=$(dirname "$(find /usr -name cudnn.h -print -quit)") && \
    ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/ && \
    ln -sf "$(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit)" \
        /opt/conda/include/nccl.h

# Clone the named branches, then detach at their verified branch-head commits.
RUN git clone --depth 1 --branch "${SGLANG_APERTUS_BRANCH}" \
        "${SGLANG_APERTUS_REPOSITORY}" /workspace/sglang && \
    git -C /workspace/sglang checkout --detach "${SGLANG_APERTUS_COMMIT}" && \
    git clone --depth 1 --branch "${TRANSFORMERS_BRANCH}" \
        "${TRANSFORMERS_REPOSITORY}" /workspace/transformers && \
    git -C /workspace/transformers checkout --detach "${TRANSFORMERS_COMMIT}" 

# Install the fork from source. The binary-only rule matches setup.sh's kernel
# install and prevents silently compiling any dependency from an sdist.
RUN python -m pip install --pre --only-binary=:all: --no-cache-dir \
        --extra-index-url https://docs.sglang.ai/whl/cu130/ \
        -e /workspace/sglang/python

# The source project pins flashinfer-python. Add the matching precompiled
# cubin and JIT-cache artifacts instead of compiling them at runtime.
RUN python -m pip install --no-cache-dir --no-deps --only-binary=:all: \
        --index-url https://flashinfer.ai/whl \
        "flashinfer-cubin==0.6.15.post1" && \
    python -m pip install --no-cache-dir --no-deps --only-binary=:all: \
        --index-url https://flashinfer.ai/whl/cu130 \
        "flashinfer-jit-cache==0.6.15.post1"

RUN python -m pip install --no-cache-dir --no-deps /workspace/transformers
ENV FLASHINFER_WORKSPACE_BASE=/opt/flashinfer-cache
RUN mkdir -p "${FLASHINFER_WORKSPACE_BASE}" && conda clean -ya

WORKDIR /workspace
CMD ["bash"]
