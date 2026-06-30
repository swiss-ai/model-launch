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

# Install the cu130 PyTorch stack up front. sglang (below) reuses these
# already-satisfied versions, and the from-source sgl-kernel build links against
# this exact torch ABI.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION}

# Expose cuDNN/NCCL headers where the sgl-kernel CUDA build can find them.
RUN CUDNN_DIR=$(dirname "$(find /usr -name 'cudnn.h' -print -quit)") && \
    ln -sf "${CUDNN_DIR}"/cudnn*.h /opt/conda/include/ && \
    ln -sf "$(find /opt/conda/lib -path '*/nvidia/nccl/include/nccl.h' -print -quit)" /opt/conda/include/nccl.h

# Install sglang and all of its dependencies. As of torch 2.11 the default PyPI
# wheels are CUDA 13 and the cu130 torch stack above already satisfies sglang's
# torch==2.11.0 pin, so it is reused rather than downgraded. This also pulls the
# prebuilt sglang-kernel, which is replaced below to guarantee FA3 on every arch.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    "sglang[all]==0.5.14"

# Provide an sglang-kernel that includes FA3 (flash_ops), which GLM-5.2's DSA
# attention path hard-requires. Upstream builds FA3 only for x86_64; on aarch64 it
# is disabled by default (sgl-kernel CMakeLists: DEFAULT_SGL_KERNEL_ENABLE_FA3 OFF),
# so the published arm wheel has no flash_ops and `from sgl_kernel import flash_ops`
# fails at runtime. On aarch64 we therefore rebuild sgl-kernel from source with
# -DSGL_KERNEL_ENABLE_FA3=ON (gencode sm90a / Hopper); on x86_64 the prebuilt cu130
# wheel already ships flash_ops.
ARG SGL_KERNEL_VERSION=0.4.4
ARG SGLANG_REF=v0.5.14
ARG SGL_KERNEL_BUILD_JOBS=16
# hadolint ignore=SC2086
RUN set -eux; \
    arch="$(uname -m)"; \
    if [ "${arch}" = "x86_64" ]; then \
        pip install --no-cache-dir --no-deps --force-reinstall \
            "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_x86_64.whl"; \
    else \
        pip install --no-cache-dir "scikit-build-core==0.12.2"; \
        git clone --depth 1 --branch "${SGLANG_REF}" https://github.com/sgl-project/sglang.git /tmp/sglang; \
        CMAKE_BUILD_PARALLEL_LEVEL="${SGL_KERNEL_BUILD_JOBS}" \
        CMAKE_ARGS="-DSGL_KERNEL_ENABLE_FA3=ON -DGITHUB_ARTIFACTORY=github.com -DSGL_KERNEL_COMPILE_THREADS=${SGL_KERNEL_BUILD_JOBS}" \
            pip wheel --no-build-isolation --no-deps /tmp/sglang/sgl-kernel -w /tmp/sgl-wheels; \
        pip install --no-cache-dir --no-deps --force-reinstall /tmp/sgl-wheels/sglang_kernel-*.whl; \
        python -c "import glob, os, sgl_kernel; d=os.path.dirname(sgl_kernel.__file__); assert glob.glob(os.path.join(d, 'flash_ops*.so')), 'flash_ops .so missing from built wheel'; print('flash_ops compiled in')"; \
        rm -rf /tmp/sglang /tmp/sgl-wheels; \
    fi

# cu130 cuBLAS (torch 2.11 no longer bundles it), sglang-router, and curl for
# router health checks.
RUN pip install --no-cache-dir nvidia-cublas==13.6.0.2 && \
    pip install --no-cache-dir sglang-router==0.3.2 && \
    conda install -y curl && conda clean -ya

WORKDIR /opt
