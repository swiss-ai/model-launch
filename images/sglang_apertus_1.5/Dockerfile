ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
# Digest of the multi-arch manifest list for 13.0.0-cudnn-devel-ubuntu24.04
# (resolves to the right arch on both clusters). This freezes the whole OS
# layer, which is why the apt packages below don't need version pins.
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}@sha256:c2621d98e7de80c2aec5eb8403b19c67454c8f5b0c929e8588fd3563c9b6558d

FROM ${DEVEL_BASE_IMAGE}

ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:${PATH} \
    SGLANG_BUILD_RUST_EXTS=none

# NOTE: these pins go stale by design. The Ubuntu archive only serves the
# newest version of each package, so when a security update lands upstream the
# pinned version 404s and this step fails. That failure is the signal to bump:
# take the new version from `apt-cache policy <pkg>` in the base image (or
# /opt/apt-manifest.txt of the last good build) — it must be identical on both
# the amd64 (archive.ubuntu.com) and arm64 (ports.ubuntu.com) archives.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates=20260601~24.04.1 \
        git=1:2.43.0-1ubuntu7.3 \
        bzip2=1.0.8-5.1build0.1 \
        curl=8.5.0-2ubuntu10.11 \
    && rm -rf /var/lib/apt/lists/*

# Record the deb versions this build resolved, since they can't be pinned as
# inputs. The file ships inside the image and the cat lands in the CI log.
RUN dpkg-query -W -f '${Package}=${Version}\n' > /opt/apt-manifest.txt \
    && cat /opt/apt-manifest.txt

# Download the Miniforge installer matching the build arch so native arm64 and
# amd64 builds both work from the same Dockerfile. Pinned to a release tag —
# GitHub release assets are immutable, unlike the releases/latest redirect.
ARG MINIFORGE_VERSION=26.3.2-3
RUN set -eux; \
    ARCH="$(uname -m)"; \
    case "${ARCH}" in \
      x86_64|aarch64) ;; \
      *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac; \
    curl -fL "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-${ARCH}.sh" -o /root/miniforge.sh; \
    bash /root/miniforge.sh -b -p /opt/conda; \
    rm /root/miniforge.sh; \
    conda install -y \
        python=${PYTHON_VERSION} \
        curl=8.20.0 && \
    python -m pip install --upgrade --no-cache-dir \
        pip==26.1.2 \
        setuptools==69.5.1 && \
    conda clean -ya

# Must match the current fork's python/pyproject.toml.
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG INSTALL_CHANNEL=whl
ARG CUDA_VERSION_PATH

ARG SGLANG_APERTUS_REPOSITORY=https://github.com/swiss-ai/sglang-apertus.git
ARG SGLANG_APERTUS_BRANCH=tool-call-fix
ARG SGLANG_APERTUS_COMMIT=862531b265e1ee64e38e26b5099c29080d3421f9
ARG TRANSFORMERS_REPOSITORY=https://github.com/swiss-ai/transformers.git
ARG TRANSFORMERS_BRANCH=add-apertus1p5
ARG TRANSFORMERS_COMMIT=aa9e3c1a3a80ccbf12577885c9b30f494687d5e4

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
