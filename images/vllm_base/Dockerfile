# Baseline vLLM image: upstream release on CUDA 13, no model-specific patches.
#
# cu130 matches the torch and vLLM wheels. The devel variant is required, not
# incidental: FlashInfer, tilelang, quack-kernels and humming-kernels all
# JIT-compile at runtime and need nvcc.
FROM docker.io/nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION_PATH=cu130
ARG UV_VERSION=0.12.3

ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG VLLM_VERSION=0.27.1
ARG TRANSFORMERS_VERSION=5.15.0
ARG RAY_VERSION=2.57.0
ARG FLASHINFER_VERSION=0.6.16.post3

# Build provenance from CI (.github/scripts/build_image.py); "unknown" in a published image means it was not built by the pipeline.
ARG BUILD_IMAGE=unknown
ARG BUILD_ARCH=unknown
ARG BUILD_CHANNEL=unknown
ARG BUILD_REVISION=unknown
ARG BUILD_DATE=unknown

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy

# Keep host NVIDIA driver libraries first, so the cuda-compat stubs in /usr/local/cuda/compat are not loaded ahead of them (CUDA Error 803).
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt

# DL3008 forces pinning versions explicitly in apt command. 
# We ignore this for apt installed packages because we want the latest images for security reasons.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash-completion ca-certificates curl git less vim-tiny \
    build-essential pkgconf \
    libibverbs-dev libnuma1 libnuma-dev \
    ffmpeg libsndfile1 \
    "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv" \
    && rm -rf /var/lib/apt/lists/*

RUN "python${PYTHON_VERSION}" -m venv /opt/venv && \
    /opt/venv/bin/pip install "uv==${UV_VERSION}" && \
    /opt/venv/bin/python --version && \
    /opt/venv/bin/uv --version
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"

# The torch trio is named explicitly so the cu130 index supplies it; through
# vLLM's pins alone it comes from PyPI, whose default variant is not guaranteed
# to be a CUDA 13 build. UV_INDEX_STRATEGY=unsafe-best-match lets the extra
# indexes win.
#
# flashinfer-cubin ships FlashInfer's kernels prebuilt. vLLM pins it in
# requirements/cuda.txt but leaves it out of the wheel metadata (it is not on
# PyPI), so it must be named here -- otherwise FlashInfer JIT-compiles
# attention kernels on first use in every job.
#
# The otel extra matches the opentela-share mount in the env tomls.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION_PATH}" \
    --extra-index-url "https://flashinfer.ai/whl/" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "vllm[audio,video,otel]==${VLLM_VERSION}" \
    "flashinfer-cubin==${FLASHINFER_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "ray[default]==${RAY_VERSION}"

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/venv/lib/python${PYTHON_VERSION}/site-packages/torch/lib

# Put the cuDNN/NCCL headers where the runtime JIT paths can find them, taken
# from the wheels rather than the base image so they describe the libraries
# actually loaded: LD_LIBRARY_PATH omits /usr/local/cuda/lib64, so CUDA comes
# entirely from the nvidia-* wheels. The /usr fallback matters only if that
# layout changes.
#
# The nccl.h guard exists because `ln -sf ""` would fail the build, not because
# the header is optional -- sanity_check.sh asserts both symlinks resolve.
RUN SITE="/opt/venv/lib/python${PYTHON_VERSION}/site-packages" && \
    CUDNN_H="$(find "${SITE}/nvidia/cudnn/include" -name cudnn.h -print -quit 2>/dev/null || true)" && \
    if [ -z "${CUDNN_H}" ]; then CUDNN_H="$(find /usr -name cudnn.h -print -quit)"; fi && \
    [ -n "${CUDNN_H}" ] || { echo "cudnn.h not found" >&2; exit 1; } && \
    CUDNN_DIR="$(dirname "${CUDNN_H}")" && \
    find "${CUDNN_DIR}" -maxdepth 1 -name 'cudnn*.h' -exec ln -sf {} /opt/venv/include/ \; && \
    NCCL_H="$(find "${SITE}" -path '*/nvidia/nccl/include/nccl.h' -print -quit)" && \
    if [ -n "${NCCL_H}" ]; then ln -sf "${NCCL_H}" /opt/venv/include/nccl.h; fi

# The contract sanity_check.sh asserts against the built image. CI runs it
# between `podman build` and `podman push` (.github/scripts/build_image.py), so
# a mismatch blocks the release instead of reaching a build log nobody reads.
#
# Distributions are checked through metadata; torch and vLLM are also imported
# -- torch catches a mismatched CUDA ABI, and vLLM initialises platform
# detection, which proves the CUDA runtime comes up on the build node's GPUs.
ENV SML_SANITY_PYTHON="${PYTHON_VERSION}" \
    SML_SANITY_COMMANDS="python uv curl ffmpeg vllm ray" \
    SML_SANITY_PACKAGES="torch==${TORCH_VERSION} \
torchvision==${TORCHVISION_VERSION} \
torchaudio==${TORCHAUDIO_VERSION} \
vllm==${VLLM_VERSION} \
flashinfer-cubin==${FLASHINFER_VERSION} \
transformers==${TRANSFORMERS_VERSION} \
ray==${RAY_VERSION}" \
    SML_SANITY_IMPORTS="torch vllm" \
    SML_SANITY_FILES="${VIRTUAL_ENV}/include/cudnn.h ${VIRTUAL_ENV}/include/nccl.h"

# Provenance in the environment because that is what reaches the cluster:
# `enroot import` yields a bare rootfs and keeps only the image environment (as
# /etc/environment). One variable, not five: every process inherits it.
ENV SML_IMAGE_BUILD="${BUILD_IMAGE} ${BUILD_CHANNEL} ${BUILD_ARCH} ${BUILD_REVISION} ${BUILD_DATE}"
