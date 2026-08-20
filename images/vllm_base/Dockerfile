# Baseline vLLM image: upstream release on CUDA 13, no model-specific patches.
#
# The base tracks CUDA 13.0.x because the torch and vLLM wheels are cu130
# builds. The devel variant is required, not incidental: FlashInfer, tilelang,
# quack-kernels and humming-kernels all JIT-compile at runtime and need nvcc.
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

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy

# Keep host NVIDIA driver libraries first, so the cuda-compat stubs in
# /usr/local/cuda/compat are not loaded ahead of them (CUDA Error 803).
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt

# libibverbs/libnuma back the RDMA and NUMA paths on Grace nodes; ffmpeg and
# libsndfile1 decode audio/video inputs; curl serves the router health checks.
# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    bash-completion ca-certificates curl git less vim-tiny \
    build-essential pkgconf \
    libibverbs-dev libnuma1 libnuma-dev \
    ffmpeg libsndfile1 \
    python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install "uv==${UV_VERSION}" && \
    /opt/venv/bin/python --version && \
    /opt/venv/bin/uv --version
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"

# The torch trio is named explicitly so the cu130 index supplies it; resolved
# through vLLM's own pins alone it would come from PyPI, whose default variant
# is not guaranteed to be a CUDA 13 build. UV_INDEX_STRATEGY=unsafe-best-match
# is what lets the extra indexes win.
#
# flashinfer-cubin ships FlashInfer's kernels as prebuilt cubins. vLLM pins it
# in requirements/cuda.txt but excludes it from the published wheel metadata,
# since it is not on PyPI, so it has to be requested by name -- otherwise
# FlashInfer JIT-compiles attention kernels on first use in every job.
#
# The otel extra matches the opentela-share mount the env tomls provide.
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

# Put the cuDNN/NCCL headers where the runtime JIT paths can find them, then
# check the stack. torch is imported, which catches a mismatched CUDA wheel
# ABI; vLLM is read through metadata only, because importing it initialises
# platform detection and the build host has no GPU.
#
# Both header sets are taken from the wheels rather than from the base image,
# so they describe the libraries that actually get loaded: LD_LIBRARY_PATH does
# not include /usr/local/cuda/lib64, so CUDA comes entirely from the nvidia-*
# wheels at runtime. The /usr fallback only matters if that layout changes.
RUN SITE="/opt/venv/lib/python${PYTHON_VERSION}/site-packages" && \
    CUDNN_H="$(find "${SITE}/nvidia/cudnn/include" -name cudnn.h -print -quit 2>/dev/null || true)" && \
    if [ -z "${CUDNN_H}" ]; then CUDNN_H="$(find /usr -name cudnn.h -print -quit)"; fi && \
    [ -n "${CUDNN_H}" ] || { echo "cudnn.h not found" >&2; exit 1; } && \
    CUDNN_DIR="$(dirname "${CUDNN_H}")" && \
    find "${CUDNN_DIR}" -maxdepth 1 -name 'cudnn*.h' -exec ln -sf {} /opt/venv/include/ \; && \
    NCCL_H="$(find "${SITE}" -path '*/nvidia/nccl/include/nccl.h' -print -quit)" && \
    if [ -n "${NCCL_H}" ]; then ln -sf "${NCCL_H}" /opt/venv/include/nccl.h; fi && \
    python -c "import torch; from importlib.metadata import version; \
print('torch', torch.__version__, '| vllm', version('vllm'), '| transformers', version('transformers'))"
