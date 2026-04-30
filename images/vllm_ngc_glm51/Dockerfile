ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.02-py3-alps4

FROM ${BASE_IMAGE}

ARG VLLM_VERSION=0.19.1
ARG DEEPGEMM_REF=main

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# uv for fast package installation.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# hadolint ignore=DL3013
RUN uv pip install --system "vllm==${VLLM_VERSION}" "ray[default]"

# DeepGEMM: FP8 grouped-GEMM kernels needed for GLM-5.1-FP8's block-wise
# quant scheme (block_size=[128,128]). Kernels are JIT-compiled at first use;
# no GPU is required at image-build time.
RUN git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git /tmp/deepgemm && \
    uv pip install --system --no-build-isolation /tmp/deepgemm && \
    rm -rf /tmp/deepgemm
