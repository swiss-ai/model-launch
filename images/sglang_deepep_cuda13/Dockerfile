# Plain SGLang v0.5.14 image for CUDA 13 / aarch64 (Grace-Hopper GH200, SM90).
#
# NOTE ON THE NAME: this directory is still called "sglang_deepep_cuda13" for
# continuity (sglang_deepep.toml points at its .sqsh), but it NO LONGER builds
# DeepEP. DeepEP multi-node is unusable on Clariden (4-GPU GH200 nodes vs
# DeepEP's hard-coded 8-GPU-node assumption + Slingshot-vs-IBGDA transport),
# and GLM-5.2 is served with --moe-a2a-backend none, so DeepEP/NVSHMEM/libibverbs
# are all dropped. This bumps SGLang 0.5.10.post1 -> 0.5.14, which adds proper
# GLM-5.2 (GlmMoeDsa) + MTP/EAGLE support; 0.5.10 only half-supported GLM-5.2
# (draft loaded as a generic DeepseekV3 NextN head) and corrupted spec decoding.
#
# Recipe mirrors SGLang's own v0.5.14 docker/Dockerfile (cu130 branch): for
# CUDA 13 the cu130 wheels resolve directly (no cu12 force-reinstall dance), the
# kernel package is renamed sgl-kernel -> sglang-kernel (cu130 build on PyPI,
# abi3 so it runs on cp312), and FlashAttention-3 is JIT-compiled at runtime via
# the HF `kernels` lib (no source build) — so the CUDA *devel* base (nvcc) is
# kept for runtime JIT of FA3 / deep_gemm / flashinfer.
ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=13.0.0
ARG CUDA_VERSION_PATH=cu130
ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=docker.io/library/ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}


FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
# Miniforge for the build node's native arch (`uname -m` == aarch64 / x86_64,
# the exact tokens Miniforge uses), so the same Dockerfile builds on both CI
# clusters. curl is not in the base image yet, so install it first. conda also
# brings a host compiler toolchain (`compilers`) used by the runtime CUDA JIT
# (FA3 / deep_gemm / flashinfer) since aarch64 ships no prebuilt cubins.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -o /root/miniconda.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" && \
    chmod +x /root/miniconda.sh && \
    bash /root/miniconda.sh -b -p /opt/conda && \
    rm /root/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake git compilers make && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip setuptools wheel ninja packaging html5lib six && \
    /opt/conda/bin/conda clean -ya

# 1. PyTorch stack, cu130 (aarch64 + x86_64 wheels both live at this index).
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.11.0
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
    torch==${TORCH_VERSION} torchvision==0.26.0 torchaudio==2.11.0

# 2. sglang-kernel cu130 build from PyPI (renamed from sgl-kernel). The single
#    aarch64 wheel is cp310-abi3, so it runs on cp312; --no-deps keeps it from
#    dragging in a non-cu130 torch.
ARG SGLANG_KERNEL_VERSION=0.4.4
RUN pip install --no-cache-dir --force-reinstall --no-deps "sglang-kernel==${SGLANG_KERNEL_VERSION}"

# 3. Full SGLang 0.5.14. --extra-index-url cu130 keeps torch on the +cu130 build
#    (the +cu130 local version sorts above plain 2.11.0). This pulls the rest of
#    the stack at their pinned versions: flashinfer_python[cu13]==0.6.12 +
#    flashinfer_cubin==0.6.12, nvidia-cutlass-dsl[cu13]==4.5.2, sgl-deep-gemm,
#    kernels>=0.14.1, flash-attn-4, quack-kernels, transformers, etc. torch and
#    sglang-kernel are already satisfied, so they are not re-pulled. No FA3
#    source build and no `kernels download` (aarch64 has no prebuilt cubins —
#    FA3 JIT-compiles on first use against the devel toolchain).
ARG SGLANG_VERSION=0.5.14
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
    "sglang[all]==${SGLANG_VERSION}"

WORKDIR /opt
