# SGLang on the CSCS Alps extended NGC PyTorch base.
#
# Uses the 26.01-py3-alps4 base because SGLang 0.5.10.post1 strict-pins
# torch==2.9.1; NGC PyTorch 26.01 ships torch 2.9.x. We still re-install
# torch 2.9.1 cu130 from pytorch.org because sglang's transitive deps drag
# in cu12 wheels — the dance preserves cu130 alignment.
# See: https://github.com/eth-cscs/alps-extended-images
#
# CSCS internal mirror: jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps4
ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.01-py3-alps4
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION_PATH=cu130

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Pin torch to the cu130 wheel up front so flash-attn 3 builds against the
# final torch ABI; sglang install below will not displace this.
RUN pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

# Build flash-attention 3 (Hopper kernels only) and repackage into the
# site-packages location sglang expects.
ARG FLASH_ATTN_3_SHA="92ca9da8d66f7b34ff50dc080ec0fef9661260d6"
ARG FA3_MAX_JOBS=32
WORKDIR /tmp/build
RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/Dao-AILab/flash-attention.git && \
    git -C flash-attention fetch --depth 1 origin ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention checkout ${FLASH_ATTN_3_SHA} && \
    git -C flash-attention submodule update --init --depth 1

WORKDIR /tmp/build/flash-attention/hopper
RUN FLASH_ATTENTION_DISABLE_FP16=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
        MAX_JOBS=${FA3_MAX_JOBS} python setup.py install

WORKDIR /tmp/build
RUN SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') && \
    mkdir -p "${SP}/flash_attn_3" && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_3/* "${SP}/flash_attn_3/" && \
    mv "${SP}"/flash_attn_3-*.egg/flash_attn_interface.py "${SP}/flash_attn_3/" && \
    touch "${SP}/flash_attn_3/__init__.py" && \
    rm -rf "${SP}"/flash_attn_3-* && \
    rm -rf /tmp/build/flash-attention

# Install sglang then immediately restore cu130 torch (its deps drag in
# cu12 wheels). Single RUN keeps the cu13 ABI consistent in one layer.
RUN pip install --no-cache-dir "sglang[all]==0.5.10.post1" && \
    pip install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1

# Replace sgl-kernel with the cu130 wheel; install sglang-router.
RUN pip install --no-cache-dir --no-deps --force-reinstall \
        "sgl-kernel @ https://github.com/sgl-project/whl/releases/download/v0.3.21/sgl_kernel-0.3.21+cu130-cp312-abi3-manylinux2014_aarch64.whl" && \
    pip install --no-cache-dir sglang-router==0.3.2

# Drop pip NCCL/cuBLAS wheels: alps-built libnccl on /usr and the CUDA 13
# toolkit's cuBLAS in the NGC base are the ones we want torch to dlopen.
RUN pip uninstall -y nvidia-nccl-cu13 nvidia-nccl-cu12 \
                     nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas || true

# Verify the NGC base provides curl (used by router health checks at runtime)
# and print the loaded NCCL version (expect 2.28.7 from the alps build).
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"nccl: {torch.cuda.nccl.version()}")
PY

WORKDIR /opt
