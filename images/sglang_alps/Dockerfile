# SGLang on the CSCS Alps extended NGC PyTorch base.
# Base courtesy of CSCS — https://github.com/eth-cscs/alps-extended-images
ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.01-py3-alps4
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=500 UV_INDEX_STRATEGY=unsafe-best-match UV_LINK_MODE=copy UV_BREAK_SYSTEM_PACKAGES=1

# Install torch+cu130 first (sets up the pip-nvidia layout), then sglang and
# friends. The cu130 sglang-kernel wheel is GitHub-only (PyPI ships a generic
# manylinux variant); override after the main resolve via --no-deps. flash-attn-4
# is a pure-Python wheel that JITs hopper/blackwell kernels via cutlass-dsl at
# runtime — no from-source FA3 build needed. nvidia-cuda-nvrtc-cu12 satisfies
# sglang-kernel's NEEDED entry on libnvrtc.so.12. Pip nvidia-nccl/cublas are
# uninstalled so torch dlopens the alps host-mounted libs instead.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --torch-backend=cu130 && \
    uv pip install --system --no-deps --force-reinstall \
        "sglang-kernel @ https://github.com/sgl-project/whl/releases/download/v0.4.1/sglang_kernel-0.4.1+cu130-cp310-abi3-manylinux2014_aarch64.whl" && \
    uv pip install --system --prerelease=allow \
        "sglang[all]==0.5.10.post1" \
        sglang-router==0.3.2 \
        "flash-attn-4>=4.0.0b4" \
        "nvidia-cutlass-dsl>=4.4.1" \
        nvidia-cuda-nvrtc-cu12 && \
    uv pip uninstall --system nvidia-nccl-cu13 nvidia-nccl-cu12 \
                              nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas

# Register pip-installed nvidia/* lib dirs in the dynamic linker cache so dlopen
# finds them at runtime — alps hooks rewrite LD_LIBRARY_PATH at job launch, so
# we need ld.so.cache (LD_LIBRARY_PATH-independent) to win.
RUN find /usr/local/lib/python3.12/dist-packages/nvidia -mindepth 2 -maxdepth 2 \
        -type d -name lib > /etc/ld.so.conf.d/zz-nvidia-pip.conf && \
    ldconfig && \
    ldconfig -p | grep -q libnvrtc.so.12 || { echo "libnvrtc.so.12 missing from ld cache"; exit 1; }

# Confirm torch is a CUDA build and the sm90 kernel was shipped.
RUN python -c "import torch, glob, sysconfig, importlib.metadata as m; \
    assert torch.version.cuda, f'no CUDA: {torch.__version__}'; \
    sp = sysconfig.get_paths()['purelib']; \
    assert glob.glob(f'{sp}/sgl_kernel/sm90/common_ops*.so'), 'no sm90 ops .so'; \
    print(f'torch {torch.__version__} sglang-kernel {m.version(\"sglang-kernel\")}')"
