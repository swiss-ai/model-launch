# SGLang on the CSCS Alps extended NGC PyTorch base.
# Base courtesy of CSCS — https://github.com/eth-cscs/alps-extended-images
ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.01-py3-alps4
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=500 UV_INDEX_STRATEGY=unsafe-best-match UV_LINK_MODE=copy UV_BREAK_SYSTEM_PACKAGES=1

# Stub dist-info markers for NVIDIA libs already provided by the alps base
# image as apt packages (libnccl2, libcublas-13-1) — the .so files live in
# /usr/lib/aarch64-linux-gnu/ and /usr/local/cuda/lib64/. pip's resolver
# treats these as satisfied and skips re-shipping the pip wheels (~600 MB
# saved). Versions match what torch 2.9.1+cu130 strictly pins. When alps
# hooks are active at runtime, torch's global_deps loads cleanly and the
# nvidia/{nccl,cublas}/lib preload path is never invoked.
RUN python - <<'PY'
import os, sysconfig
sp = sysconfig.get_paths()["purelib"]
for name, ver in [("nvidia-nccl-cu13", "2.27.7"), ("nvidia-cublas", "13.0.0.19")]:
    d = f"{sp}/{name.replace('-', '_')}-{ver}.dist-info"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/METADATA", "w").write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
    open(f"{d}/WHEEL", "w").write("Wheel-Version: 1.0\nGenerator: alps-stub\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    open(f"{d}/INSTALLER", "w").write("alps-stub\n")
    open(f"{d}/RECORD", "w").close()
PY

# Install torch + sglang + kernel. The cu130 sglang-kernel wheel is GitHub-only
# (PyPI ships a generic manylinux variant); override the resolved generic wheel
# via --no-deps. flash-attn-4 is a pure-Python wheel that JITs hopper/blackwell
# kernels via cutlass-dsl at runtime — no from-source FA3 build needed.
# nvidia-cuda-nvrtc-cu12 satisfies sglang-kernel's NEEDED entry on libnvrtc.so.12
# (the alps base ships only libnvrtc.so.13).
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
        nvidia-cuda-nvrtc-cu12

# Register the pip-installed nvidia/* lib dirs (now just nvidia-cuda-nvrtc-cu12
# and a few jpeg/tiff helpers) in the dynamic linker cache, so dlopen finds
# them at runtime — alps hooks rewrite LD_LIBRARY_PATH at job launch.
RUN find /usr/local/lib/python3.12/dist-packages/nvidia -mindepth 2 -maxdepth 2 \
        -type d -name lib > /etc/ld.so.conf.d/zz-nvidia-pip.conf && \
    ldconfig && \
    ldconfig -p | grep -q libnvrtc.so.12 || { echo "libnvrtc.so.12 missing from ld cache"; exit 1; }

# Confirm torch is a CUDA build, sm90 ops shipped, the stubs worked (no pip
# nvidia/{nccl,cublas} dirs), and libnccl resolves via ld cache to system apt.
RUN python -c "import torch, glob, os, sysconfig, importlib.metadata as m, subprocess; \
    assert torch.version.cuda, f'no CUDA: {torch.__version__}'; \
    sp = sysconfig.get_paths()['purelib']; \
    assert glob.glob(f'{sp}/sgl_kernel/sm90/common_ops*.so'), 'no sm90 ops .so'; \
    assert not os.path.isdir(f'{sp}/nvidia/nccl'), 'nvidia/nccl pip dir exists — stub did not work'; \
    assert not os.path.isdir(f'{sp}/nvidia/cublas'), 'nvidia/cublas pip dir exists — stub did not work'; \
    ld = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True).stdout; \
    nccl = [l for l in ld.splitlines() if 'libnccl.so.2 ' in l and 'aarch64-linux-gnu' in l]; \
    assert nccl, f'libnccl.so.2 not resolved to system /usr/lib in ld cache'; \
    print(f'torch {torch.__version__} sglang-kernel {m.version(\"sglang-kernel\")} nccl→{nccl[0].split(\"=>\")[-1].strip()}')"
