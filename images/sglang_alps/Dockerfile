# SGLang on the CSCS Alps extended NGC PyTorch base.
#
# Base image courtesy of CSCS — see https://github.com/eth-cscs/alps-extended-images
# for the Slingshot-tuned NGC PyTorch builds (NCCL, NVSHMEM, OpenMPI, libfabric/CXI).

ARG BASE_IMAGE=ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.01-py3-alps4

FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_VERSION=2.9.1
ARG CUDA_VERSION_PATH=cu130
ARG SGLANG_KERNEL_VERSION=0.4.1
ARG SGLANG_VERSION=0.5.10.post1
ARG SGLANG_ROUTER_VERSION=0.3.2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_LINK_MODE=copy \
    UV_BREAK_SYSTEM_PACKAGES=1

# Pin cu130 torch first via --torch-backend, then install the cu130 sglang-kernel
# wheel from sgl-project/whl (PyPI only ships generic manylinux wheels — the
# cu130 local-version variants are GitHub-only). flash-attn-4 is a pure-Python
# wheel that JITs hopper/blackwell kernels via nvidia-cutlass-dsl at runtime,
# so no from-source FA3 build stage is needed.
# Drop pip NCCL/cuBLAS in the same layer so torch dlopens the alps-built
# libnccl from /usr and the CUDA 13 toolkit's cuBLAS.
RUN pip install --no-cache-dir uv==0.11.7 && \
    uv pip install --system \
        torch==${TORCH_VERSION} torchvision==0.24.1 torchaudio==2.9.1 \
        --torch-backend=${CUDA_VERSION_PATH} && \
    uv pip install --system --no-deps --force-reinstall \
        "sglang-kernel @ https://github.com/sgl-project/whl/releases/download/v${SGLANG_KERNEL_VERSION}/sglang_kernel-${SGLANG_KERNEL_VERSION}+${CUDA_VERSION_PATH}-cp310-abi3-manylinux2014_aarch64.whl" && \
    uv pip install --system --prerelease=allow \
        "sglang[all]==${SGLANG_VERSION}" \
        "sglang-router==${SGLANG_ROUTER_VERSION}" \
        "flash-attn-4>=4.0.0b4" \
        "nvidia-cutlass-dsl>=4.4.1" \
        "nvidia-cuda-nvrtc-cu12" && \
    uv pip uninstall --system nvidia-nccl-cu13 nvidia-nccl-cu12 \
                              nvidia-cublas-cu13 nvidia-cublas-cu12 nvidia-cublas

# Register pip-installed nvidia/* lib dirs in the dynamic linker cache.
# The alps container hooks rewrite LD_LIBRARY_PATH at job launch to point at
# host CUDA mounts, which removes the NGC base image's defaults — without
# this ldconfig step, dlopen("libnvrtc.so.12") from sgl_kernel can't find
# the wheel-shipped libs even though ldd resolved them at build time.
RUN python - <<'PY' >/etc/ld.so.conf.d/zz-nvidia-pip.conf
import sysconfig, glob, os
sp = sysconfig.get_paths()["purelib"]
for d in sorted(glob.glob(f"{sp}/nvidia/*/lib")):
    if any(f.endswith(".so") or ".so." in f for f in os.listdir(d)):
        print(d)
PY
RUN ldconfig && cat /etc/ld.so.conf.d/zz-nvidia-pip.conf

# Verify curl (for router health checks), assert torch is a CUDA build, confirm
# the sm90 common_ops .so was shipped in the installed kernel wheel, and resolve
# its NEEDED libraries against the install set. We can't actually `import
# sgl_kernel` at build time — its loader probes the GPU via libcuda.so.1, which
# isn't present without an NVIDIA runtime mount. The ldd check + arch presence
# check together catch both "wrong wheel arch" and "missing libnvrtc/libnvjit"
# at build time, instead of at job launch.
RUN command -v curl >/dev/null 2>&1 || { echo "curl missing from base image"; exit 1; } && \
    python - <<'PY'
import sys, glob, sysconfig, subprocess, importlib.metadata, torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
if torch.version.cuda is None:
    sys.exit(f"expected a CUDA torch build, got {torch.__version__}")
print(f"nccl: {torch.cuda.nccl.version()}")
print(f"sglang-kernel {importlib.metadata.version('sglang-kernel')}")
sp = sysconfig.get_paths()["purelib"]
sos = glob.glob(f"{sp}/sgl_kernel/sm90/common_ops*.so")
if not sos:
    sys.exit(f"sm90/common_ops*.so missing under {sp}/sgl_kernel — wrong arch wheel?")
print(f"sm90 ops: {sos[0]}")
ldd = subprocess.run(["ldd", sos[0]], capture_output=True, text=True).stdout
missing = [ln.strip() for ln in ldd.splitlines() if "not found" in ln and "libcuda.so.1" not in ln]
if missing:
    sys.exit("unresolved deps in common_ops.so (libcuda.so.1 expected at runtime):\n  " + "\n  ".join(missing))
print("ldd: all deps resolved (modulo libcuda.so.1)")
# Confirm dlopen against ld.so.cache (LD_LIBRARY_PATH-independent) — this
# catches the runtime-only failure where the alps hooks rewrite LD paths.
ldcache = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
for needed in ["libnvrtc.so.12"]:
    if needed not in ldcache:
        sys.exit(f"{needed} not in ld.so.cache — dlopen will fail at runtime")
    print(f"ld.so.cache has {needed}")
PY
