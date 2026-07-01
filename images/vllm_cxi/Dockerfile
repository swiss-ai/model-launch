# =============================================================================
# vllm-cxi — vLLM image with Slingshot (HSN) NCCL transport
# =============================================================================
#
# Why this image exists
# ---------------------
# The official vLLM image ships with NCCL's built-in TCP / InfiniBand
# transport.  CSCS breithorn uses HPE Slingshot as the inter-node fabric,
# accessed through the CXI libfabric provider.  NCCL reaches Slingshot via
# the aws-ofi-nccl plugin:
#
#   vLLM
#    └── PyTorch  (torch.distributed, collective ops)
#          └── NCCL  (GPU-to-GPU allreduce / allgather)
#                └── aws-ofi-nccl  (NCCL net plugin, loaded at startup)
#                      └── libfabric  (fabric abstraction API)
#                            └── CXI provider  (Slingshot, from host mount)
#                                  └── Slingshot HSN
#
# vLLM and PyTorch are completely unmodified; the plugin is transparent to
# them.  NCCL dlopen()s the plugin when NCCL_NET_PLUGIN is set.
#
# Build stages
# ------------
# Stage 1 (builder): compiles aws-ofi-nccl and its two build-time deps from
# source inside a CUDA devel image:
#
#   GDRCopy      — GPU Direct RDMA helper; enables zero-copy GPU↔NIC DMA.
#                  Required by libfabric's --with-gdrcopy.
#
#   libfabric    — the OFI abstraction library.  We build a generic version
#                  (no CXI provider) to obtain headers and a stub .so for
#                  the aws-ofi-nccl configure step.  At runtime the host's
#                  CXI-enabled libfabric replaces it via LD_LIBRARY_PATH and
#                  hostPath mounts.
#                  IMPORTANT: LIBFABRIC_VER must match the ABI of the
#                  libfabric installed on the target compute nodes.
#                  Check: ls /opt/cray/libfabric/ on the node.
#                  libfabric < 1.20  → ABI 1.6
#                  libfabric ≥ 1.20  → ABI 1.7 (incompatible with 1.6!)
#
#   aws-ofi-nccl — the NCCL net plugin.  Built with --enable-cudart-dynamic
#                  so the binary does not embed a CUDA version; it works with
#                  whatever CUDA the runtime vLLM image provides.
#                  Output: libnccl-net-ofi.so → renamed to libnccl-net.so
#                  (NCCL's conventional name for a net plugin).
#
# Stage 2 (runtime): copies the plugin into the official vLLM image and sets
# NCCL_NET_PLUGIN so NCCL loads it automatically at startup.
#
# Runtime requirements — pod spec responsibilities
# -------------------------------------------------
# The host CXI stack is NOT baked into this image; it is injected at runtime:
#
#   Volume mounts (hostPath):
#     /dev/cxi[0-3]             — CXI NIC char devices (one per HSN port)
#     /opt/cray                 — Cray libfabric, PMI, and CXI runtime libs
#     /usr/lib64/libcxi.so.1    — CXI provider shared library
#     /usr/lib64/libcxiutils.so.0
#
#   Network:
#     hostNetwork: true         — exposes hsn0-3 to the container
#
#   Security:
#     privileged: true          — /dev/cxi* require "a *:* rwm" cgroup entry;
#                                 kubelet does not add char devs to the device
#                                 cgroup automatically (fix: device plugin with
#                                 Allocate hook, not yet deployed)
#
#   Environment variables (set in the pod spec, not here):
#     LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/cray/libfabric/<ver>/lib64:/host/lib64
#     FI_PROVIDER=cxi
#     NCCL_NET_GDR_LEVEL=PHB       # GPU Direct RDMA: GPU and CXI share PCIe
#     NCCL_CROSS_NIC=1             # stripe across hsn0-3 for multi-rail
#     NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
#     FI_CXI_DISABLE_HOST_REGISTER=1
#     FI_CXI_DEFAULT_CQ_SIZE=131072
#     FI_CXI_RX_MATCH_MODE=hybrid
#     OFI_NCCL_PROTOCOL=SENDRECV   # CXI lacks RDMA FI caps; use SENDRECV
#     GLOO_SOCKET_IFNAME=hsn0      # Gloo CPU side-channel also over HSN
#
#   See README.md for the full annotated env-var reference.
#
# Updating versions
# -----------------
# 1. Check the libfabric version on the target nodes: ls /opt/cray/libfabric/
# 2. Update LIBFABRIC_VER — must match node ABI (see note above).
# 3. Update AWS_OFI_NCCL_VER — can usually be updated independently.
# 4. Update GDRCOPY_VER only if GDRCopy API changes (rare).
# 5. VLLM_IMAGE tracks the repo's vllm_cuda13 image; bump vLLM by rebuilding
#    that image, not here. Keep it a multi-arch tag or the aarch64 build breaks.
# 6. Update CUDA_BUILD_IMAGE to match the CUDA major version in VLLM_IMAGE
#    (vllm_cuda13 is CUDA 13). --enable-cudart-dynamic keeps the plugin
#    tolerant of minor CUDA differences (builder 13.1 vs base 13.0 is fine).
# =============================================================================

# nvidia/cuda devel is multi-arch (arm64 + amd64), so stage 1 builds natively
# on both the Grace (aarch64) and x86_64 CI runners without changes.
ARG CUDA_BUILD_IMAGE=docker.io/nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04

# Base for stage 2. We layer the CXI plugin on THIS repo's own vLLM image
# rather than docker.io/vllm/vllm-openai, because the upstream image is
# amd64-only and breithorn's Slingshot nodes are GH200 (aarch64). vllm_cuda13
# is built for BOTH arches by this same CI (:latest is a multi-arch manifest)
# and already ships torch + vllm + ray[default], so the arm64 build works and
# the two arches stay on an identical vLLM build.
#   Trade-off: you get the repo's vLLM (cu130 nightly), not an upstream pinned
#   vllm-openai release. If you need a specific upstream vLLM version, build it
#   from source for aarch64 instead of switching this base back to amd64-only.
ARG VLLM_IMAGE=ghcr.io/swiss-ai/vllm_cuda13:latest

# -----------------------------------------------------------------------------
# Stage 1: build the aws-ofi-nccl plugin and its dependencies
# -----------------------------------------------------------------------------
FROM ${CUDA_BUILD_IMAGE} AS builder

# Pin all three component versions together.  Changing any one may require
# re-validating the others; do it deliberately, not as part of an unrelated
# change.
ARG LIBFABRIC_VER=2.3.1
ARG GDRCOPY_VER=2.5.1
ARG AWS_OFI_NCCL_VER=v1.19.1

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git autotools-dev libtool libhwloc-dev rename && \
    rm -rf /var/lib/apt/lists/*

# --- GDRCopy ---
# GPU Direct RDMA copy library.  Installed to /usr/local before libfabric so
# that libfabric's ./configure --with-gdrcopy can locate the headers.
# hadolint ignore=DL3003
RUN git clone --depth 1 -b v${GDRCOPY_VER} https://github.com/NVIDIA/gdrcopy.git && \
    cd gdrcopy && \
    export CUDA_PATH="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}" && \
    make CC=gcc CUDA="$CUDA_PATH" lib && \
    make lib_install && \
    cd .. && rm -rf gdrcopy

# --- libfabric ---
# Built from source at a version matching the ABI on the target compute nodes.
# --enable-gdrcopy-dlopen: libfabric loads GDRCopy at dlopen time; the binary
#   remains portable across nodes that may or may not have GDRCopy installed.
# --enable-cuda-dlopen: same treatment for CUDA — no hard link to a specific
#   cuda runtime SO.
# Note: no --with-cxi or similar; the CXI provider comes from the host mount
# at runtime.  We only need the generic libfabric headers and stub SO here.
# hadolint ignore=DL3003
RUN git clone --depth 1 -b v${LIBFABRIC_VER} https://github.com/ofiwg/libfabric.git && \
    cd libfabric && \
    autoupdate && ./autogen.sh && \
    export CUDA_PATH="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}" && \
    CC=gcc ./configure \
        --prefix=/opt/libfabric --disable-verbs --disable-psm2 \
        --disable-psm3 --disable-efa --disable-opx --disable-shm \
        --disable-sockets --disable-udp --disable-rxm --disable-rxd \
        --disable-mrail --disable-tcp --disable-usnic --disable-rstream \
        --disable-hook_debug --disable-hook_hmem \
        --with-gdrcopy=/usr/local \
        --enable-gdrcopy-dlopen \
        --with-cuda="${CUDA_PATH}" \
        --enable-cuda-dlopen && \
    make -j"$(nproc)" && make install && \
    cd .. && rm -rf libfabric

# --- aws-ofi-nccl ---
# The NCCL net plugin.  Key configure flags:
#   --disable-tests: no test binaries; keeps the install directory small.
#   --without-mpi:   vLLM / Ray do not use MPI.
#     symlink pointing at libnccl-net-ofi.so.  Disabling it avoids confusion
#     when we do the rename ourselves below.
#   --enable-cudart-dynamic: link CUDA runtime via dlopen(3) rather than at
#     link time.  The plugin binary is therefore CUDA-version-agnostic and
#     works inside any vLLM image regardless of its bundled CUDA.
#   --prefix=/opt/aws-ofi-nccl: install path that will be COPY'd to stage 2.
# hadolint ignore=DL3003
RUN git clone --depth 1 -b ${AWS_OFI_NCCL_VER} https://github.com/aws/aws-ofi-nccl.git && \
    cd aws-ofi-nccl && \
    ./autogen.sh && \
    export CUDA_PATH="${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}" && \
    CC=gcc ./configure \
        --disable-tests \
        --disable-nccl-net-symlink \
        --enable-cudart-dynamic \
        --with-libfabric=/opt/libfabric \
        --prefix=/opt/aws-ofi-nccl \
        --with-cuda="${CUDA_PATH}" && \
    make -j"$(nproc)" && make install && \
    cd .. && rm -rf aws-ofi-nccl

# Rename libnccl-net-ofi.so → libnccl-net.so.
# NCCL_NET_PLUGIN accepts either a bare name ("ofi" → looks for
# libnccl-net-ofi.so on LD_LIBRARY_PATH) or a full absolute path (any name).
# We use the full-path form in ENV below, but the conventional name avoids
# confusion when the SO is inspected manually.
RUN rename -v 's/-ofi//' /opt/aws-ofi-nccl/lib/*ofi* 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 2: final image — repo multi-arch vLLM + the plugin
# -----------------------------------------------------------------------------
FROM ${VLLM_IMAGE}

# The vllm_cuda13 base already provides torch + vllm + ray[default]==2.55.0, so
# we do NOT reinstall ray here (reinstalling unpinned would risk drifting off
# the base's pinned version).
#
# sentencepiece + tiktoken are needed by models whose tokenizer_config.json
# specifies a fast tokenizer that has to be converted from a slow one at load
# time (GLM-5.1 and similar). Without them HF raises
# "Couldn't instantiate the backend tokenizer … You need to have sentencepiece
# or tiktoken installed to convert a slow tokenizer to a fast one."
# hadolint ignore=DL3013
RUN pip install --no-cache-dir sentencepiece tiktoken

# Copy only the installed plugin tree (~a few MB); all build tooling stays in
# the builder stage and is discarded here.
COPY --from=builder /opt/aws-ofi-nccl /opt/aws-ofi-nccl

# Tell NCCL to load the OFI plugin at startup.  Full absolute path is
# unambiguous and survives any LD_LIBRARY_PATH changes in the pod spec.
# The plugin in turn looks for libfabric.so (provided at runtime via the
# /opt/cray hostPath mount) and CUDA (already in the vLLM image).
ENV NCCL_NET_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-net.so

# Add the plugin's lib directory to LD_LIBRARY_PATH so its own transitive
# dependencies (e.g. libgdrcopy) are found without requiring the pod spec to
# list this path explicitly.
ENV LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:${LD_LIBRARY_PATH:-}

# Pre-create directories that the pod spec mounts over at runtime.
# If these paths do not exist, kubelet bind-mount may silently fail or error.
#   /opt/cray   — Cray libfabric runtime, PMI libs, CXI utilities
#   /host/lib64 — receives libcxi.so.1 and libcxiutils.so.0 from the node
RUN mkdir -p /opt/cray /host/lib64
