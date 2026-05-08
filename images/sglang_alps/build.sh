#!/bin/bash
set -euo pipefail

IMAGE_NAME=sglang_alps
IMAGE_DIR=images/${IMAGE_NAME}
DEST=/capstor/store/cscs/swissai/infra01/container-images/ci/${IMAGE_NAME}.sqsh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_TAG="${SLURM_JOB_ID:-$$}"
SCRATCH_SQSH="${SCRATCH}/${IMAGE_NAME}-${JOB_TAG}.sqsh"
TAG="${IMAGE_NAME}:srun-${JOB_TAG}"

export DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/podman-runtime-${JOB_TAG}"
mkdir -p "${XDG_RUNTIME_DIR}"

cleanup() {
    podman rmi "${TAG}" 2>/dev/null || true
    rm -f "${SCRATCH_SQSH}" 2>/dev/null || true
    rm -rf "${XDG_RUNTIME_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

cd "${REPO_ROOT}"

echo "=== [$(date)] podman build ${TAG} from ${IMAGE_DIR} ==="
podman build -t "${TAG}" "${IMAGE_DIR}"

echo "=== [$(date)] enroot import -> ${SCRATCH_SQSH} ==="
rm -f "${SCRATCH_SQSH}"
# enroot can exit non-zero on benign cleanup warnings (e.g. cgroup access)
# even when the sqsh was written successfully — gate on file size, not exit.
enroot import -o "${SCRATCH_SQSH}" "podman://${TAG}" || true
test -s "${SCRATCH_SQSH}" || { echo "enroot import produced empty sqsh"; exit 1; }

echo "=== [$(date)] smoke test sgl_kernel import on GPU ==="
srun --overlap --ntasks=1 \
     --container-image="${SCRATCH_SQSH}" \
     --container-mounts=/capstor,/iopsstor \
     python -c "import sgl_kernel; print('sgl_kernel ok:', sgl_kernel.__file__)"

echo "=== [$(date)] install -> ${DEST} ==="
mkdir -p "$(dirname "${DEST}")"
cp "${SCRATCH_SQSH}" "${DEST}.tmp"
mv "${DEST}.tmp" "${DEST}"
chmod o+rx "${DEST}"

echo "=== [$(date)] done ==="
ls -lh "${DEST}"
