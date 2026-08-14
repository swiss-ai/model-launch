#!/bin/bash
# Build a TP-presharded checkpoint for servekit's fast weight loading, split
# across the nodes/GPUs this job is allocated (nodes x --gpus-per-node = TP).
#
# Override any of these at submit time, e.g.:
#   MODEL=/path/to/hf/checkpoint OUT=/path/to/out TP=8 sbatch --nodes=2 prepare.sbatch
#
#   MODEL  - source checkpoint, a local HF-format directory (default: Apertus-70B-Instruct-2509)
#   OUT    - where to write the presharded checkpoint + servekit.json manifest
#   TP     - tensor-parallel size the shards are locked to; must equal nodes x gpus-per-node
#   EDF    - container environment definition file (sglang image + NCCL/aws_ofi_nccl hooks)
#   --nodes (sbatch flag, not an env var) - how many nodes to shard across; keep TP in sync
#SBATCH --job-name=servekit-prepare
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-70B-Instruct-2509}"
OUT="${OUT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus70b-tp8-sharded}"
TP="${TP:-8}"
NNODES="${SLURM_JOB_NUM_NODES}"
DIST_PORT=20000

REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)"
EDF="${EDF:-${REPO_ROOT}/src/swiss_ai_model_launch/assets/envs/sglang.toml}"
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
unset BASH_ENV
pip install --quiet git+https://github.com/eth-easl/servekit.git@e9fde0b

servekit prepare --model "${MODEL}" --out "${OUT}" --tp "${TP}" \
  --nnodes ${NNODES} \
  --node-rank \${SLURM_PROCID} \
  --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
  -- \
  --trust-remote-code \
  --context-length 65536 \
  --mem-fraction-static 0.85
EOF
)

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== prepared checkpoint ==="
for rank in $(seq 0 $((TP - 1))); do
  [ -e "${OUT}/model-rank-${rank}-part-0.safetensors" ] || { echo "GATE FAIL: rank ${rank} wrote no shards" >&2; exit 1; }
done
[ -e "${OUT}/config.json" ] || { echo "GATE FAIL: no config.json" >&2; exit 1; }
[ -e "${OUT}/servekit.json" ] || { echo "GATE FAIL: no manifest" >&2; exit 1; }
du -sh "${OUT}"
ls "${OUT}"
cat "${OUT}/servekit.json"
