#!/bin/bash
# Apertus-8B TP1 verification profile with an 8 GiB LMCache RAM tier.
# Usage: bash examples/clariden/lmcache/launch.sh [vllm|sglang] [sml options]
# MODEL_PATH, SML_ACCOUNT, and SML_BIN can override the local defaults.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
engine="${1:-vllm}"
if [[ $# -gt 0 ]]; then shift; fi
model="${MODEL_PATH:-/iopsstor/scratch/cscs/anunay/models/Apertus-8B-Instruct-2509}"
pre_launch_cmds=$'ulimit -c 0\nexport OPENTELA_BIN=/opentelabin/prod/otela-arm64'
case "$engine" in
  vllm)
    framework_args="--model $model --served-model-name swiss-ai/lmcache-apertus8b-vllm
      --host 0.0.0.0 --tensor-parallel-size 1 --dtype bfloat16 --enforce-eager
      --max-model-len 10240 --max-num-seqs 2 --kv-cache-memory-bytes 1610612736
      --max-num-batched-tokens 2048 --enable-prefix-caching --generation-config vllm
      --compilation-config.pass_config.fuse_allreduce_rms false
      --config $script_dir/vllm-engine.yaml"
    ;;
  sglang)
    framework_args="--model-path $model --served-model-name swiss-ai/lmcache-apertus8b-sglang
      --host 0.0.0.0 --tp-size 1 --dtype bfloat16 --disable-cuda-graph
      --context-length 12288 --max-total-tokens 12288 --max-running-requests 2
      --chunked-prefill-size 2048 --mem-fraction-static 0.65 --page-size 64
      --attention-backend flashinfer --enable-metrics --enable-lmcache"
    pre_launch_cmds+=$'\nexport LMCACHE_LOG_LEVEL=DEBUG'
    pre_launch_cmds+=$'\n'"export LMCACHE_CONFIG_FILE=$script_dir/sglang-lmcache.yaml"
    ;;
  *)
    echo "Usage: $0 [vllm|sglang] [additional sml arguments]" >&2
    exit 2
    ;;
esac
framework_args="${framework_args//$'\n'/ }"
cd -- "$project_root"
exec "${SML_BIN:-$project_root/.venv/bin/sml}" advanced \
  --system clariden --account "${SML_ACCOUNT:-infra01}" \
  --partition debug --time 01:30:00 --replicas 1 --nodes-per-replica 1 \
  --framework "$engine" --environment "$script_dir/$engine.toml" \
  --disable-metrics --disable-dcgm-exporter --pre-launch-cmds "$pre_launch_cmds" \
  --no-tui --framework-args "$framework_args" "$@"
