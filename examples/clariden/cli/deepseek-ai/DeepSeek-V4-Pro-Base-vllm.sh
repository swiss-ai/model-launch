#!/bin/bash
# DeepSeek-V4-Pro-Base, 1.6T params, fp8 block-quantised. 8 nodes / 32 GH200.
# Same setup story as DeepSeek-V4-Pro-vllm.sh (read its header for the full
# rationale) -- this is the larger sibling, so a couple of points are sharper:
#
#   * Image: vllm_deepseek_v4 (vllm 0.21.0 + transformers 5.9.0). deepseek_v4 is
#     not in mainline sglang/transformers yet, and the checkpoint ships no remote
#     code, so the stock images can't load it.
#   * --kv-cache-dtype fp8: required -- V4's FlashMLA-Sparse backend is fp8-only.
#   * TP=8 + PP=4 (not TP=32): fp8 weights are block-quantised in 128x128 blocks,
#     so each TP shard must be a multiple of 128. TP=32 shards a 3072 dim to 96
#     (96 % 128 != 0) and fails; TP=8 gives 384 (=3x128). PP=4 fans across all
#     8 nodes / 32 GPUs.
#   * Warmup / JIT timeout: requests JIT-compile the sparse-attn Triton kernels
#     the first time each new shape is seen. At this size that took ~5.5 min and
#     overran vllm's default execute-model RPC deadline -> "RPC call to
#     sample_tokens timed out" and the engine died. Note this is NOT first-request
#     only -- a new-shape prompt later in the session can re-trigger it (it took
#     down the V4-Pro replica too). Mitigation: VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
#     is raised in src/swiss_ai_model_launch/assets/envs/vllm_deepseek_v4.toml.
#     A sturdier alternative is to pre-warm the expected shapes at startup.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 8 \
  --slurm-time 3:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_deepseek_v4.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Pro-Base \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro-Base-$(whoami) \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --trust-remote-code"
