#!/bin/bash
# Post-trained DeepSeek-V4-Pro (~806 GB on disk), 4 nodes / 16 GH200.
# Verified serving on Clariden 2026-05-29 (returns coherent completions).
#
# Notes from getting this to run (each was a separate failure we hit):
#
#   1. Image. deepseek_v4 is NOT in mainline sglang/transformers yet. The stock
#      sglang_cuda13 image (sglang 0.5.10 + transformers 5.3.0) errors with
#      `model type deepseek_v4 ... not recognized` / `KeyError: 'deepseek_v4'`,
#      and the checkpoint ships no remote code so --trust-remote-code can't help.
#      Fix: the vllm_deepseek_v4 image pins official releases that DO support it
#      -- vllm 0.21.0 (ships DeepseekV4ForCausalLM) + transformers 5.9.0
#      (registers the deepseek_v4 model_type). See images/vllm_deepseek_v4.
#
#   2. --kv-cache-dtype fp8 is mandatory. V4's only attention backend in vllm is
#      FlashMLA-Sparse, which stores the (MLA latent) KV cache exclusively in
#      DeepSeek's fp8_ds_mla format. The default `auto` trips an assert
#      ("DeepseekV4 only supports fp8 kv-cache format for now").
#
#   3. TP must stay 128-aligned. Weights are fp8 block-quantised in 128x128
#      blocks, so every tensor-parallel shard must be a multiple of 128.
#      --tensor-parallel-size 16 shards a dim of 3072 to 192 (not a multiple of
#      128) -> "input_size_per_partition = 192 is not divisible by block_k=128".
#      So use TP=8 (3072/8 = 384 = 3x128) and reach all 16 GPUs via PP=2.
#
#   4. First-request warmup. The first completion JIT-compiles the sparse-attn
#      Triton kernels; for V4-Pro this finishes within vllm's RPC deadline, but
#      for the larger Base model it can exceed it -- see the Base example, which
#      bumps VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS in the env toml.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 4 \
  --serving-framework vllm \
  --slurm-time 3:00:00 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_deepseek_v4.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/deepseek-ai/DeepSeek-V4-Pro \
    --served-model-name deepseek-ai/DeepSeek-V4-Pro-$(whoami) \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --kv-cache-dtype fp8 \
    --host 0.0.0.0 \
    --trust-remote-code"
