#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
   --slurm-time 6:00:00 \
   --serving-framework vllm \
   --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_qwen3_omni.toml \
   --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Qwen/Qwen3-Omni-30B-A3B-Captioner \
    --served-model-name Qwen/Qwen3-Omni-30B-A3B-Captioner-$(whoami) \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --dtype bfloat16 --max-model-len 32768 --trust-remote-code"