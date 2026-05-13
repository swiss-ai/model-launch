#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/apertus_1p5/sft_in_cooldown_ablation/apertus-1p5-8b-64node-sft-cooldown-1pct-vision-sft/HF \
    --served-model-name swiss-ai/Apertus-1.5-8B-Instruct-cooldown-1pct-vision-sft \
    --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6"