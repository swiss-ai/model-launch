#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-reservation SD-69241-apertus-1-5 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/apertus-8b-sft-1.5--lr8e-5-MaxMin_4096-Filtered-dpo-lr1e-06-beta25.0-lenNormTrue-ebs128-ep1 \
    --served-model-name swiss-ai/Apertus-1.5-8B-Instruct \
    --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6"
