#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-reservation SD-69241-apertus-1-5 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /iopsstor/scratch/cscs/hyukhymenko/apertus-sft-runs/ap-1p5-cooldown-sft-21-04-lr-8e-5/2026-04-23_19-08-56/global_step_9688/huggingface \
    --served-model-name swiss-ai/apertus1p5-lr-8e-5-2026-04-23_19-08-56 \
    --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6"
