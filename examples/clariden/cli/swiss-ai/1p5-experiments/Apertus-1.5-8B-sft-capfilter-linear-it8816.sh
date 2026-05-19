#!/bin/bash

# https://swissai-initiative.slack.com/archives/C0A9JJ7C5K6/p1779184636000049 
# to comp between constant and linear
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-reservation SD-69241-apertus-1-5-0 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-8B-sft-capfilter-linear-it8816 \
    --served-model-name swiss-ai/Apertus-1.5-8B-sft-capfilter-linear-it8816 \
    --tokenizer /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6"
