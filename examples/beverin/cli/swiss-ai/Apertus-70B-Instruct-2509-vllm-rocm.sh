#!/bin/bash

sml advanced \
  --firecrest-system beverin \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_rocm.toml \
  --slurm-time "12:00:00" \
  --partition mi300 \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-70B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm \
    --host 0.0.0.0 \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.85"
