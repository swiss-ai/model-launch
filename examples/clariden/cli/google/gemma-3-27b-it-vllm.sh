#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/google/gemma-3-27b-it \
    --served-model-name google/gemma-3-27b-it-$(whoami) \
    --host 0.0.0.0 \
    --tensor-parallel-size 4"
