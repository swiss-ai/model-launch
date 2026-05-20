#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mistral-Small-24B-Instruct-2501 \
    --host 0.0.0.0 \
    --served-model-name mistralai/Mistral-Small-24B-Instruct-2501-$(whoami) \
    --dp-size 4 \
    --enable-metrics"
