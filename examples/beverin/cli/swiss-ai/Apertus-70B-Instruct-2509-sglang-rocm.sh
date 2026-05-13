#!/bin/bash

sml advanced \
  --firecrest-system beverin \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang_rocm.toml \
  --slurm-time "12:00:00" \
  --partition mi300 \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-70B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang \
    --host 0.0.0.0 \
    --tp-size 4 \
    --mem-fraction-static 0.5 \
    --enable-metrics"
