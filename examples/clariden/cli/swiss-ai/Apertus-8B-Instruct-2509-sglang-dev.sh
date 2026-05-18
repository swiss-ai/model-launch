#!/bin/bash
# Dev mesh variant of Apertus-8B-Instruct-2509-sglang.sh.
# --dev points OpenTela at the dev bootstrap peer (.177) instead of prod (.178).
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-reservation SD-69241-apertus-1-5-0 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami)-dev \
    --host 0.0.0.0 \
    --enable-metrics" \
  --dev
