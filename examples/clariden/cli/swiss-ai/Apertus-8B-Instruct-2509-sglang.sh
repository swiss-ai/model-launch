#!/bin/bash
# Note: a model named swiss-ai/Apertus-8B-Instruct-2509 is usually already running.
# The --served-model-name flag avoids name collisions.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-$(whoami) \
    --host 0.0.0.0 \
    "
