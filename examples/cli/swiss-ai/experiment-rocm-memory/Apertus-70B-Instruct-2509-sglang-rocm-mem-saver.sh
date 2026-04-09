#!/bin/bash

sml advanced \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang_rocm.toml \
  --slurm-time "12:00:00" \
  --partition mi300 \
  --pre-launch-cmds "pip install torch-memory-saver" \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-70B-Instruct-2509 \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-mem-saver \
    --host 0.0.0.0 \
    --port 8080 --tp-size 4 \
    --enable-memory-saver "
