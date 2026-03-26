#!/bin/bash
sml advanced \
  --slurm-nodes 2 \
  --serving-framework sglang \
  --disable-ocf \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Mixtral-8x22B-Instruct-v0.1 \
    --host 0.0.0.0 \
    --port 8080 \
    --tp-size 8 \
    --served-model-name mistralai/Mixtral-8x22B-Instruct-v0.1-$(whoami)"
