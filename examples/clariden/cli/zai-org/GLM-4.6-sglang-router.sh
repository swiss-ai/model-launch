#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-replicas 2 \
  --slurm-nodes-per-replica 4 \
  --router SGL \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --pre-launch-cmds "pip install blobfile" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.6 \
    --tp-size 16 \
    --host 0.0.0.0 \
    --served-model-name zai-org/GLM-4.6-$(whoami) \
    --trust-remote-code \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --enable-metrics"
