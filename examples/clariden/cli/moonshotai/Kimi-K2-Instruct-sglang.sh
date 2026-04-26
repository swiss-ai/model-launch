#!/bin/bash
# Runs with 4 nodes, TP16. Requires blobfile. Requires some time to start.
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes-per-replica 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang_kimi.toml \
  --pre-launch-cmds "pip install blobfile" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2-Instruct \
    --served-model-name moonshotai/Kimi-K2-Instruct-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tool-call-parser kimi_k2"
