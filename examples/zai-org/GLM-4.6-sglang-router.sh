#!/bin/bash
# 2 workers x 4 nodes each. Requires latest sglang env. Experimental.
# No bundled env available — provide your own
sml advanced \
  --slurm-nodes 8 \
  --slurm-workers 2 \
  --slurm-nodes-per-worker 4 \
  --use-router \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment <path/to/sglang_latest.toml> \
  --pre-launch-cmds "pip install blobfile" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.6 \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name zai-org/GLM-4.6-$(whoami) \
    --trust-remote-code \
    --tool-call-parser glm45 \
    --reasoning-parser glm45"
