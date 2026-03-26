#!/bin/bash
# Runs with 4 nodes, TP16. Uses custom glm45 reasoning and tool-call parsers.
# No bundled env available — provide your own
SGLANG_GLM_ENV=<path/to/sglang_glm.toml>

sml advanced \
  --slurm-nodes 4 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment "$SGLANG_GLM_ENV" \
  --pre-launch-cmds "pip install blobfile" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.6 \
    --tp-size 16 \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name zai-org/GLM-4.6-$(whoami) \
    --trust-remote-code \
    --tool-call-parser glm45 \
    --reasoning-parser glm45"
