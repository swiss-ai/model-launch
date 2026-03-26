#!/bin/bash
# No bundled env available — provide your own
SGLANG_GLM_ENV=<path/to/sglang_glm.toml>

sml advanced \
  --slurm-nodes 8 \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment "$SGLANG_GLM_ENV" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5 \
    --served-model-name zai-org/GLM-5-$(whoami) \
    --tp-size 32 \
    --host 0.0.0.0 \
    --port 8080 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.85 \
    --disable-cuda-graph"
