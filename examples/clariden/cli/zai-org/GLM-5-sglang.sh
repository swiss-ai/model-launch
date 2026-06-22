#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --nodes-per-replica 8 \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5 \
    --served-model-name zai-org/GLM-5-$(whoami) \
    --tp-size 32 \
    --host 0.0.0.0 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --mem-fraction-static 0.85 \
    --enable-metrics"
