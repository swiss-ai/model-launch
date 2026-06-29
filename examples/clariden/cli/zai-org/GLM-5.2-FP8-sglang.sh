#!/bin/bash
# also can be run with src/swiss_ai_model_launch/assets/envs/sglang.toml
sml advanced \
  --tui \
  --system clariden \
  --partition normal \
  --nodes-per-replica 4 \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang_deepep.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2-FP8 \
    --served-model-name zai-org/GLM-5.2-FP8-$(whoami) \
    --tp-size 16 \
    --host 0.0.0.0 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --mem-fraction-static 0.8"
