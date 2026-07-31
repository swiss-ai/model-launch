#!/bin/bash
sml advanced \
  --system clariden \
  --partition normal \
  --framework sglang \
  --environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/MLLM/OCR/rednote-hilab_dots.mocr \
    --host 0.0.0.0 \
    --served-model-name dots_mocr-$(whoami) \
    --dp-size 4 \
    --tp-size 1 \
    --trust-remote-code \
    --context-length 16384 \
    --mem-fraction-static 0.85 \
    --enable-metrics"
