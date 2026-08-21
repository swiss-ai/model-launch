#!/bin/bash
# flashinfer is required: the sglang-kernel wheels ship without FA3 on
# aarch64, so the fa3 attention backend is unavailable on this cluster.
sml advanced \
  --tui \
  --partition normal \
  --framework sglang \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/sglang_apertus_1.5.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-v1.5-8B \
    --served-model-name swiss-ai/Apertus-v1.5-8B-sglang-$(whoami) \
    --attention-backend flashinfer \
    --mem-fraction-static 0.6 \
    --tool-call-parser apertus2509"
