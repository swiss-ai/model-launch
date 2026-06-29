#!/bin/bash
sml advanced \
  --partition normal \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-70b-sft-262k-2400 \
    --served-model-name swiss-ai/ap1p5-70b-sft-262k-2400 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling"
