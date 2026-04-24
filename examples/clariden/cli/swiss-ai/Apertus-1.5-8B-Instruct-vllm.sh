#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-reservation SD-69241-apertus-1-5 \
  --slurm-time 6:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf-checkpoints/Apertus-1p5-8B-it430000 \
    --served-model-name swiss-ai/apertus1.5 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 8192 \
    --limit-mm-per-prompt \"{\"image\": 1}\" \
    --mm-processor-kwargs \"{\"apertus_vision_tokenizer_device\":\"cuda\",\"apertus_emu35_codebase\":\"/workspace/Emu3.5\"}\" \
    --gpu-memory-utilization 0.6"
