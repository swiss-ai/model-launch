#!/bin/bash

sml advanced \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 12:00:00 \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_5800 \
    --served-model-name swiss-ai/Apertus-1.5-8B-Instruct-LC-5800 \
    --tokenizer /capstor/store/cscs/swissai/infra01/hf_tokenizers/tokenizers/Apertus-v1p5-tool_output_toks-think_toks \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --skip-mm-profiling \
    --gpu-memory-utilization 0.6"