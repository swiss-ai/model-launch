#!/bin/bash

# from https://swissai-initiative.slack.com/archives/C0A9JJ7C5K6/p1780418221820059

sml advanced \
  --system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --time 12:00:00 \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/Alignment/ap_1p5/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-online-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout8-images-2453762-2453767 \
    --served-model-name swiss-ai/Apertus-1.5-8B-Instruct-sft-dpo \
    --tokenizer /capstor/store/cscs/swissai/infra01/hf_tokenizers/tokenizers/Apertus-v1p5-tool_output_toks-think_toks \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling"