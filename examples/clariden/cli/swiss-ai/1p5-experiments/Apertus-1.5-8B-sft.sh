#!/bin/bash

# from https://swissai-initiative.slack.com/archives/C0A9JJ7C5K6/p1780484229747019?thread_ts=1780480992.715039&cid=C0A9JJ7C5K6
# purpose of this is to compare the SFT-DPO model to the SFT-only model, which is in the other script in this directory

sml advanced \
  --system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --time 12:00:00 \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200 \
    --served-model-name swiss-ai/Apertus-1.5-8B-Instruct-sft \
    --tokenizer /capstor/store/cscs/swissai/infra01/hf_tokenizers/tokenizers/Apertus-v1p5-tool_output_toks-think_toks \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8080 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling"
