#!/bin/bash
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --serving-framework vllm \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/mistralai/Ministral-3-8B-Instruct-2512 \
    --served-model-name mistralai/Ministral-3-8B-Instruct-2512-$(whoami) \
    --host 0.0.0.0 \
    --data-parallel-size 4 \
    --tokenizer_mode mistral \
    --load_format mistral \
    --config_format mistral \
    --tool-call-parser mistral \
    --enable-auto-tool-choice"
