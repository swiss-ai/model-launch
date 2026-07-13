#!/bin/bash
sml advanced \
  --tui \
  --partition normal \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/ahadinia_models/apertus-ai/Apertus-v1.5-8B-RC-2607 \
    --served-model-name apertus-ai/Apertus-v1.5-8B-RC-2607-think \
    --skip-mm-profiling \
    --trust-remote-code \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --tool-parser-plugin /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_tool_parser.py \
    --reasoning-parser apertus \
    --reasoning-parser-plugin /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_reasoning_parser.py \
    --default-chat-template-kwargs.enable_thinking true"
