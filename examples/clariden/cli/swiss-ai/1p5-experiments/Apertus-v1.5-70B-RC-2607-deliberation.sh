#!/bin/bash
sml advanced \
  --tui \
  --partition normal \
  --framework vllm \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/ahadinia_models/apertus-ai/Apertus-v1.5-70B-RC \
    --served-model-name apertus-ai/Apertus-v1.5-70B-RC-deliberation \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --skip-mm-profiling \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --tool-parser-plugin /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_tool_parser.py \
    --reasoning-parser apertus \
    --reasoning-parser-plugin /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_reasoning_parser.py \
    --default-chat-template-kwargs.enable_thinking true"
