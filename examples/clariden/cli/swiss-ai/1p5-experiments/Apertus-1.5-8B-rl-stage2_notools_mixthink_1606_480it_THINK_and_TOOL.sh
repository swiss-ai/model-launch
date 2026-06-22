#!/bin/bash



# THINK-TOOL - swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it-THINK-TOOL
sml advanced \
  --system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 12:00:00 \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/models/rleval/rl_1p5-8b-stage2_notools_mixthink_1606_480it \
    --served-model-name swiss-ai/Apertus-1.5-8B-rl-stage2_notools_mixthink_1606_480it-THINK-TOOL \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --trust-request-chat-template \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.6 \
    --skip-mm-profiling \
    --reasoning-parser apertus \
    --reasoning-parser-plugin /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_reasoning_parser.py \
    --enable-auto-tool-choice \
    --tool-parser-plugin /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_tool_parser.py  \
    --tool-call-parser apertus \
    --chat-template /capstor/store/cscs/swissai/infra01/tool-parser-vllm/apertus_chat_template.jinja \
    --default-chat-template-kwargs.enable_thinking true"

# apertus_chat_template.jinja = model dir template + tool_call.arguments dict/string fix (no prime).
# Required for multi-turn tool calls (model dir template throws "str + dict" on tool-call replay).
# TODO: drop --chat-template once the fix lands upstream (swiss-ai/apertus-omni-tokenizer PR #3).