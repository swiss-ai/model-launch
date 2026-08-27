#!/bin/bash
# Speculative decoding: Apertus-v1.5-70B target with Apertus-v1.5-8B as draft model.
# Lossless — outputs match the plain 70B; the draft only proposes tokens the 70B verifies.
# num-speculative-tokens is a tuning knob: raise it if acceptance rate is high
# (greedy/structured workloads), lower it if the 8B disagrees with the 70B often.
sml advanced \
  --tui \
  --partition normal \
  --framework vllm \
  --time 12:00:00 \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_apertus_1.5_release.toml \
  --framework-args "--model swiss-ai/Apertus-v1.5-70B \
    --served-model-name swiss-ai/Apertus-v1.5-70B \
    --chat-template-content-format string \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --enable-auto-tool-choice \
    --tool-call-parser apertus \
    --speculative-config.method draft_model \
    --speculative-config.model swiss-ai/Apertus-v1.5-8B \
    --speculative-config.num_speculative_tokens 3 \
    --speculative-config.draft_tensor_parallel_size 4 \
    --compilation-config.pass_config.fuse_allreduce_rms false"
