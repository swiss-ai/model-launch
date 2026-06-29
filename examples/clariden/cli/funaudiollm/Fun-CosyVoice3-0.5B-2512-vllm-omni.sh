#!/bin/bash
# Launch Fun-CosyVoice3 0.5B with vLLM on one Clariden GH200 node.
#
# Model: FunAudioLLM/Fun-CosyVoice3-0.5B-2512
#
# Note:
#   CosyVoice3 requires ref_audio/ref_text for /v1/audio/speech.
#   If using local file:// reference audio, set:
#     export COSYVOICE_ALLOWED_LOCAL_MEDIA_PATH=/path/to/reference/audio/root

EXTRA_MEDIA_ARGS=""

if [ -n "${COSYVOICE_ALLOWED_LOCAL_MEDIA_PATH:-}" ]; then
  EXTRA_MEDIA_ARGS="--allowed-local-media-path ${COSYVOICE_ALLOWED_LOCAL_MEDIA_PATH}"
fi

sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time 6:00:00 \
  --serving-framework vllm \
  --served-model-name FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --worker-port 8080 \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/vllm_cuda13_v2.toml \
  --framework-args "FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --host 0.0.0.0 \
    --port 8080 \
    --deploy-config /opt/conda/lib/python3.12/site-packages/vllm_omni/deploy/cosyvoice3.yaml \
    --omni \
    --trust-remote-code \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.40 \
    ${EXTRA_MEDIA_ARGS}" \
  --no-tui
