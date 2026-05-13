#!/bin/bash
# Launch rednote-hilab/dots.mocr (1.5B multilingual document-OCR VLM) on
# one Clariden GH200 node with SGLang, DP=4 TP=1 (4 independent replicas,
# one per GPU). Suitable for high-throughput batch OCR over many images.
#
# dots.mocr requires --trust-remote-code (custom DotsOCRForCausalLM
# architecture). SGLang ≥ 0.5.x has the model class registered.
#
# Model weights (downloaded separately):
#   /capstor/store/cscs/swissai/infra01/MLLM/OCR/rednote-hilab_dots.mocr/
#
sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-time 6:00:00 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/MLLM/OCR/rednote-hilab_dots.mocr \
    --host 0.0.0.0 \
    --served-model-name dots_mocr-$(whoami) \
    --dp-size 4 \
    --tp-size 1 \
    --trust-remote-code \
    --context-length 16384 \
    --mem-fraction-static 0.85 \
    --enable-metrics"
