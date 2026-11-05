#!/bin/bash
# JUPITER (JSC): run from the repo root on a JUPITER login node with `sml init` set to the slurm launcher and
#   export SML_ACCOUNT=<budget account> SML_PARTITION=booster
# Compute nodes have no internet, so the weights are read from disk: download them first (see docs/jupiter.md).
# --model points at the directory holding the Hugging Face snapshot; change it for another model.
sml advanced \
  --partition "${SML_PARTITION:-booster}" \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_jupiter.toml \
  --container-runtime apptainer \
  --sbatch-arg=--gres=gpu:4 \
  --disable-opentela \
  --disable-metrics \
  --time 00:30:00 \
  --framework-args "--model /e/home/jusers/$USER/jupiter/sml/models/Qwen/Qwen3-0.6B \
    --served-model-name Qwen/Qwen3-0.6B \
    --max-model-len 4096"
