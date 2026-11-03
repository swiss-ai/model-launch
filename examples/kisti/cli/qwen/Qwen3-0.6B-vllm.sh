#!/bin/bash
# KISTI-6: run from the repo root on a KISTI login node with `sml init` set to the slurm launcher and
#   export SML_ACCOUNT=<project> SML_WORKING_DIR=/scratch/$USER/sml   (jobs may not run from $HOME)
# KISTI rejects jobs without --comment, and only hands out GPUs requested via --gres.
# The pip uninstall works around torchcodec in vllm-openai:*-cu129 being built for CUDA 13.
sml advanced \
  --partition "${SML_PARTITION:-intTest}" \
  --framework vllm \
  --environment src/swiss_ai_model_launch/assets/envs/vllm_kisti.toml \
  --container-runtime pyxis \
  --sbatch-arg="--comment=field=nlp;appl=vllm" \
  --sbatch-arg=--gres=gpu:4 \
  --disable-opentela \
  --disable-metrics \
  --time 00:30:00 \
  --pre-launch-cmds "pip uninstall -y -q torchcodec || true" \
  --framework-args "--model Qwen/Qwen3-0.6B \
    --served-model-name Qwen/Qwen3-0.6B \
    --max-model-len 4096"
