#!/bin/bash
# Launch all Apertus 70B ROCm memory optimization experiments.
set -euo pipefail

MODEL="/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-70B-Instruct-2509"
ENV="src/swiss_ai_model_launch/assets/envs/sglang_rocm.toml"
TIME="12:00:00"

# baseline: mem-fraction 0.5
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-mem-fraction-05 \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --mem-fraction-static 0.5"

# delete checkpoint after loading
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-delete-ckpt \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --delete-ckpt-after-loading"

# disable mmap
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-disable-mmap \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --weight-loader-disable-mmap"

# disable mmap + mem-fraction 0.5
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-disable-mmap-mem-fraction-05 \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --weight-loader-disable-mmap \
    --mem-fraction-static 0.5"

# disable mmap + mem-fraction 0.7
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-disable-mmap-mem-fraction \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --weight-loader-disable-mmap \
    --mem-fraction-static 0.7"

# disable mmap + delete ckpt + mem-fraction 0.7
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-disable-mmap-delete-ckpt-mem-fraction \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --weight-loader-disable-mmap \
    --delete-ckpt-after-loading \
    --mem-fraction-static 0.7"

# memory saver
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --pre-launch-cmds "pip install torch-memory-saver" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-mem-saver \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --enable-memory-saver"

# all memory opts combined
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --pre-launch-cmds "pip install torch-memory-saver" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-all-mem-opts \
    --host 0.0.0.0 --port 8080 --tp-size 4 \
    --enable-memory-saver \
    --delete-ckpt-after-loading \
    --weight-loader-disable-mmap"

# 2 nodes, tp-size 8
sml advanced \
  --firecrest-system beverin \
  --partition mi300 \
  --slurm-nodes 2 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model $MODEL \
    --served-model-name swiss-ai/Apertus-70B-Instruct-2509-rocm-sglang-2nodes \
    --host 0.0.0.0 --port 8080 --tp-size 8 \
    --mem-fraction-static 0.5"
