#!/bin/bash
# Loadtest Apertus-8B on Clariden (NVIDIA / sglang) under two parallelism configs:
#   1) TP=4
#   2) TP=1, DP=4
# Uses `sml loadtest advanced` which fuses launch + wait-healthy + k6 + cancel
# into a single command. Each experiment runs sequentially (no parallel queueing).
set -euo pipefail

MODEL="/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509"
ENV="src/swiss_ai_model_launch/assets/envs/sglang.toml"
TIME="04:00:00"
SUFFIX="$(whoami)-$(date +%Y%m%d-%H%M%S)"
export SML_RESERVATION=SD-69241-apertus-1-5

# --- Experiment 1: TP=4 ---
sml loadtest advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model-path $MODEL \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-tp4-${SUFFIX} \
    --host 0.0.0.0 --port 8080 \
    --tp-size 4 \
    --enable-metrics" \
  --loadtest-scenario throughput \
  --cancel-after-loadtest

# --- Experiment 2: TP=1, DP=4 ---
sml loadtest advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model-path $MODEL \
    --served-model-name swiss-ai/Apertus-8B-Instruct-2509-tp1-dp4-${SUFFIX} \
    --host 0.0.0.0 --port 8080 \
    --tp-size 1 --dp-size 4 \
    --enable-metrics" \
  --loadtest-scenario throughput \
  --cancel-after-loadtest
