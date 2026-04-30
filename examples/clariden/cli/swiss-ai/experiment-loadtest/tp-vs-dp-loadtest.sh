#!/bin/bash
# Loadtest Apertus-8B on Clariden (NVIDIA / sglang) under two parallelism configs:
#   1) TP=4
#   2) TP=1, DP=4
# Both jobs are submitted up-front so they queue/start in parallel; loadtests
# run sequentially as each model becomes healthy.
set -euo pipefail

: "${CSCS_SERVING_API:?export CSCS_SERVING_API}"

LOADTEST_SERVER_URL="https://api.swissai.svc.cscs.ch"
LOADTEST_PROMPTS_FILE="/capstor/store/cscs/swissai/infra01/loadtest/prompts.json"
LOADTEST_SCENARIO="throughput"

MODEL="/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509"
ENV="src/swiss_ai_model_launch/assets/envs/sglang.toml"
TIME="04:00:00"
SUFFIX="$(whoami)"
export SML_RESERVATION=SD-69241-apertus-1-5

SERVED_1="swiss-ai/Apertus-8B-Instruct-2509-tp4-${SUFFIX}"
SERVED_2="swiss-ai/Apertus-8B-Instruct-2509-tp1-dp4-${SUFFIX}"

# --- Submit both jobs up-front ---

# Job 1: TP=4
OUT_1=$(sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model-path $MODEL \
    --served-model-name $SERVED_1 \
    --host 0.0.0.0 --port 8080 \
    --tp-size 4 \
    --enable-metrics")
echo "$OUT_1"
JOB_1=$(echo "$OUT_1" | grep "Job submitted:" | awk '{print $3}')

# Job 2: TP=1, DP=4
OUT_2=$(sml advanced \
  --firecrest-system clariden \
  --partition normal \
  --slurm-nodes 1 \
  --slurm-time "$TIME" \
  --serving-framework sglang \
  --slurm-environment "$ENV" \
  --framework-args "--model-path $MODEL \
    --served-model-name $SERVED_2 \
    --host 0.0.0.0 --port 8080 \
    --tp-size 1 --dp-size 4 \
    --enable-metrics")
echo "$OUT_2"
JOB_2=$(echo "$OUT_2" | grep "Job submitted:" | awk '{print $3}')

# --- Experiment 1: TP=4 ---
echo "Waiting for $SERVED_1 to be healthy..."
until curl -fsS -H "Authorization: Bearer $CSCS_SERVING_API" "$LOADTEST_SERVER_URL/v1/models" | grep -q "$SERVED_1"; do
  sleep 30
done

sml loadtest run \
  --firecrest-system clariden \
  --partition normal \
  --loadtest-server-url "$LOADTEST_SERVER_URL" \
  --loadtest-api-key "$CSCS_SERVING_API" \
  --loadtest-model "$SERVED_1" \
  --loadtest-scenario "$LOADTEST_SCENARIO" \
  --loadtest-prompts-file "$LOADTEST_PROMPTS_FILE" \
  --no-wait-until-healthy

scancel "$JOB_1"

# --- Experiment 2: TP=1, DP=4 ---
echo "Waiting for $SERVED_2 to be healthy..."
until curl -fsS -H "Authorization: Bearer $CSCS_SERVING_API" "$LOADTEST_SERVER_URL/v1/models" | grep -q "$SERVED_2"; do
  sleep 30
done

sml loadtest run \
  --firecrest-system clariden \
  --partition normal \
  --loadtest-server-url "$LOADTEST_SERVER_URL" \
  --loadtest-api-key "$CSCS_SERVING_API" \
  --loadtest-model "$SERVED_2" \
  --loadtest-scenario "$LOADTEST_SCENARIO" \
  --loadtest-prompts-file "$LOADTEST_PROMPTS_FILE" \
  --no-wait-until-healthy

scancel "$JOB_2"
