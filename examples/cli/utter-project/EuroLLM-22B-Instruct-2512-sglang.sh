#!/bin/bash
sml advanced \
  --slurm-nodes 1 \
  --serving-framework sglang \
  --slurm-environment src/swiss_ai_model_launch/assets/envs/sglang.toml \
  --framework-args "--model /capstor/store/cscs/swissai/infra01/hf_models/models/utter-project/EuroLLM-22B-Instruct-2512 \
    --served-model-name utter-project/EuroLLM-22B-Instruct-2512-$(whoami) \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 8080"
