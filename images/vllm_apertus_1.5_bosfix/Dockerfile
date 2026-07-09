# Hotfix layer on top of the published vllm_apertus_1.5 image: ports
# vllm-project/vllm#39842 (skip add_special_tokens when the chat template
# already inserts BOS) to ApertusProcessingInfo. The base image installs
# /workspace/vllm as an editable package, so patching the source in place
# is sufficient — no reinstall needed.
# Base pinned to the :latest multi-arch index as of 2026-07-09.
FROM ghcr.io/swiss-ai/vllm_apertus_1.5@sha256:44bbfc13ccb68ca8b06d0d2f85888a31eb56ad804d3d710383712fd3df4d431a

COPY apertus_bos.patch /tmp/apertus_bos.patch

RUN git -C /workspace/vllm apply --verbose /tmp/apertus_bos.patch && \
    python -m py_compile /workspace/vllm/vllm/model_executor/models/apertus.py && \
    rm /tmp/apertus_bos.patch
