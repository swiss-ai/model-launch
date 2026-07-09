# Hotfix layer on top of the published vllm_apertus_1.5 image.
#
# Two Apertus-local BOS fixes, both confined to
# vllm/model_executor/models/apertus.py -- no vLLM core files are touched.
#
# 1. ApertusProcessingInfo.get_default_tok_params: skip add_special_tokens when
#    the tokenizer carries a chat template, since that template already renders
#    `{{ bos_token }}` and the tokenizer would prepend a second BOS. Base
#    checkpoints (no chat template) keep the default so raw prompts still get
#    their BOS. Straight port of vllm-project/vllm#39842, which applies the
#    same fix to Gemma 4; the pinned base predates that PR.
#
# 2. ApertusMultiModalProcessor.apply decoded an already-tokenized prompt back
#    to text and re-encoded it with the tokenizer's default
#    add_special_tokens=True, duplicating BOS on image/audio requests. It now
#    suppresses special tokens when the prompt arrived as token ids. This one
#    is Apertus-specific -- the stock BaseMultiModalProcessor avoids the
#    text round-trip -- and is not covered upstream.
#
# Scope note: (1) only reaches callers that read the model's default tokenize
# params, i.e. the offline LLM.chat()/LLM.generate() API. The OpenAI server
# takes add_special_tokens from the request (chat defaults False, completions
# defaults True), so /v1/completions with a pre-rendered chat prompt still
# double-BOSes; that must be fixed by sending "add_special_tokens": false.
#
# The base image installs /workspace/vllm as an editable package, so patching
# the source in place is sufficient -- no reinstall needed.
# Base pinned to the :latest multi-arch index as of 2026-07-09.
FROM ghcr.io/swiss-ai/vllm_apertus_1.5@sha256:44bbfc13ccb68ca8b06d0d2f85888a31eb56ad804d3d710383712fd3df4d431a

COPY apertus_bos.patch verify_patch.py /tmp/

# Verify against the parsed AST, not just py_compile: a syntax check would still
# pass if an upstream rename left the override unreachable, silently shipping an
# image that does nothing.
RUN git -C /workspace/vllm apply --verbose /tmp/apertus_bos.patch && \
    python /tmp/verify_patch.py && \
    rm /tmp/apertus_bos.patch /tmp/verify_patch.py
