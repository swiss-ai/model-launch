# Debug layer on top of the published vllm_apertus_1.5 image.
#
# Instruments BaseProcessingInfo.get_default_tok_params /
# default_tok_params (vllm/multimodal/processing/context.py) with
# [TOKDEBUG] prints (caller stack, pid, resulting add_special_tokens) to
# establish whether that path is exercised at all on the 0.19-line
# renderer rework -- for both `vllm serve` request handling and the
# offline LLM.chat()/generate() API. Groundwork for porting the BOS fix
# (see images/vllm_apertus_1.5_bosfix on branch vllm-apertus-1.5-bosfix):
# on this version apertus.py no longer overrides get_default_tok_params,
# so the base implementation is the only candidate call site.
#
# The patch is the working-tree diff of the local checkout at
# /capstor/store/cscs/swissai/infra01/users/rosmith/vllm (branch
# apertus_integration, commit 40a5516b). The pinned base was built from
# the older 09eae2122 (main's Dockerfile pin), but context.py is
# byte-identical between the two commits and neither has an apertus.py
# get_default_tok_params override, so the patch and any conclusions
# drawn from it transfer. The rev-parse guard below fails the build
# early if the base moves to yet another vLLM commit.
#
# The base image installs /workspace/vllm as an editable package, so
# patching the source in place is sufficient -- no reinstall needed.
# Base pinned to the :latest multi-arch index as of 2026-07-09 (same
# digest the bosfix image pins).
FROM ghcr.io/swiss-ai/vllm_apertus_1.5@sha256:44bbfc13ccb68ca8b06d0d2f85888a31eb56ad804d3d710383712fd3df4d431a

COPY tokdebug.patch /tmp/

RUN set -eux; \
    base_commit=$(git -C /workspace/vllm rev-parse HEAD); \
    test "${base_commit}" = "09eae2122ed71aa27765bbbf8c043009bfb86477"; \
    git -C /workspace/vllm apply --verbose /tmp/tokdebug.patch; \
    grep -q TOKDEBUG /workspace/vllm/vllm/multimodal/processing/context.py; \
    rm /tmp/tokdebug.patch
