# Hotfix layer on top of the published vllm_apertus_1.5 image.
#
# Flips CompletionRequest.add_special_tokens to default False so that
# /v1/completions does not prepend a second BOS to prompts that already
# begin with one. Apertus' chat template renders `{{ bos_token }}` itself,
# so clients that apply the template and post the rendered string to
# /v1/completions (eval harnesses, OpenWebUI raw completion) would
# otherwise get a duplicated `<s>`.
#
# Note: this also means a genuine raw-text completion whose prompt does not
# already carry a BOS no longer gets one. Such callers must now pass
# "add_special_tokens": true explicitly.
#
# /v1/chat/completions is unaffected: ChatCompletionRequest already defaults
# add_special_tokens to False.
#
# The base image installs /workspace/vllm as an editable package, so patching
# the source in place is sufficient -- no reinstall needed.
# Base pinned to the :latest multi-arch index as of 2026-07-09.
FROM ghcr.io/swiss-ai/vllm_apertus_1.5@sha256:44bbfc13ccb68ca8b06d0d2f85888a31eb56ad804d3d710383712fd3df4d431a

COPY apertus_bos.patch verify_patch.py /tmp/

# Verify against the parsed AST, not just py_compile: a syntax check would still
# pass if an upstream rename left the field untouched, silently shipping an
# image that does nothing.
RUN git -C /workspace/vllm apply --verbose /tmp/apertus_bos.patch && \
    python /tmp/verify_patch.py && \
    rm /tmp/apertus_bos.patch /tmp/verify_patch.py
