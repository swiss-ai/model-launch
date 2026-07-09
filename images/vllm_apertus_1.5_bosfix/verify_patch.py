"""Assert that apertus_bos.patch actually landed on ApertusProcessingInfo.

Run at image build time. `git apply` succeeding only proves the context matched;
it does not prove the override is reachable, nor that the multimodal
re-tokenization path stopped re-adding special tokens. Check both on the AST.
"""

import ast
import sys

PATH = "/workspace/vllm/vllm/model_executor/models/apertus.py"

tree = ast.parse(open(PATH).read())
classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def fail(msg: str) -> None:
    sys.exit(f"verify_patch: {msg} -- apertus_bos.patch did not take effect")


# 1. ApertusProcessingInfo.get_default_tok_params exists and forces the flag off.
info = classes.get("ApertusProcessingInfo") or fail("ApertusProcessingInfo not found")
override = next(
    (n for n in info.body if isinstance(n, ast.FunctionDef) and n.name == "get_default_tok_params"),
    None,
)
if override is None:
    fail("ApertusProcessingInfo.get_default_tok_params not defined")

kwargs = [
    kw
    for call in ast.walk(override)
    if isinstance(call, ast.Call)
    for kw in call.keywords
    if kw.arg == "add_special_tokens"
]
if not kwargs:
    fail("get_default_tok_params never passes add_special_tokens")
if not all(isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in kwargs):
    fail("get_default_tok_params does not force add_special_tokens=False")

# 2. The multimodal path no longer forwards inputs.tokenization_kwargs straight
#    into _tokenize_text; it must go through the locally-adjusted copy.
proc = classes.get("ApertusMultiModalProcessor") or fail("ApertusMultiModalProcessor not found")
apply_fn = next(
    (n for n in proc.body if isinstance(n, ast.FunctionDef) and n.name == "apply"),
    None,
)
if apply_fn is None:
    fail("ApertusMultiModalProcessor.apply not found")

for call in ast.walk(apply_fn):
    if not (isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "_tokenize_text"):
        continue
    for arg in call.args:
        if isinstance(arg, ast.Attribute) and arg.attr == "tokenization_kwargs":
            fail("_tokenize_text still receives inputs.tokenization_kwargs unmodified")

print("verify_patch: apertus.py BOS overrides present and reachable OK")
