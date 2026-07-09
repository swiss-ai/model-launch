"""Assert that apertus_bos.patch actually landed on CompletionRequest.

Run at image build time. Guards against the patch applying to a renamed or
restructured upstream field and leaving the served default untouched.
"""

import ast
import sys

PATH = "/workspace/vllm/vllm/entrypoints/openai/completion/protocol.py"

tree = ast.parse(open(PATH).read())

defaults = [
    kw.value.value
    for cls in tree.body
    if isinstance(cls, ast.ClassDef) and cls.name == "CompletionRequest"
    for stmt in cls.body
    if isinstance(stmt, ast.AnnAssign)
    and getattr(stmt.target, "id", None) == "add_special_tokens"
    and isinstance(stmt.value, ast.Call)
    for kw in stmt.value.keywords
    if kw.arg == "default"
]

if defaults != [False]:
    sys.exit(
        f"CompletionRequest.add_special_tokens default is {defaults!r}, "
        "expected [False] -- apertus_bos.patch did not take effect"
    )

print("verify_patch: CompletionRequest.add_special_tokens default=False OK")
