"""The example-recipe path extractor, plus coverage guards on the real examples."""

from pathlib import Path

from swiss_ai_model_launch.launchers.path_check import MODEL_MARKERS, TOKENIZER_MARKERS
from tests.example_paths import (
    CLARIDEN_EXAMPLES,
    REPO_ROOT,
    discover_references,
    extract_references,
    group_references,
)

_SCRIPT = Path("examples/clariden/cli/vendor/model.sh")


def _values(text: str) -> list[str]:
    return [reference.value for reference in extract_references(_SCRIPT, text)]


def test_extracts_both_framework_spellings_of_the_model_flag() -> None:
    # sglang takes --model-path, vllm --model; --model-path must not be read as
    # --model with a value of "-path".
    sglang = 'sml advanced --framework-args "--model-path /store/vendor/model --tp-size 4"'
    vllm = 'sml advanced --framework-args "--model /store/vendor/model --tensor-parallel-size 4"'

    assert _values(sglang) == ["/store/vendor/model"]
    assert _values(vllm) == ["/store/vendor/model"]


def test_ignores_flags_that_merely_start_the_same_way() -> None:
    text = 'sml advanced --framework-args "--served-model-name vendor/served --tokenizer-mode mistral"'

    assert _values(text) == []


def test_follows_line_continuations_and_skips_comments() -> None:
    text = (
        "#!/bin/bash\n"
        "# --model /commented/out\n"
        "sml advanced \\\n"
        '  --framework-args "--model /store/vendor/model \\\n'
        '    --host 0.0.0.0"\n'
    )

    assert _values(text) == ["/store/vendor/model"]


def test_expands_literal_shell_variables() -> None:
    text = 'MODEL="/store/vendor/model"\nsml advanced --framework-args "--model-path $MODEL --tp-size 4"'
    braced = 'export MODEL=/store/vendor/model\nsml advanced --framework-args "--model-path ${MODEL}"'

    assert _values(text) == ["/store/vendor/model"]
    assert _values(braced) == ["/store/vendor/model"]


def test_marks_substitutions_it_cannot_expand_as_unresolved() -> None:
    text = 'sml advanced --framework-args "--model $HOME/models/mine"'

    (reference,) = extract_references(_SCRIPT, text)

    assert reference.unresolved


def test_classifies_repo_ids_apart_from_absolute_paths() -> None:
    text = 'sml advanced --framework-args "--model vendor/model --tokenizer /store/tokenizers/vendor/"'

    model, tokenizer = extract_references(_SCRIPT, text)

    assert (model.kind, model.markers) == ("repo-id", MODEL_MARKERS)
    assert model.resolve(Path("/registry")) == "/registry/vendor/model"
    # A trailing slash would otherwise make one directory look like two paths.
    assert (tokenizer.kind, tokenizer.markers, tokenizer.value) == (
        "absolute",
        TOKENIZER_MARKERS,
        "/store/tokenizers/vendor",
    )
    assert tokenizer.resolve(Path("/registry")) == "/store/tokenizers/vendor"


def test_grouping_collapses_repeated_paths_and_keeps_every_label() -> None:
    shared = 'sml advanced --framework-args "--model /store/vendor/model"'
    other = Path("examples/clariden/cli/vendor/model-vllm.sh")

    groups = group_references(extract_references(_SCRIPT, shared) + extract_references(other, shared))

    assert len(groups) == 1
    assert groups[0].labels == (f"{other} (--model)", f"{_SCRIPT} (--model)")


def test_every_clariden_example_contributes_a_checked_path() -> None:
    # A script the extractor can't read would silently drop out of CI coverage.
    scripts = {path.relative_to(REPO_ROOT) for path in CLARIDEN_EXAMPLES.rglob("*.sh")}
    covered = {reference.script for reference in discover_references()}

    assert scripts, "no example scripts found"
    assert not scripts - covered, f"no model path extracted from: {sorted(scripts - covered)}"


def test_no_example_path_is_left_unresolved() -> None:
    # An unexpandable substitution means CI cannot check that recipe's weights;
    # spell the path out (or assign it literally) so it stays covered.
    unresolved = [reference.label for reference in discover_references() if reference.unresolved]

    assert not unresolved, f"paths CI cannot resolve: {unresolved}"
