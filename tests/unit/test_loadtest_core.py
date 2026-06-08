import json
from pathlib import Path

import pytest

import swiss_ai_model_launch.loadtest.core as core_module
import swiss_ai_model_launch.loadtest.models as models_module
from swiss_ai_model_launch.loadtest.core import build_run_config
from swiss_ai_model_launch.loadtest.models import LoadtestConfig, ServerConfig, load_scenarios
from swiss_ai_model_launch.loadtest.setup import DEFAULT_CLUSTER_PROMPTS_FILE, resolve_prompts_file


def test_resolve_prompts_file_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_LOADTEST_PROMPTS_FILE", "/cluster/env-prompts.json")

    assert resolve_prompts_file("/cluster/explicit-prompts.json") == Path("/cluster/explicit-prompts.json")


def test_resolve_prompts_file_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_LOADTEST_PROMPTS_FILE", "/cluster/env-prompts.json")

    assert resolve_prompts_file() == Path("/cluster/env-prompts.json")


def test_resolve_prompts_file_falls_back_to_cluster_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SML_LOADTEST_PROMPTS_FILE", raising=False)

    assert resolve_prompts_file() == DEFAULT_CLUSTER_PROMPTS_FILE


def test_load_scenarios_loads_builtin_and_custom_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "throughput.yaml").write_text("name: throughput\nmax_tokens: '2048'\nthink_time: '2'\n")
    (custom_dir / "profile.json").write_text(json.dumps({"name": "profile", "max_tokens": "128"}))
    (custom_dir / "custom.yaml").write_text("name: custom\nmax_tokens: '1'\n")
    monkeypatch.setattr(models_module, "_BUILTIN_SCENARIOS_DIR", builtin_dir)
    monkeypatch.setattr(models_module, "_CUSTOM_SCENARIOS_DIR", custom_dir)

    scenarios = load_scenarios()

    assert [s.name for s in scenarios] == ["throughput", "profile"]
    assert scenarios[0].think_time == "2"
    assert scenarios[1].max_tokens == "128"


def test_load_scenarios_custom_file_overrides_builtin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "throughput.yaml").write_text("name: throughput\nmax_tokens: '2048'\n")
    (custom_dir / "throughput.yaml").write_text("name: throughput\nmax_tokens: '64'\nprompt_labels:\n  - short\n")
    monkeypatch.setattr(models_module, "_BUILTIN_SCENARIOS_DIR", builtin_dir)
    monkeypatch.setattr(models_module, "_CUSTOM_SCENARIOS_DIR", custom_dir)

    scenario = load_scenarios()[0]

    assert scenario.name == "throughput"
    assert scenario.max_tokens == "64"
    assert scenario.prompt_labels == ["short"]


def test_build_run_config_embeds_scenario_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "smoke.yaml").write_text(
        "name: smoke\nexecutor: constant-vus\nvus: 1\nduration: 10s\nmax_tokens: '16'\n"
    )
    monkeypatch.setattr(core_module, "_BUILTIN_SCENARIOS_DIR", builtin_dir)
    monkeypatch.setattr(core_module, "_CUSTOM_SCENARIOS_DIR", custom_dir)

    config = build_run_config(
        ServerConfig(url="https://api.example.test", api_key="secret", model="test-model", is_swissai=True),
        LoadtestConfig(
            scenario="smoke",
            think_time="0",
            max_tokens="16",
            prompt_labels=["short"],
            ignore_eos=True,
            prompt_seed=7,
        ),
    )

    assert config["server_url"] == "https://api.example.test"
    assert config["api_key"] == "secret"
    assert config["model"] == "test-model"
    assert config["scenario"] == "smoke"
    assert config["scenario_definition"]["executor"] == "constant-vus"
    assert config["scenario_definition"]["vus"] == 1
    assert config["think_time"] == "0"
    assert config["max_tokens"] == "16"
    assert config["prompt_labels"] == ["short"]
    assert config["ignore_eos"] is True
    assert config["prompt_seed"] == 7
    assert config["custom"] is None


def test_build_run_config_uses_custom_scenario_before_builtin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "smoke.yaml").write_text("name: smoke\nexecutor: constant-vus\nvus: 1\n")
    (custom_dir / "smoke.json").write_text(json.dumps({"name": "smoke", "executor": "constant-vus", "vus": 2}))
    monkeypatch.setattr(core_module, "_BUILTIN_SCENARIOS_DIR", builtin_dir)
    monkeypatch.setattr(core_module, "_CUSTOM_SCENARIOS_DIR", custom_dir)

    config = build_run_config(
        ServerConfig(url="https://api.example.test", api_key="secret", model="test-model", is_swissai=True),
        LoadtestConfig(scenario="smoke", think_time="0", max_tokens="16"),
    )

    assert config["scenario_definition"]["vus"] == 2


def test_build_run_config_rejects_non_object_scenario_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builtin_dir = tmp_path / "builtin"
    custom_dir = tmp_path / "custom"
    builtin_dir.mkdir()
    custom_dir.mkdir()
    (builtin_dir / "broken.yaml").write_text("- not\n- an\n- object\n")
    monkeypatch.setattr(core_module, "_BUILTIN_SCENARIOS_DIR", builtin_dir)
    monkeypatch.setattr(core_module, "_CUSTOM_SCENARIOS_DIR", custom_dir)

    with pytest.raises(ValueError, match="top-level object"):
        build_run_config(
            ServerConfig(url="https://api.example.test", api_key="secret", model="test-model", is_swissai=True),
            LoadtestConfig(scenario="broken", think_time="0", max_tokens="16"),
        )
