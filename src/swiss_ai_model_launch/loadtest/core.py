from __future__ import annotations

import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

import yaml

from .models import LoadtestConfig, ServerConfig

_BUILTIN_SCENARIOS_DIR = files("swiss_ai_model_launch.assets").joinpath("scenarios")
_CUSTOM_SCENARIOS_DIR = Path.cwd() / "scenarios"


def _load_scenario_definition_file(path: Traversable | Path, ext: str) -> dict[str, Any]:
    text = path.read_text()
    data = yaml.safe_load(text) if ext in (".yaml", ".yml") else json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must contain a top-level object: {path}")
    return data


def _load_scenario_definition(name: str) -> dict[str, Any] | None:
    """Load a scenario definition from custom or built-in scenario files.

    Supports .yaml, .yml, and .json; returns a plain dict ready for JSON serialization.
    """
    for ext in (".yaml", ".yml", ".json"):
        custom_path = _CUSTOM_SCENARIOS_DIR / f"{name}{ext}"
        if custom_path.exists():
            return _load_scenario_definition_file(custom_path, ext)

    for ext in (".yaml", ".yml", ".json"):
        builtin_path = _BUILTIN_SCENARIOS_DIR.joinpath(f"{name}{ext}")
        if builtin_path.is_file():
            return _load_scenario_definition_file(builtin_path, ext)

    return None


def build_run_config(server: ServerConfig, bench: LoadtestConfig) -> dict[str, Any]:
    return {
        "server_url": server.url,
        "api_key": server.api_key,
        "model": server.model,
        "scenario": bench.scenario,
        "scenario_definition": _load_scenario_definition(bench.scenario),
        "think_time": bench.think_time,
        "max_tokens": bench.max_tokens,
        "request_timeout": bench.request_timeout,
        "prompt_labels": bench.prompt_labels,
        "ignore_eos": bench.ignore_eos,
        "prompt_seed": bench.prompt_seed,
        "realistic": None,
        "custom": None,
    }
