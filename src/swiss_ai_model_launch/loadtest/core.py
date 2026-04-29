from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import LoadtestConfig, ServerConfig

_BUILTIN_SCENARIOS_DIR = Path(__file__).parent / "scenarios"
_CUSTOM_SCENARIOS_DIR = Path.cwd() / "scenarios"


def _load_scenario_definition(name: str) -> dict[str, Any] | None:
    """Load a scenario definition from custom or built-in scenario files.

    Supports .yaml, .yml, and .json; returns a plain dict ready for JSON serialization.
    """
    for base_dir in (_CUSTOM_SCENARIOS_DIR, _BUILTIN_SCENARIOS_DIR):
        for ext in (".yaml", ".yml", ".json"):
            path = base_dir / f"{name}{ext}"
            if not path.exists():
                continue

            text = path.read_text()
            data = yaml.safe_load(text) if ext in (".yaml", ".yml") else json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"Scenario file must contain a top-level object: {path}")
            return data
    return None


def build_run_config(server: ServerConfig, bench: LoadtestConfig) -> dict[str, Any]:
    return {
        "server_url": server.url,
        "api_key": server.api_key,
        "chat_mode": server.chat_mode,
        "model": server.model,
        "scenario": bench.scenario,
        "scenario_definition": _load_scenario_definition(bench.scenario),
        "think_time": bench.think_time,
        "max_tokens": bench.max_tokens,
        "request_timeout": bench.request_timeout,
        "prompt_labels": bench.prompt_labels,
        "realistic": None,
        "custom": None,
    }
