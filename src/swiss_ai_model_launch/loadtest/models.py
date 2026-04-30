from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import yaml


@dataclass
class ScenarioConfig:
    name: str
    max_tokens: str
    think_time: str | None = None
    prompt_labels: list[str] | None = None  # None = all labels (weighted mix)


_BUILTIN_SCENARIOS_DIR = files("swiss_ai_model_launch.assets").joinpath("scenarios")
_CUSTOM_SCENARIOS_DIR = Path.cwd() / "scenarios"


def _scenario_suffix(path: Traversable | Path) -> str:
    return Path(path.name).suffix


def _load_scenario_file(path: Traversable | Path) -> ScenarioConfig:
    text = path.read_text()
    data = yaml.safe_load(text) if _scenario_suffix(path) in (".yaml", ".yml") else json.loads(text)
    return ScenarioConfig(
        name=data["name"],
        max_tokens=data.get("max_tokens", "2048"),
        think_time=data.get("think_time"),
        prompt_labels=data.get("prompt_labels") or None,
    )


def load_scenarios() -> list[ScenarioConfig]:
    """Load built-in scenarios, with any user-defined ones from CWD/scenarios/ appended."""
    scenarios: dict[str, ScenarioConfig] = {}
    for path in sorted(_BUILTIN_SCENARIOS_DIR.iterdir(), key=lambda p: p.name):
        if _scenario_suffix(path) in (".yaml", ".yml", ".json"):
            s = _load_scenario_file(path)
            scenarios[s.name] = s
    if _CUSTOM_SCENARIOS_DIR.exists():
        for path in sorted(_CUSTOM_SCENARIOS_DIR.glob("*")):
            if _scenario_suffix(path) in (".yaml", ".yml", ".json"):
                s = _load_scenario_file(path)
                scenarios[s.name] = s
    # custom is always appended last as the catch-all
    scenarios.pop("custom", None)
    return list(scenarios.values())


@dataclass
class ServerConfig:
    url: str
    api_key: str
    chat_mode: bool
    model: str
    is_swissai: bool


@dataclass
class LoadtestConfig:
    scenario: str
    think_time: str
    max_tokens: str
    request_timeout: str | None = None
    prompt_labels: list[str] | None = None  # None = all labels (weighted mix)
