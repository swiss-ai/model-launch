from pathlib import Path

import keyring
import pytest
import yaml

from swiss_ai_model_launch.cli.configuration import init_wizard
from swiss_ai_model_launch.cli.configuration.init_wizard import InitConfig

_KEYRING_SERVICE = "swiss_ai_model_launch"

_LEGACY_CONFIG = {
    "name": "init_config",
    "type": "chain",
    "chain": [
        {
            "name": "launcher_configuration",
            "type": "branch",
            "head_configuration": {
                "name": "launcher",
                "type": "options",
                "value": "slurm",
                "options": {"slurm": ["SLURM Commands", "Already SSH'd into the cluster."]},
            },
            "branches": {"firecrest": None, "slurm": None},
        },
        {"name": "cscs_api_key", "type": "password", "value": "__keyring__"},
    ],
}


@pytest.fixture  # type: ignore[misc]
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "get_password", lambda service, name: store.get((service, name)))
    monkeypatch.setattr(keyring, "set_password", lambda service, name, value: store.__setitem__((service, name), value))
    return store


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: dict[str, object]) -> None:
    config_file = tmp_path / "config.yml"
    config_file.write_text(yaml.dump(data))
    monkeypatch.setattr(init_wizard, "_CONFIG_FILE", config_file)


def test_load_migrates_legacy_cscs_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    fake_keyring[(_KEYRING_SERVICE, "cscs_api_key")] = "secret"
    _write_config(tmp_path, monkeypatch, _LEGACY_CONFIG)

    config = InitConfig.load()

    assert config.get_non_none_value("swissai_research_api_key") == "secret"
    with pytest.raises(KeyError):
        config.get_value("cscs_api_key")


def test_load_keeps_existing_new_key_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    fake_keyring[(_KEYRING_SERVICE, "cscs_api_key")] = "stale"
    fake_keyring[(_KEYRING_SERVICE, "swissai_research_api_key")] = "current"
    _write_config(tmp_path, monkeypatch, _LEGACY_CONFIG)

    config = InitConfig.load()

    assert config.get_non_none_value("swissai_research_api_key") == "current"


def test_load_leaves_current_config_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    fake_keyring[(_KEYRING_SERVICE, "swissai_research_api_key")] = "secret"
    current = {
        **_LEGACY_CONFIG,
        "chain": [
            _LEGACY_CONFIG["chain"][0],  # type: ignore[index]
            {"name": "swissai_research_api_key", "type": "password", "value": "__keyring__"},
        ],
    }
    _write_config(tmp_path, monkeypatch, current)

    config = InitConfig.load()

    assert config.get_non_none_value("swissai_research_api_key") == "secret"
    assert (_KEYRING_SERVICE, "cscs_api_key") not in fake_keyring
