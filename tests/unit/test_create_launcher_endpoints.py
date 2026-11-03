"""`sml` picks the FirecREST instance by system: clusters.yml first, the init config otherwise."""

import argparse
import importlib
from pathlib import Path
from typing import Any

import pytest

# `swiss_ai_model_launch.cli.main` is shadowed by the `main` function the package exports.
main = importlib.import_module("swiss_ai_model_launch.cli.main")

_INIT_CONFIG_CLIENT = object()


class _FirecrestInitConfig:
    def get_non_none_value(self, name: str) -> str:
        assert name == "launcher"
        return "firecrest"

    def get_value(self, name: str) -> str:
        raise KeyError(name)


@pytest.fixture
def used_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    (tmp_path / "clusters.yml").write_text(
        "systems:\n"
        "  beverin:\n"
        "    firecrest_url: https://api.svc.cscs.ch/beverin/firecrest/v2\n"
        "    client_id_env: BEVERIN_CLIENT_ID\n"
        "    client_secret_env: BEVERIN_CLIENT_SECRET\n"
    )
    monkeypatch.setenv("SML_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("BEVERIN_CLIENT_ID", "id")
    monkeypatch.setenv("BEVERIN_CLIENT_SECRET", "secret")
    monkeypatch.delenv("SML_SYSTEM", raising=False)
    monkeypatch.setattr(main, "_get_firecrest_client_from_init_config", lambda config: _INIT_CONFIG_CLIENT)

    used: list[Any] = []

    async def record(client: Any, **_: Any) -> Any:
        used.append(client)
        return object()

    monkeypatch.setattr(main, "_get_firecrest_launcher_with_client", record)
    return used


async def _launch(system: str | None) -> None:
    await main._create_launcher(_FirecrestInitConfig(), argparse.Namespace(system=system))  # type: ignore[arg-type]


async def test_listed_system_is_reached_through_its_own_instance(used_client: list[Any]) -> None:
    await _launch("beverin")
    (client,) = used_client
    assert client._firecrest_url == "https://api.svc.cscs.ch/beverin/firecrest/v2"


async def test_sml_system_env_var_selects_the_endpoint_too(
    used_client: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SML_SYSTEM", "beverin")
    await _launch(None)
    (client,) = used_client
    assert client._firecrest_url == "https://api.svc.cscs.ch/beverin/firecrest/v2"


@pytest.mark.parametrize("system", ["clariden", None])
async def test_other_systems_go_through_the_init_config(used_client: list[Any], system: str | None) -> None:
    await _launch(system)
    assert used_client == [_INIT_CONFIG_CLIENT]
