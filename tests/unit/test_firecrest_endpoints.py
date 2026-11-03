from pathlib import Path

import firecrest as f7t
import pytest

from swiss_ai_model_launch.launchers.firecrest_endpoints import (
    EndpointConfigError,
    FirecrestEndpoint,
    client_for_system,
    load_endpoints,
)

_BEVERIN = FirecrestEndpoint(
    firecrest_url="https://api.svc.cscs.ch/beverin/firecrest/v2",
    client_id_env="BEVERIN_CLIENT_ID",
    client_secret_env="BEVERIN_CLIENT_SECRET",  # noqa: S106 - a variable name, not a secret
)
_CREDS = {"BEVERIN_CLIENT_ID": "id", "BEVERIN_CLIENT_SECRET": "secret"}


class _DefaultClientUsedError(Exception):
    pass


def _default_client() -> f7t.v2.AsyncFirecrest:
    raise _DefaultClientUsedError


def test_missing_file_means_no_extra_endpoints(tmp_path: Path) -> None:
    assert load_endpoints(tmp_path / "clusters.yml") == {}


def test_endpoints_are_read_per_system(tmp_path: Path) -> None:
    path = tmp_path / "clusters.yml"
    path.write_text(
        "systems:\n"
        "  beverin:\n"
        "    firecrest_url: https://api.svc.cscs.ch/beverin/firecrest/v2\n"
        "    client_id_env: BEVERIN_CLIENT_ID\n"
        "    client_secret_env: BEVERIN_CLIENT_SECRET\n"
    )
    assert load_endpoints(path) == {"beverin": _BEVERIN}


@pytest.mark.parametrize(
    "content",
    ["- beverin\n", "systems: [beverin]\n", "systems:\n  beverin:\n    url: https://x\n"],
)
def test_malformed_files_are_rejected_with_their_path(tmp_path: Path, content: str) -> None:
    path = tmp_path / "clusters.yml"
    path.write_text(content)
    with pytest.raises(EndpointConfigError, match=str(path)):
        load_endpoints(path)


def test_listed_system_gets_its_own_instance() -> None:
    client = client_for_system("beverin", _default_client, env=_CREDS, endpoints={"beverin": _BEVERIN})
    assert client._firecrest_url == _BEVERIN.firecrest_url


@pytest.mark.parametrize("system", ["clariden", None])
def test_other_systems_use_the_init_config_as_before(system: str | None) -> None:
    with pytest.raises(_DefaultClientUsedError):
        client_for_system(system, _default_client, env=_CREDS, endpoints={"beverin": _BEVERIN})


def test_listed_system_without_credentials_names_the_variables() -> None:
    with pytest.raises(EndpointConfigError, match="export BEVERIN_CLIENT_ID or BEVERIN_CLIENT_SECRET"):
        client_for_system("beverin", _default_client, env={}, endpoints={"beverin": _BEVERIN})


def test_api_key_alone_is_enough() -> None:
    endpoint = FirecrestEndpoint(firecrest_url="https://f7t-pat.api.svc.cscs.ch/mlp", api_key_env="KEY")
    client = client_for_system("clariden", _default_client, env={"KEY": "k"}, endpoints={"clariden": endpoint})
    assert client._firecrest_url == endpoint.firecrest_url
