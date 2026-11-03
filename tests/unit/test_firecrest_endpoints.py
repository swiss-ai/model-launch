from pathlib import Path

import firecrest as f7t
import keyring
import pytest

from swiss_ai_model_launch.cli.configuration import models
from swiss_ai_model_launch.launchers.firecrest_endpoints import (
    BUILTIN_ENDPOINTS,
    KEYRING_SERVICE,
    EndpointConfigError,
    FirecrestEndpoint,
    SystemCredentials,
    client_for_system,
    load_endpoints,
    save_endpoint,
    stored_credentials,
)

_BEVERIN_URL = "https://api.svc.cscs.ch/beverin/firecrest/v2"
_ENV_ENTRY = FirecrestEndpoint(
    client_id_env="BEVERIN_CLIENT_ID",
    client_secret_env="BEVERIN_CLIENT_SECRET",  # noqa: S106 - a variable name, not a secret
)
_ENV = {"BEVERIN_CLIENT_ID": "id", "BEVERIN_CLIENT_SECRET": "secret"}
_STORED = SystemCredentials("stored-id", "stored-secret")


class _DefaultClientUsedError(Exception):
    pass


def _default_client() -> f7t.v2.AsyncFirecrest:
    raise _DefaultClientUsedError


def _nothing_stored(system: str) -> SystemCredentials | None:
    return None


def _stored(system: str) -> SystemCredentials | None:
    return _STORED


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "get_password", lambda service, name: store.get((service, name)))
    monkeypatch.setattr(keyring, "set_password", lambda service, name, value: store.__setitem__((service, name), value))
    return store


# Which client a system gets


def test_system_with_stored_credentials_uses_its_builtin_instance() -> None:
    client = client_for_system("beverin", _default_client, env={}, endpoints={}, lookup=_stored)
    assert client._firecrest_url == _BEVERIN_URL


@pytest.mark.parametrize("system", ["clariden", "beverin", "unknown", None])
def test_system_without_credentials_of_its_own_uses_the_init_config(system: str | None) -> None:
    with pytest.raises(_DefaultClientUsedError):
        client_for_system(system, _default_client, env={}, endpoints={}, lookup=_nothing_stored)


def test_env_credentials_named_in_clusters_yml_win_over_stored_ones() -> None:
    client = client_for_system("beverin", _default_client, env=_ENV, endpoints={"beverin": _ENV_ENTRY}, lookup=_stored)
    assert client._firecrest_url == _BEVERIN_URL


def test_clusters_yml_url_overrides_the_builtin_one() -> None:
    entry = FirecrestEndpoint(firecrest_url="https://f7t.test/v2")
    client = client_for_system("beverin", _default_client, env={}, endpoints={"beverin": entry}, lookup=_stored)
    assert client._firecrest_url == "https://f7t.test/v2"


def test_listed_system_without_credentials_says_how_to_add_them() -> None:
    with pytest.raises(EndpointConfigError, match=r"export BEVERIN_CLIENT_ID / .*sml init --system beverin"):
        client_for_system("beverin", _default_client, env={}, endpoints={"beverin": _ENV_ENTRY}, lookup=_nothing_stored)


def test_unknown_system_with_credentials_needs_a_url() -> None:
    with pytest.raises(EndpointConfigError, match="no FirecREST URL"):
        client_for_system("santis", _default_client, env={}, endpoints={}, lookup=_stored)


def test_api_key_alone_is_enough() -> None:
    entry = FirecrestEndpoint(firecrest_url="https://f7t-pat.api.svc.cscs.ch/mlp", api_key_env="KEY")
    client = client_for_system(
        "clariden", _default_client, env={"KEY": "k"}, endpoints={"clariden": entry}, lookup=_nothing_stored
    )
    assert client._firecrest_url == entry.firecrest_url


# Where credentials and URLs are kept


def test_stored_credentials_live_under_per_system_keys(fake_keyring: dict[tuple[str, str], str]) -> None:
    fake_keyring[(KEYRING_SERVICE, "firecrest_client_id@beverin")] = "id"
    fake_keyring[(KEYRING_SERVICE, "firecrest_client_secret@beverin")] = "secret"
    fake_keyring[(KEYRING_SERVICE, "firecrest_client_id")] = "init-id"

    assert stored_credentials("beverin") == SystemCredentials("id", "secret")
    assert stored_credentials("clariden") is None


def test_keyring_service_is_the_init_configs() -> None:
    assert KEYRING_SERVICE == models._KEYRING_SERVICE


def test_builtin_endpoints_cover_both_instances_seen_in_practice() -> None:
    assert BUILTIN_ENDPOINTS["clariden"] == "https://api.cscs.ch/ml/firecrest/v2"
    assert BUILTIN_ENDPOINTS["beverin"] == _BEVERIN_URL


# clusters.yml


def test_missing_file_means_no_extra_endpoints(tmp_path: Path) -> None:
    assert load_endpoints(tmp_path / "clusters.yml") == {}


def test_entries_round_trip_and_keep_the_others(tmp_path: Path) -> None:
    path = tmp_path / "clusters.yml"
    save_endpoint("beverin", _ENV_ENTRY, path)
    save_endpoint("santis", FirecrestEndpoint(firecrest_url="https://f7t.test/v2"), path)
    assert load_endpoints(path) == {
        "beverin": _ENV_ENTRY,
        "santis": FirecrestEndpoint(firecrest_url="https://f7t.test/v2"),
    }


@pytest.mark.parametrize(
    "content",
    ["- beverin\n", "systems: [beverin]\n", "systems:\n  beverin:\n    url: https://x\n"],
)
def test_malformed_files_are_rejected_with_their_path(tmp_path: Path, content: str) -> None:
    path = tmp_path / "clusters.yml"
    path.write_text(content)
    with pytest.raises(EndpointConfigError, match=str(path)):
        load_endpoints(path)
