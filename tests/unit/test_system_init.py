from pathlib import Path

import keyring
import pytest

from swiss_ai_model_launch.cli.configuration.system_init import register_system
from swiss_ai_model_launch.launchers.firecrest_endpoints import (
    KEYRING_SERVICE,
    FirecrestEndpoint,
    SystemCredentials,
    load_endpoints,
    stored_credentials,
)


@pytest.fixture
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "get_password", lambda service, name: store.get((service, name)))
    monkeypatch.setattr(keyring, "set_password", lambda service, name, value: store.__setitem__((service, name), value))
    return store


@pytest.fixture(autouse=True)
def answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_FIRECREST_CLIENT_ID", "beverin-id")
    monkeypatch.setenv("SML_FIRECREST_CLIENT_SECRET", "beverin-secret")


async def test_credentials_are_stored_next_to_the_init_ones(
    fake_keyring: dict[tuple[str, str], str], tmp_path: Path
) -> None:
    fake_keyring[(KEYRING_SERVICE, "firecrest_client_id")] = "clariden-id"

    url = await register_system("beverin", endpoints_path=tmp_path / "clusters.yml")

    assert url == "https://api.svc.cscs.ch/beverin/firecrest/v2"
    assert stored_credentials("beverin") == SystemCredentials("beverin-id", "beverin-secret")
    assert fake_keyring[(KEYRING_SERVICE, "firecrest_client_id")] == "clariden-id"


async def test_known_system_needs_no_clusters_yml(fake_keyring: dict[tuple[str, str], str], tmp_path: Path) -> None:
    await register_system("beverin", endpoints_path=tmp_path / "clusters.yml")
    assert not (tmp_path / "clusters.yml").exists()


async def test_custom_url_is_remembered_in_clusters_yml(
    fake_keyring: dict[tuple[str, str], str], tmp_path: Path
) -> None:
    path = tmp_path / "clusters.yml"
    await register_system("santis", firecrest_url="https://f7t.test/v2", endpoints_path=path)
    assert load_endpoints(path) == {"santis": FirecrestEndpoint(firecrest_url="https://f7t.test/v2")}


async def test_unknown_system_without_url_is_rejected_before_asking_for_secrets(
    fake_keyring: dict[tuple[str, str], str], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="pass --firecrest-url"):
        await register_system("santis", endpoints_path=tmp_path / "clusters.yml")
    assert fake_keyring == {}
