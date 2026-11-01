import pytest

from swiss_ai_model_launch import serving_api
from swiss_ai_model_launch.launchers import launcher as launcher_module


@pytest.fixture(autouse=True)
def _no_real_serving_api_key(monkeypatch):
    """A launch defaults to authorization="private", which a Launcher resolves
    with the user's Swiss AI Research API key — read from the environment or
    the config `sml init` wrote. Unit tests must never find the developer's
    real key there and call the real Serving API with it; a test that needs a
    key passes one (or sets the variable), and one that forgot fails with the
    missing-key error."""
    monkeypatch.delenv(launcher_module.SERVING_API_KEY_ENV, raising=False)
    monkeypatch.setattr(launcher_module, "_stored_serving_api_key", lambda: None)


@pytest.fixture(autouse=True)
def _prod_gateway(monkeypatch):
    """Which Serving API to talk to is process state (`--dev` flips it); no
    test may leave it pointing anywhere but the default."""
    monkeypatch.delenv(serving_api.BASE_URL_ENV, raising=False)
    serving_api._reset_gateway_for_tests()
    yield
    serving_api._reset_gateway_for_tests()
