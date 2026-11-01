"""A `--dev` launch joins the dev mesh, which only the dev Serving API routes
to — so SML has to ask that gateway who you are, whether the name conflicts,
and whether the model is healthy, with a key it knows."""

import argparse
import asyncio
import importlib

import httpx

from swiss_ai_model_launch import serving_api
from swiss_ai_model_launch.cli.healthcheck import checker

# The module, not the `main` function the package re-exports under that name.
main_module = importlib.import_module("swiss_ai_model_launch.cli.main")


def _run(coro):
    return asyncio.run(coro)


def test_prod_is_the_default():
    assert serving_api.base_url() == "https://api.swissai.svc.cscs.ch"


def test_use_dev_gateway_switches_to_dev():
    serving_api.use_dev_gateway()
    assert serving_api.base_url() == "https://apidev.swissai.svc.cscs.ch"


def test_an_explicit_url_wins_over_dev(monkeypatch):
    monkeypatch.setenv(serving_api.BASE_URL_ENV, "http://localhost:8080/")
    serving_api.use_dev_gateway()
    assert serving_api.base_url() == "http://localhost:8080"


def _record_requests(monkeypatch, response):
    urls: list[str] = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, **kwargs):
            urls.append(url)
            return response

        async def post(self, url, **kwargs):
            urls.append(url)
            return response

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return urls


def test_identity_and_health_checks_follow_the_gateway(monkeypatch):
    urls = _record_requests(monkeypatch, httpx.Response(200, json={"email": "a@epfl.ch", "choices": []}))
    serving_api._reset_whoami_cache_for_tests()
    serving_api.use_dev_gateway()

    _run(serving_api.whoami("sk-rc-dev"))
    _run(checker.check_model_health("alice/org/m", "sk-rc-dev"))

    assert urls == [
        "https://apidev.swissai.svc.cscs.ch/v1/whoami",
        "https://apidev.swissai.svc.cscs.ch/v1/chat/completions",
    ]
    serving_api._reset_whoami_cache_for_tests()


def test_the_dev_flag_selects_the_dev_gateway(monkeypatch):
    async def launch(_args):
        return None

    monkeypatch.setattr(main_module, "_run_advanced", launch)
    _run(main_module._main(argparse.Namespace(subcommand="advanced", dev=True)))
    assert serving_api.base_url() == "https://apidev.swissai.svc.cscs.ch"


def test_without_the_dev_flag_prod_stays(monkeypatch):
    async def launch(_args):
        return None

    monkeypatch.setattr(main_module, "_run_advanced", launch)
    _run(main_module._main(argparse.Namespace(subcommand="advanced", dev=False)))
    assert serving_api.base_url() == "https://api.swissai.svc.cscs.ch"


def test_a_key_in_the_environment_beats_the_stored_one(monkeypatch):
    """The dev gateway has its own key database, so a dev launch needs a dev
    key — given for one command, without overwriting the stored prod key."""
    from swiss_ai_model_launch.cli.configuration import InitConfig

    monkeypatch.setattr(InitConfig, "get_non_none_value", lambda self, name: "sk-rc-stored")
    config = InitConfig()
    assert config.serving_api_key() == "sk-rc-stored"
    monkeypatch.setenv(serving_api.API_KEY_ENV, "sk-rc-dev")
    assert config.serving_api_key() == "sk-rc-dev"
