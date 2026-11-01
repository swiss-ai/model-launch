"""SML's default audience is the launcher alone — also for launches built in
Python, which never pass through the CLI that resolves `--authorization`.

A Launcher resolves the default "private" to the launcher's email itself,
before it renders anything, and refuses to launch when it can't: the
literal never reaches the mesh, and a launch never falls back to public.
"""

import asyncio
import re

import pytest

from swiss_ai_model_launch.launchers import FirecRESTLauncher, LaunchRequest
from swiss_ai_model_launch.launchers import launcher as launcher_module
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.serving_api import ServingApiError


def _run(coro):
    return asyncio.run(coro)


class _RecordingClient:
    """Just enough FirecREST to launch: records every script submitted."""

    def __init__(self):
        self.scripts: list[str] = []

    async def mkdir(self, **kwargs):
        return None

    async def upload(self, **kwargs):
        return None

    async def submit(self, script_str, **kwargs):
        self.scripts.append(script_str)
        return {"jobId": 100 + len(self.scripts) - 1}

    async def job_info(self, *args, **kwargs):
        return []


def _launcher(client=None, **kwargs) -> FirecRESTLauncher:
    return FirecRESTLauncher(
        client=client or _RecordingClient(),
        system_name="clariden",
        username="alice",
        account="infra01",
        partition="normal",
        **kwargs,
    )


def _args(**overrides) -> LaunchArgs:
    base = dict(
        job_name="job",
        served_model_name="alice/swiss-ai/Apertus-8B",
        account="infra01",
        partition="normal",
        environment="/env.toml",
        framework="sglang",
        time="02:00:00",
    )
    return LaunchArgs(**{**base, **overrides})


@pytest.fixture
def whoami_calls(monkeypatch):
    """Stands in for the Serving API's /v1/whoami; records the keys asked."""
    calls: list[str] = []

    async def whoami(api_key):
        calls.append(api_key)
        return "Alice@EPFL.ch"

    monkeypatch.setattr(launcher_module, "whoami", whoami)
    return calls


def _label(script: str, key: str) -> str:
    match = re.search(rf"--label '?{key}=([^'\s\\]*)", script)
    assert match, f"no {key} label in the submitted script"
    return match.group(1)


def test_private_resolves_to_the_launchers_email(whoami_calls):
    args = _args()
    resolved = _run(_launcher(serving_api_key="sk-rc-alice").resolve_identity(args))

    assert resolved.authorization == "alice@epfl.ch"
    # The email is in hand, so the launch is also stamped as Alice's to manage.
    assert resolved.launched_by_email == "Alice@EPFL.ch"
    assert resolved.launch_id == args.launch_id
    assert whoami_calls == ["sk-rc-alice"]


def test_an_explicit_policy_is_left_alone_without_a_network_call(whoami_calls):
    for policy in ("public", "b@ethz.ch,a@epfl.ch"):
        args = _args(authorization=policy)
        assert _run(_launcher(serving_api_key="sk-rc-alice").resolve_identity(args)) is args
    assert whoami_calls == []


def test_a_known_owner_is_not_overwritten(whoami_calls):
    resolved = _run(_launcher(serving_api_key="k").resolve_identity(_args(launched_by_email="ops@epfl.ch")))
    assert resolved.authorization == "alice@epfl.ch"
    assert resolved.launched_by_email == "ops@epfl.ch"


def test_without_a_key_private_refuses_to_launch(whoami_calls):
    """Failing closed: no key means no identity, and neither public nor the
    literal "private" is an acceptable stand-in."""
    with pytest.raises(ValueError, match="needs your Swiss AI Research API key"):
        _run(_launcher().resolve_identity(_args()))
    assert whoami_calls == []


def test_an_unreachable_serving_api_refuses_to_launch(monkeypatch):
    async def down(_key):
        raise ServingApiError("Could not reach the Serving API")

    monkeypatch.setattr(launcher_module, "whoami", down)
    with pytest.raises(ValueError, match="needs your identity from the Serving API"):
        _run(_launcher(serving_api_key="k").resolve_identity(_args()))


def test_the_key_can_come_from_the_environment(monkeypatch, whoami_calls):
    monkeypatch.setenv(launcher_module.SERVING_API_KEY_ENV, "sk-rc-from-env")
    resolved = _run(_launcher().resolve_identity(_args()))
    assert resolved.authorization == "alice@epfl.ch"
    assert whoami_calls == ["sk-rc-from-env"]


def test_a_passed_key_wins_over_the_environment(monkeypatch, whoami_calls):
    monkeypatch.setenv(launcher_module.SERVING_API_KEY_ENV, "sk-rc-from-env")
    _run(_launcher(serving_api_key="sk-rc-passed").resolve_identity(_args()))
    assert whoami_calls == ["sk-rc-passed"]


def test_a_python_launch_with_no_policy_goes_out_private(whoami_calls):
    """End to end through launch_model: a request that says nothing about its
    audience is submitted labelled with the launcher's email alone."""
    client = _RecordingClient()
    request = LaunchRequest(
        model="swiss-ai/Apertus-8B-Instruct-2509",
        framework="sglang",
        nodes_per_replica=1,
        replicas=1,
        time="01:00:00",
    )
    _run(_launcher(client, serving_api_key="sk-rc-alice").launch_model(request))

    (script,) = client.scripts
    assert _label(script, "authorization") == "alice@epfl.ch"
    assert _label(script, "launched_by_email") == "Alice@EPFL.ch"


def test_launch_with_args_resolves_too(whoami_calls):
    client = _RecordingClient()
    _run(_launcher(client, serving_api_key="sk-rc-alice").launch_with_args(_args()))
    assert _label(client.scripts[0], "authorization") == "alice@epfl.ch"


def test_a_chain_resolves_once_and_every_job_carries_it(whoami_calls):
    client = _RecordingClient()
    _run(
        _launcher(client, serving_api_key="sk-rc-alice").launch_consecutive_with_args(
            _args(time="12:00:00"), total_time="30:00:00", handover_time="03:00:00"
        )
    )
    assert len(client.scripts) == 3
    assert {_label(s, "authorization") for s in client.scripts} == {"alice@epfl.ch"}
    assert whoami_calls == ["sk-rc-alice"]
