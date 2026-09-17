import argparse
import asyncio

from swiss_ai_model_launch.cli.main import _get_firecrest_launcher_with_client, _get_slurm_launcher


class _FakeFirecrestClient:
    async def systems(self):
        return [{"name": "clariden", "ssh": {"host": "clariden"}}]

    async def partitions(self, system_name):
        return [{"name": "highprio"}]

    async def userinfo(self, system_name):
        return {"user": {"name": "rosmith"}, "group": {"name": "infra01"}}


def _args(**overrides):
    defaults = dict(system="clariden", partition="highprio", account=None, reservation=None, qos=None)
    return argparse.Namespace(**{**defaults, **overrides})


def test_firecrest_launcher_qos_non_interactive_from_args():
    args = _args(qos="highprio")
    launcher = asyncio.run(_get_firecrest_launcher_with_client(_FakeFirecrestClient(), args=args, non_interactive=True))
    assert launcher.qos == "highprio"


def test_firecrest_launcher_qos_non_interactive_defaults_to_none():
    launcher = asyncio.run(
        _get_firecrest_launcher_with_client(_FakeFirecrestClient(), args=_args(), non_interactive=True)
    )
    assert launcher.qos is None


def test_firecrest_launcher_qos_interactive_resolves_from_args():
    # Passing --qos (and blanking reservation/account) still short-circuits
    # the interactive prompts even when non_interactive=False (args always
    # win over asking the user).
    args = _args(qos="highprio", reservation="", account="")
    launcher = asyncio.run(
        _get_firecrest_launcher_with_client(_FakeFirecrestClient(), args=args, non_interactive=False)
    )
    assert launcher.qos == "highprio"


def test_slurm_launcher_qos_non_interactive_from_args(monkeypatch):
    class FakeProc:
        async def communicate(self):
            return b"highprio\n", b""

    async def fake_exec(*argv, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    args = _args(qos="highprio")
    launcher = asyncio.run(_get_slurm_launcher(args=args, non_interactive=True))
    assert launcher.qos == "highprio"


def test_slurm_launcher_qos_interactive_resolves_from_args(monkeypatch):
    class FakeProc:
        async def communicate(self):
            return b"highprio\n", b""

    async def fake_exec(*argv, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    args = _args(qos="highprio", reservation="", account="infra01")
    launcher = asyncio.run(_get_slurm_launcher(args=args, non_interactive=False))
    assert launcher.qos == "highprio"
