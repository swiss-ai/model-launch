"""The --authorization policy: its grammar, how `private` is resolved before
launch, the label it puts on the mesh, and the served-name conflict guard."""

import argparse
import asyncio

import pytest

from swiss_ai_model_launch.cli import authorization as cli_authorization
from swiss_ai_model_launch.cli.authorization import (
    requested_from_args,
    resolve_authorization,
    warn_or_refuse_conflict,
)
from swiss_ai_model_launch.cli.main import _build_parser, build_launch_args_from_advanced
from swiss_ai_model_launch.launchers.authorization import (
    describe,
    is_private,
    is_public,
    normalize_authorization,
    policy_of,
)
from swiss_ai_model_launch.launchers.framework import _opentela_labels
from swiss_ai_model_launch.launchers.launch_args import LaunchArgs
from swiss_ai_model_launch.serving_api import ServingApiError


def _run(coro):
    return asyncio.run(coro)


# ── grammar ─────────────────────────────────────────────────────────────────


def test_public_and_private_are_canonicalized():
    assert normalize_authorization("public") == "public"
    assert normalize_authorization("  PUBLIC ") == "public"
    assert normalize_authorization("Private") == "private"


def test_email_list_is_lowercased_stripped_and_deduped():
    assert normalize_authorization(" User1@EPFL.ch , user2@ethz.ch ") == "user1@epfl.ch,user2@ethz.ch"
    assert normalize_authorization("a@x.ch,a@X.ch") == "a@x.ch"


def test_non_email_values_are_rejected():
    for bad in ("", "   ", "alice", "alice@", "@epfl.ch", "a@epfl.ch b@ethz.ch", ","):
        with pytest.raises(ValueError):
            normalize_authorization(bad)


def test_mixing_a_keyword_into_a_list_is_rejected():
    """'public,a@epfl.ch' is ambiguous — the gateway would read it as an email
    list containing a bogus address, silently restricting a model the user
    thought was public."""
    with pytest.raises(ValueError, match="mixes"):
        normalize_authorization("public,a@epfl.ch")


def test_policy_ignores_order_case_and_spacing():
    """Conflict detection compares policies, not strings — the same rule the
    gateway applies, so a reordered relaunch is not a conflict."""
    assert policy_of("a@x.ch, B@Y.ch") == policy_of("b@y.ch,a@x.ch")
    assert policy_of("public") is None
    assert policy_of("") is None
    assert policy_of("a@x.ch") != policy_of("a@x.ch,b@y.ch")


def test_predicates_and_description():
    assert is_public("") and is_public("Public")
    assert is_private("PRIVATE")
    assert not is_public("a@epfl.ch")
    assert "public" in describe("public")
    assert "a@epfl.ch" in describe("a@epfl.ch")


# ── LaunchArgs guardrail + label emission ───────────────────────────────────


def _launch_args(**overrides) -> LaunchArgs:
    base = dict(
        job_name="job",
        served_model_name="alice/swiss-ai/Apertus-70B",
        account="proj01",
        partition="normal",
        environment="/env.toml",
        framework="sglang",
        time="02:00:00",
    )
    base.update(overrides)
    return LaunchArgs(**base)


def test_launch_args_default_is_public():
    assert _launch_args().authorization == "public"


def test_launch_args_refuses_an_unresolved_private():
    """The mesh grammar has no 'private'; emitting it literally would publish
    the model to everyone. Any path that forgets to resolve must fail loudly."""
    with pytest.raises(ValueError, match="must be resolved"):
        _launch_args(authorization="private")


def test_launch_args_normalizes_the_label():
    assert _launch_args(authorization=" A@EPFL.ch , b@ethz.ch ").authorization == "a@epfl.ch,b@ethz.ch"


def test_labels_carry_the_authorization_policy():
    labels = _opentela_labels(_launch_args(authorization="a@epfl.ch,b@ethz.ch"))
    # Commas and @ need no shell quoting, so the label lands as-is — one
    # argument, which is what OpenTela's --label expects.
    assert "--label authorization=a@epfl.ch,b@ethz.ch" in labels


def test_public_launch_still_emits_the_label():
    """An explicit 'public' label and a missing one mean the same thing to the
    gateway, but emitting it keeps the policy visible in the DNT."""
    assert "authorization=public" in _opentela_labels(_launch_args())


# ── CLI plumbing ────────────────────────────────────────────────────────────


def _advanced_args(*extra: str):
    return _build_parser().parse_args(
        [
            "advanced",
            "--partition",
            "normal",
            "--framework",
            "sglang",
            "--environment",
            "/path/to/env.toml",
            "--framework-args",
            "--served-model-name swiss-ai/Apertus-70B",
            *extra,
        ]
    )


def test_advanced_defaults_to_public():
    args = _advanced_args()
    assert args.authorization == "public"
    la = build_launch_args_from_advanced(
        args,
        username="alice",
        account="proj01",
        partition="normal",
        authorization=_run(resolve_authorization(requested_from_args(args), "sk-rc-key")),
    )
    assert la.authorization == "public"


def test_advanced_accepts_an_email_list():
    args = _advanced_args("--authorization", "User1@EPFL.ch,user2@ethz.ch")
    resolved = _run(resolve_authorization(requested_from_args(args), "sk-rc-key"))
    assert resolved == "user1@epfl.ch,user2@ethz.ch"


def test_private_is_resolved_to_the_launchers_email(monkeypatch):
    """The gateway can't know who launched a job, so 'private' has to become
    the launcher's own email before submission."""
    seen = {}

    async def fake_whoami(api_key):
        seen["api_key"] = api_key
        return "Alice@EPFL.ch"

    monkeypatch.setattr(cli_authorization, "whoami", fake_whoami)
    args = _advanced_args("--authorization", "private")

    assert _run(resolve_authorization(requested_from_args(args), "sk-rc-key")) == "alice@epfl.ch"
    assert seen["api_key"] == "sk-rc-key"


def test_private_fails_the_launch_when_whoami_is_unavailable(monkeypatch):
    """Falling back to public here would be the exact opposite of what the
    user asked for, so the launch stops instead."""

    async def broken_whoami(api_key):
        raise ServingApiError("boom")

    monkeypatch.setattr(cli_authorization, "whoami", broken_whoami)
    args = _advanced_args("--authorization", "private")

    with pytest.raises(SystemExit, match="boom"):
        _run(resolve_authorization(requested_from_args(args), "sk-rc-key"))


def test_an_invalid_policy_exits_with_a_message_not_a_traceback():
    args = _advanced_args("--authorization", "not-an-email")
    with pytest.raises(SystemExit, match="not an email address"):
        _run(resolve_authorization(requested_from_args(args), "sk-rc-key"))


def test_env_var_supplies_the_default(monkeypatch):
    monkeypatch.setenv("SML_AUTHORIZATION", "a@epfl.ch")
    # The advanced flag reads the environment when the parser is built.
    args = _build_parser().parse_args(
        [
            "advanced",
            "--partition",
            "normal",
            "--framework",
            "sglang",
            "--environment",
            "/env.toml",
            "--framework-args",
            "--served-model-name m",
        ]
    )
    assert args.authorization == "a@epfl.ch"


def test_guided_flow_asks_for_the_policy():
    """`sml preconfigured` collects it through the same interactive chain as
    replicas/time — so it is a visible choice, not a silent default — while
    still accepting --authorization / SML_AUTHORIZATION non-interactively."""
    args = _build_parser().parse_args(["preconfigured", "--partition", "normal", "--authorization", "a@epfl.ch"])
    assert args.authorization == "a@epfl.ch"


# ── served-name conflict guard ──────────────────────────────────────────────


def _patch_existing(monkeypatch, values, error=None):
    async def fake(api_key, served_model_name):
        if error is not None:
            raise error
        return values

    monkeypatch.setattr(cli_authorization, "served_model_authorizations", fake)


def test_conflicting_policy_on_the_same_name_is_refused(monkeypatch):
    """OpenTela load-balances a name across every peer serving it, so a
    second launch with a different policy makes the gateway refuse the name
    for everyone — including the model that is already running."""
    _patch_existing(monkeypatch, ["a@epfl.ch"])
    with pytest.raises(SystemExit, match="different authorization policy"):
        _run(warn_or_refuse_conflict("sk-rc-key", "alice/swiss-ai/Apertus-70B", "public"))


def test_same_policy_written_differently_is_not_a_conflict(monkeypatch):
    _patch_existing(monkeypatch, ["b@ethz.ch,a@epfl.ch"])
    _run(warn_or_refuse_conflict("sk-rc-key", "alice/swiss-ai/Apertus-70B", "A@EPFL.ch,b@ethz.ch"))


def test_an_unlabeled_running_peer_reads_as_public(monkeypatch):
    _patch_existing(monkeypatch, [""])
    _run(warn_or_refuse_conflict("sk-rc-key", "alice/swiss-ai/Apertus-70B", "public"))


def test_no_running_peers_is_not_a_conflict(monkeypatch):
    _patch_existing(monkeypatch, [])
    _run(warn_or_refuse_conflict("sk-rc-key", "alice/swiss-ai/Apertus-70B", "a@epfl.ch"))


def test_an_unreachable_serving_api_warns_but_does_not_block(monkeypatch, capsys):
    """The gateway is the authority; this check is a courtesy. A launch must
    not be blocked because the API was briefly unreachable."""
    _patch_existing(monkeypatch, None, error=ServingApiError("network down"))
    _run(warn_or_refuse_conflict("sk-rc-key", "alice/swiss-ai/Apertus-70B", "a@epfl.ch"))
    assert "could not check for an authorization conflict" in capsys.readouterr().err


def test_namespace_makes_cross_user_collisions_impossible():
    """A sanity check on the interaction with namespacing: two users cannot
    collide on a served name at all, so the conflict guard only ever fires on
    one user relaunching their own name with a new policy."""
    from swiss_ai_model_launch.launchers.served_name import namespace_served_model_name

    alice = namespace_served_model_name("swiss-ai/Apertus-70B", "alice")
    bob = namespace_served_model_name("swiss-ai/Apertus-70B", "bob")
    assert alice != bob


def test_loadtest_advanced_applies_the_policy_too(monkeypatch, tmp_path):
    """`sml loadtest advanced` submits a real launch through the same advanced
    arguments, so a policy passed there must reach the LaunchArgs rather than
    being silently dropped."""
    from swiss_ai_model_launch.cli import loadtest as loadtest_module

    seen = {}

    def fake_build(args, **kwargs):
        seen.update(kwargs)
        raise SystemExit("stop-after-build")

    class _Config:
        @classmethod
        def exists(cls):
            return True

        @classmethod
        def load(cls):
            return cls()

        def get_non_none_value(self, name):
            return "sk-rc-key"

        def get_value(self, name):
            return None

    async def fake_create_launcher(config, args, non_interactive=False):
        class _L:
            username = "alice"
            account = "proj01"
            partition = "normal"

        return _L()

    monkeypatch.setenv("SML_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(loadtest_module, "InitConfig", _Config)
    args = _build_parser().parse_args(
        [
            "loadtest",
            "advanced",
            "--framework",
            "sglang",
            "--environment",
            "/path/to/env.toml",
            "--authorization",
            "a@epfl.ch",
        ]
    )

    with pytest.raises(SystemExit, match="stop-after-build"):
        _run(
            loadtest_module._run_loadtest_advanced(
                args,
                create_launcher=fake_create_launcher,
                build_launch_args_from_advanced=fake_build,
            )
        )
    assert seen["authorization"] == "a@epfl.ch"


def test_argparse_namespace_without_the_flag_defaults_to_public():
    """Callers that build a Namespace by hand (tests, example scripts) must
    not accidentally launch with no policy at all."""
    assert _run(resolve_authorization(requested_from_args(argparse.Namespace()), "sk-rc-key")) == "public"
