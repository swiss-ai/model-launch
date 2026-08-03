import pytest

from swiss_ai_model_launch.launchers.authorization import (
    AUTH_PRIVATE,
    AUTH_PUBLIC,
    default_authorization,
    is_valid_authorization,
    normalize_email_list,
    parse_authorization,
    resolve_authorization,
)
from swiss_ai_model_launch.serving_api import ServingApiError, whoami


def test_parse_public_and_private_pass_through() -> None:
    assert parse_authorization(AUTH_PUBLIC) == AUTH_PUBLIC
    assert parse_authorization(AUTH_PRIVATE) == AUTH_PRIVATE


def test_parse_keywords_are_case_and_whitespace_insensitive() -> None:
    assert parse_authorization(" Public ") == AUTH_PUBLIC
    assert parse_authorization("PRIVATE") == AUTH_PRIVATE


def test_parse_normalizes_email_list() -> None:
    assert parse_authorization(" User1@EPFL.ch , user2@ethz.ch ") == "user1@epfl.ch,user2@ethz.ch"


def test_normalize_email_list_dedupes_preserving_order() -> None:
    assert normalize_email_list("b@x.ch,a@x.ch,B@X.CH,a@x.ch") == "b@x.ch,a@x.ch"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not-an-email",
        "user@",
        "@epfl.ch",
        "user@host",  # no TLD dot
        "a@b.ch,,c@d.ch",  # empty entry
        "a@b.ch,nonsense",
        "two words@epfl.ch",
    ],
)
def test_invalid_values_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="authorization"):
        parse_authorization(raw)
    assert not is_valid_authorization(raw)


def test_is_valid_authorization_accepts_grammar() -> None:
    for raw in (AUTH_PUBLIC, AUTH_PRIVATE, "user@epfl.ch", "a@b.ch, C@D.ch"):
        assert is_valid_authorization(raw)


def test_default_authorization_is_private(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SML_AUTHORIZATION", raising=False)
    assert default_authorization() == AUTH_PRIVATE


def test_default_authorization_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SML_AUTHORIZATION", AUTH_PUBLIC)
    assert default_authorization() == AUTH_PUBLIC


def test_mcp_tool_defers_to_the_shared_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MCP launch tool must not freeze 'private' into its signature: its
    default is None so the shared env-aware default chain runs at call time
    (a long-running server tracks SML_AUTHORIZATION like the CLI does)."""
    import inspect

    from swiss_ai_model_launch.mcp.server import launch_preconfigured_model

    sig = inspect.signature(launch_preconfigured_model)
    assert sig.parameters["authorization"].default is None


async def test_resolve_private_uses_whoami_email(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_whoami(api_key: str) -> str:
        assert api_key == "sk-rc-test"
        # The backend also compares case-insensitively, but the label should
        # already carry the canonical (stripped, lowercased) form.
        return " User@EPFL.ch "

    monkeypatch.setattr("swiss_ai_model_launch.launchers.authorization.whoami", fake_whoami)
    assert await resolve_authorization(AUTH_PRIVATE, "sk-rc-test") == "user@epfl.ch"


async def test_resolve_public_and_emails_skip_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_whoami(api_key: str) -> str:
        raise AssertionError("whoami must only run for 'private'")

    monkeypatch.setattr("swiss_ai_model_launch.launchers.authorization.whoami", fail_whoami)
    assert await resolve_authorization(AUTH_PUBLIC, "") == AUTH_PUBLIC
    assert await resolve_authorization("User@epfl.ch, other@ethz.ch", "") == "user@epfl.ch,other@ethz.ch"


async def test_resolve_propagates_whoami_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_whoami(api_key: str) -> str:
        raise ServingApiError("whoami is down")

    monkeypatch.setattr("swiss_ai_model_launch.launchers.authorization.whoami", fake_whoami)
    with pytest.raises(ServingApiError, match="whoami is down"):
        await resolve_authorization(AUTH_PRIVATE, "sk-rc-test")


async def test_whoami_without_key_fails_before_the_network() -> None:
    # Missing key must abort with a `sml init` hint instead of a confusing 401.
    with pytest.raises(ServingApiError, match="sml init"):
        await whoami("")
