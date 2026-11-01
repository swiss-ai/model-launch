"""Thin client for the Serving API (the gateway in front of the mesh).

SML talks to it for two launch-time reasons:

- ``whoami`` resolves the user's Swiss AI Research API key to the email the
  gateway knows them by, which is what ``--authorization private`` becomes.
- ``served_model_access`` reports the access a served name has right now, so
  a launch that would collide with a *different* policy under the same name
  can be refused before submission rather than becoming an unroutable model
  (see the conflict rule in serving-api ADR-0001).
"""

from dataclasses import dataclass
from urllib.parse import quote

import httpx

BASE_URL = "https://api.swissai.svc.cscs.ch"
_TIMEOUT_SECONDS = 10


class ServingApiError(RuntimeError):
    """The Serving API could not answer — network failure, or a non-2xx."""


# One launch asks who you are for two reasons — resolving `--authorization
# private`, and stamping the launched_by_email label the gateway uses to
# decide who may change the model's access later. The answer is the same and
# does not change mid-run, so cache it rather than pay a second round-trip.
_whoami_cache: dict[str, str] = {}


def _reset_whoami_cache_for_tests() -> None:
    """Test helper — the cache is process-global and a CLI process only ever
    launches with one key, so nothing clears it in normal use."""
    _whoami_cache.clear()


async def whoami(api_key: str) -> str:
    """The email the gateway associates with ``api_key``.

    Raises ServingApiError on an invalid key (401) or an unreachable API, so
    the caller can fail the launch loudly: silently falling back to public
    would be the opposite of what `--authorization private` asked for.

    Only successful answers are cached; a failure stays retryable.
    """
    if api_key in _whoami_cache:
        return _whoami_cache[api_key]
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/v1/whoami",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_TIMEOUT_SECONDS,
            )
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        raise ServingApiError(f"Could not reach the Serving API at {BASE_URL}: {exc}") from exc

    if response.status_code == 401:
        raise ServingApiError(
            "The Serving API rejected your Swiss AI Research API key. "
            "Check it with `sml init` (get one at https://serving.swissai.svc.cscs.ch)."
        )
    if not response.is_success:
        raise ServingApiError(f"GET /v1/whoami returned HTTP {response.status_code}.")

    email = (response.json() or {}).get("email")
    if not email:
        raise ServingApiError("GET /v1/whoami returned no email address.")
    _whoami_cache[api_key] = str(email)
    return str(email)


@dataclass(frozen=True)
class ServedAccess:
    """Who may use a served name right now, as the gateway enforces it."""

    # The effective policy as a label value ("public" or an email list): the
    # post-launch override where there is one, else the launch label. None
    # when it cannot be compared — see served_model_access.
    policy: str | None
    # The policy was changed after launch from the web UI.
    overridden: bool
    # The launches serving the name already disagree, so the gateway is
    # refusing it for everyone.
    conflict: bool


async def served_model_access(api_key: str, served_model_name: str) -> ServedAccess | None:
    """The access ``served_model_name`` has right now, or None if nothing this
    key can see is serving it.

    Read from GET /v1/model-access rather than from the model listing: the
    listing publishes people as names, not addresses, and does not say whether
    access was changed after launch. This endpoint reports the EFFECTIVE policy
    — the same thing the gateway compares when it decides a name's launches
    disagree — and returns it as an email list to whoever may edit the model
    (its launcher or an admin). Anyone else gets names, which cannot be
    compared with an email list, so ``policy`` is None for them unless the
    model is public.

    Best effort by design: it only sees what the *caller* is allowed to see.
    The gateway is the authority — this exists to turn the common case
    (relaunching your own model under a different policy while the old job is
    still up) into an error at submission time instead of a model nobody can
    route to.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/v1/model-access/{quote(served_model_name, safe='/')}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_TIMEOUT_SECONDS,
            )
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        raise ServingApiError(f"Could not reach the Serving API at {BASE_URL}: {exc}") from exc

    if response.status_code == 404:
        return None
    if response.status_code == 401:
        raise ServingApiError("The Serving API rejected your Swiss AI Research API key.")
    if not response.is_success:
        raise ServingApiError(f"GET /v1/model-access returned HTTP {response.status_code}.")

    body = response.json() or {}
    conflict = bool(body.get("conflict"))
    effective = body.get("effective_authorization")
    comparable = (
        not conflict
        and isinstance(effective, str)
        and (bool(body.get("can_edit")) or effective.strip().lower() == "public")
    )
    return ServedAccess(
        policy=effective if comparable else None,
        overridden=bool(body.get("is_overridden")),
        conflict=conflict,
    )
