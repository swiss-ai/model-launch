"""Thin client for the Serving API (the gateway in front of the mesh).

SML talks to it for two launch-time reasons:

- ``whoami`` resolves the user's Swiss AI Research API key to the email the
  gateway knows them by, which is what ``--authorization private`` becomes.
- ``served_model_authorizations`` reports the ``authorization`` labels already
  on the mesh for a served name, so a launch that would collide with a
  *different* policy under the same name can be refused before submission
  rather than becoming an unroutable model (see the conflict rule in
  serving-api ADR-0001).
"""

import httpx

BASE_URL = "https://api.swissai.svc.cscs.ch"
_TIMEOUT_SECONDS = 10


class ServingApiError(RuntimeError):
    """The Serving API could not answer — network failure, or a non-2xx."""


async def whoami(api_key: str) -> str:
    """The email the gateway associates with ``api_key``.

    Raises ServingApiError on an invalid key (401) or an unreachable API, so
    the caller can fail the launch loudly: silently falling back to public
    would be the opposite of what `--authorization private` asked for.
    """
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
    return str(email)


async def served_model_authorizations(api_key: str, served_model_name: str) -> list[str]:
    """The ``authorization`` label of every peer currently serving that name.

    Best effort by design: it only sees what the *caller* is allowed to see,
    so a same-named model restricted to somebody else is invisible here. The
    gateway is the authority — this exists to turn the common case (relaunching
    your own model under a different policy while the old job is still up) into
    an error at submission time instead of a model nobody can route to.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/v1/models_detailed",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_TIMEOUT_SECONDS,
            )
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        raise ServingApiError(f"Could not reach the Serving API at {BASE_URL}: {exc}") from exc

    if not response.is_success:
        raise ServingApiError(f"GET /v1/models_detailed returned HTTP {response.status_code}.")

    entries = (response.json() or {}).get("data") or []
    return [
        str((entry.get("labels") or {}).get("authorization", ""))
        for entry in entries
        if entry.get("id") == served_model_name
    ]
