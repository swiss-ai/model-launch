import httpx

# Public Serving API gateway. Shared by the health checker (inference probes go
# through the same gateway users hit) and the whoami lookup that resolves
# `--authorization private` to the launcher's own email before submission.
SERVING_API_BASE_URL = "https://api.swissai.svc.cscs.ch"

_WHOAMI_URL = f"{SERVING_API_BASE_URL}/v1/whoami"
_TIMEOUT_SECONDS = 10


class ServingApiError(RuntimeError):
    """A Serving API call failed in a way the user has to act on (bad key, no network)."""


async def whoami(api_key: str) -> str:
    """Resolve the email that owns ``api_key`` via the Serving API whoami endpoint."""
    if not api_key:
        raise ServingApiError(
            "No CSCS API key is configured, so SML cannot look up your email. "
            "Run `sml init` to set one (get a key at https://serving.swissai.svc.cscs.ch)."
        )
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                _WHOAMI_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_TIMEOUT_SECONDS,
            )
    except (httpx.TransportError, httpx.TimeoutException) as e:
        raise ServingApiError(
            f"Could not reach the Serving API at {_WHOAMI_URL} ({e}). Check your network connection and retry."
        ) from e
    if response.status_code != 200:
        raise ServingApiError(
            f"The Serving API rejected the CSCS API key (HTTP {response.status_code}). "
            "Run `sml init` to update it (get a key at https://serving.swissai.svc.cscs.ch)."
        )
    try:
        email = response.json().get("email")
    except ValueError:
        email = None
    if not email or not isinstance(email, str):
        raise ServingApiError(f"The Serving API whoami response at {_WHOAMI_URL} did not contain an email.")
    return email
