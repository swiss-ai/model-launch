import os
import re

from swiss_ai_model_launch.serving_api import whoami

# Values of the OpenTela peer label "authorization", read by the Serving API to
# decide who may see and use a model. "public" (or an absent label — backward
# compat with pre-feature launches) means anyone; a comma-separated email list
# restricts it to those users. The literal "private" never reaches the mesh:
# SML resolves it to the launcher's own email (via the Serving API whoami
# endpoint) before submission.
AUTH_PUBLIC = "public"
AUTH_PRIVATE = "private"

# Deliberately simple: enough to catch typos (missing @, spaces, no domain)
# without re-implementing RFC 5322. The Serving API compares case-insensitively.
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def normalize_email_list(raw: str) -> str:
    """Normalise a comma-separated email list: strip, lowercase, dedupe preserving order.

    Raises ValueError when any entry is empty or not a syntactically valid email.
    """
    seen: list[str] = []
    for part in raw.split(","):
        email = part.strip().lower()
        if not _EMAIL_RE.fullmatch(email):
            raise ValueError(
                f"Invalid email address in authorization list: {part.strip()!r}. "
                f"Expected '{AUTH_PUBLIC}', '{AUTH_PRIVATE}', or comma-separated emails."
            )
        if email not in seen:
            seen.append(email)
    return ",".join(seen)


def default_authorization() -> str:
    """The default raw value when the user gives none: SML_AUTHORIZATION from
    the environment, else "private". Shared by every launch surface (advanced
    parser, wizard, MCP tool) so their defaults can't drift."""
    return os.environ.get("SML_AUTHORIZATION", AUTH_PRIVATE)


def parse_authorization(raw: str) -> str:
    """Validate and normalise a raw --authorization value, without resolving "private"."""
    value = raw.strip().lower()
    if value in (AUTH_PUBLIC, AUTH_PRIVATE):
        return value
    return normalize_email_list(raw)


def is_valid_authorization(raw: str) -> bool:
    """Predicate form of `parse_authorization`, for wizard/prompt validators."""
    try:
        parse_authorization(raw)
    except ValueError:
        return False
    return True


async def resolve_authorization(raw: str, cscs_api_key: str) -> str:
    """Resolve a raw --authorization value to the label emitted on the mesh.

    Only the literal "private" needs the network: it becomes the launcher's own
    email, looked up via the Serving API whoami endpoint with their CSCS API
    key. "public" and explicit email lists resolve locally.
    """
    parsed = parse_authorization(raw)
    if parsed == AUTH_PRIVATE:
        return (await whoami(cscs_api_key)).strip().lower()
    return parsed
