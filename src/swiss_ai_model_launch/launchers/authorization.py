"""Launch-time access control for a served model.

Every model SML puts on the mesh carries an OpenTela peer label
``authorization`` that tells the Serving API gateway who may list and use it:

- ``public`` (the default) — anyone, including anonymous ``/v1/models`` callers.
- a comma-separated email list — only those users.

``private`` is a THIRD spelling that exists only in the CLI. The gateway has no
idea who launched a job, so "only me" cannot be expressed as a label; SML
resolves ``private`` to the launcher's own email (via the Serving API's
``/v1/whoami``) before the job is submitted, and the literal string never
reaches the mesh. :class:`~swiss_ai_model_launch.launchers.launch_args.LaunchArgs`
rejects it as a guardrail against a code path that forgets to resolve it.

Labels are self-asserted — anyone who can join the mesh can claim any label —
so this is access control at the gateway, not authentication on the mesh. See
serving-api ADR-0001.
"""

import re

PUBLIC = "public"
PRIVATE = "private"

# Deliberately loose: the gateway matches the label against an API key's
# owner_email verbatim, so our job is to catch typos and shell-quoting
# accidents ("--authorization user1@epfl.ch user2@ethz.ch" arriving as one
# arg), not to adjudicate what a valid address is.
_EMAIL_RE = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")


def normalize_authorization(value: str) -> str:
    """Canonicalize a ``--authorization`` value, or raise ValueError.

    ``public``/``private`` are returned lowercased and unchanged otherwise.
    An email list is stripped, lowercased, de-duplicated (order preserved)
    and rejoined with commas — matching what the gateway normalizes to, so
    two launches that mean the same policy emit the same label string and
    do not read as a served-name conflict.
    """
    text = (value or "").strip()
    if not text:
        raise ValueError("--authorization must not be empty; use 'public', 'private', or an email list.")

    lowered = text.lower()
    if lowered in (PUBLIC, PRIVATE):
        return lowered

    emails: list[str] = []
    for part in text.split(","):
        email = part.strip().lower()
        if not email:
            continue
        if email in (PUBLIC, PRIVATE):
            raise ValueError(
                f"--authorization {value!r} mixes {email!r} with an email list. "
                f"Use either '{PUBLIC}', '{PRIVATE}', or a comma-separated list of emails."
            )
        if not _EMAIL_RE.match(email):
            raise ValueError(
                f"--authorization {value!r} is not a valid access policy: {email!r} is not an email address. "
                f"Use '{PUBLIC}', '{PRIVATE}', or a comma-separated list like 'a@epfl.ch,b@ethz.ch'."
            )
        if email not in emails:
            emails.append(email)

    if not emails:
        raise ValueError(f"--authorization {value!r} lists no email addresses.")
    return ",".join(emails)


def is_private(value: str) -> bool:
    return (value or "").strip().lower() == PRIVATE


def is_public(value: str) -> bool:
    """Public is also how the gateway reads a missing or empty label, so
    pre-feature launches keep working."""
    text = (value or "").strip().lower()
    return not text or text == PUBLIC


def policy_of(value: str) -> frozenset[str] | None:
    """A label value as a comparable policy: None for public, else the email
    set. Two label strings that differ only in order, case or spacing give
    the same policy — which is what a conflict check must compare, so that
    re-launching with a reordered list is not treated as a conflict.

    Mirrors ``normalize_policy`` in the gateway's authorization service.
    """
    text = (value or "").strip()
    if not text or text.lower() == PUBLIC:
        return None
    return frozenset(part.strip().lower() for part in text.split(",") if part.strip())


def describe(value: str) -> str:
    """One-line human summary for launch output."""
    if is_public(value):
        return "public (anyone can list and use this model)"
    emails = sorted(policy_of(value) or ())
    return f"restricted to {len(emails)} user(s): {', '.join(emails)}"
