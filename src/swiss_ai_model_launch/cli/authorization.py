"""CLI-side handling of ``--authorization``.

Kept out of ``cli/main.py`` so the loadtest subcommand — which submits real
launches through the same ``advanced`` arguments — can apply the identical
policy handling without importing main (which imports loadtest).
"""

import argparse
import sys

from swiss_ai_model_launch.launchers.authorization import (
    PRIVATE,
    PUBLIC,
    is_private,
    normalize_authorization,
    policy_of,
)
from swiss_ai_model_launch.serving_api import ServingApiError, served_model_access, whoami


def is_valid_authorization(raw: str) -> bool:
    """Predicate form of ``normalize_authorization``, for the interactive
    prompt's validator — it needs a bool, not an exception."""
    try:
        normalize_authorization(raw)
    except ValueError:
        return False
    return True


def requested_from_args(args: argparse.Namespace) -> str:
    """The raw ``--authorization`` value off a parsed namespace, defaulting to
    public — SML's default — for namespaces built without the flag."""
    return getattr(args, "authorization", None) or PUBLIC


async def resolve_authorization(requested: str | None, api_key: str) -> str:
    """A raw authorization value as it will be labelled on the mesh.

    ``private`` is turned into the launcher's own email here — the gateway has
    no way to know who submitted a job, so "only me" has to be spelled out
    before submission. Everything else is validated and canonicalized so that
    two launches meaning the same policy emit the same label string.

    Takes the raw string rather than the argparse namespace because the guided
    flow collects it through the interactive config chain, not as a flag.

    Raises SystemExit with a readable message rather than a traceback: this is
    the last thing between the user and a model with the wrong audience.
    """
    requested = requested or PUBLIC
    try:
        normalized = normalize_authorization(requested)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    if not is_private(normalized):
        return normalized

    try:
        email = await whoami(api_key)
    except ServingApiError as exc:
        raise SystemExit(
            f"error: authorization {PRIVATE!r} needs your identity from the Serving API, but: {exc}"
        ) from exc
    return normalize_authorization(email)


async def resolve_launcher_email(api_key: str) -> str:
    """The launcher's email, for the ``launched_by_email`` label — or "" if we
    could not find out.

    That label is what lets the Serving API decide, later, who is allowed to
    change this model's access from the web UI: the ``launched_by`` label
    next to it is a cluster shell account (``$USER``), which no platform
    identity can be compared against.

    Best effort ON PURPOSE, unlike the ``private`` resolution in
    ``resolve_authorization``. There the email IS the access policy, so not
    knowing it has to fail the launch. Here it only decides who can edit the
    policy afterwards, and refusing to launch a public model because the
    Serving API was briefly unreachable would be a much worse trade. The cost
    of an empty label is that the model becomes admin-managed.
    """
    try:
        return await whoami(api_key)
    except ServingApiError as exc:
        print(
            f"warning: could not resolve your identity for the launch owner label: {exc}\n"
            "         The model will launch, but its access will only be changeable by an admin.",
            file=sys.stderr,
        )
        return ""


async def warn_or_refuse_conflict(api_key: str, served_model_name: str, authorization: str) -> None:
    """Refuse a launch that would make ``served_model_name`` unroutable.

    OpenTela load-balances a served name across every peer advertising it, so
    the gateway cannot keep a request off a colliding launch's replicas — when
    two launches share a name with *different* policies it refuses the name for
    everyone (serving-api ADR-0001). Since names are namespaced by username,
    the realistic way to hit this is relaunching your own model under a new
    policy while the old job is still up; catching it here turns a confusing
    403-for-everyone into an error at submission time.

    The comparison is against the running model's CURRENT access, not the
    policy it was launched with: the gateway compares effective policies, so a
    model whose access was changed from the web UI conflicts with a relaunch
    that only matches its original --authorization.

    Best effort: the check needs the Serving API, and only sees models this key
    is allowed to see. If it can't ask, it says so and lets the launch proceed —
    the gateway remains the authority either way.
    """
    try:
        access = await served_model_access(api_key, served_model_name)
    except ServingApiError as exc:
        print(
            f"warning: could not check for an authorization conflict on {served_model_name!r}: {exc}",
            file=sys.stderr,
        )
        return

    if access is None:
        return

    if access.conflict:
        raise SystemExit(
            f"error: {served_model_name!r} is already being refused by the Serving API: the launches "
            "serving it disagree on who may use it.\n"
            "       Another launch under this name cannot fix that. Cancel the jobs that disagree first, "
            "or launch under a different --served-model-name."
        )

    if access.policy is None:
        print(
            f"warning: could not check for an authorization conflict on {served_model_name!r}: "
            "it is already running, and only its launcher or an admin can see its access policy.",
            file=sys.stderr,
        )
        return

    if policy_of(access.policy) == policy_of(authorization):
        return

    if access.overridden:
        match_hint = (
            "or match its current access (it was changed after launch in the web UI — "
            "you can also Reset it there to its launch-time policy)."
        )
    else:
        match_hint = "or match its --authorization."
    raise SystemExit(
        f"error: {served_model_name!r} is already served with a different authorization policy "
        f"({access.policy!r} vs {authorization!r}).\n"
        "       The Serving API refuses to route a name whose replicas disagree, so this launch "
        "would take the running one down with it.\n"
        "       Cancel the running job first, launch under a different --served-model-name, "
        f"{match_hint}"
    )
