"""Served-model-name namespacing.

Every model SML puts on the mesh is served under ``<username>/<vendor>/<model>``,
where ``username`` is the cluster account that submitted the SLURM job — the
same value the job advertises as the OpenTela ``launched_by`` label. The
gateway cross-checks the two (see the serving-api authorization service), so
the namespace must be the launcher's own username and nothing else.

Namespacing replaces the random salt that used to disambiguate served names:
two users launching the same model no longer collide, and a name is now
predictable from who launched what. Two launches by the SAME user of the same
model do share a name — deliberately, since they carry identical labels and
OpenTela simply load-balances across them as extra replicas.
"""

# A name with fewer than this many segments predates namespacing (bare
# "model" or "vendor/model"); the gateway skips its namespace check, and we
# prepend the username rather than reject it.
_NAMESPACED_SEGMENTS = 3


def is_namespaced(served_model_name: str) -> bool:
    """Is this name already of the ``user/vendor/model`` shape?"""
    return len(served_model_name.split("/")) >= _NAMESPACED_SEGMENTS


def namespace_of(served_model_name: str) -> str | None:
    """The username namespace of a served name, or None if it has none."""
    if not is_namespaced(served_model_name):
        return None
    return served_model_name.split("/", 1)[0]


def namespace_served_model_name(served_model_name: str, username: str) -> str:
    """Return ``served_model_name`` under ``username``'s namespace.

    An unnamespaced name ("Apertus-70B", "swiss-ai/Apertus-70B") gets the
    username prepended, so pre-namespacing launch scripts keep working as
    written. A name that is already namespaced is returned untouched when the
    namespace is the launcher's own, and rejected otherwise — publishing under
    someone else's username would make the served name disagree with the
    ``launched_by`` label the job emits, and the gateway refuses to route that.
    """
    username = username.strip()
    if not username:
        raise ValueError("Cannot namespace a served model name without a username.")
    if "/" in username:
        raise ValueError(f"Username must not contain '/': {username!r}")

    name = served_model_name.strip()
    if not name:
        raise ValueError("Served model name must not be empty.")

    namespace = namespace_of(name)
    if namespace is None:
        return f"{username}/{name}"
    if namespace != username:
        raise ValueError(
            f"--served-model-name {name!r} is namespaced under {namespace!r}, but this "
            f"job is submitted by {username!r}. Launch it as "
            f"{username}/{name.split('/', 1)[1]} or drop the namespace and let SML add it."
        )
    return name
