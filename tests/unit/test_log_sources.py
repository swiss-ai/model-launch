from swiss_ai_model_launch.cli.main import _log_sources


def _labels(sources: list[tuple[str, str, str]]) -> list[str]:
    return [label for label, _, _ in sources]


def test_single_replica_no_router() -> None:
    assert _labels(_log_sources(1, router="OPENTELA")) == ["Master", "Replica 0"]


def test_multi_replica_with_router() -> None:
    sources = _log_sources(3, router="SGL")
    assert _labels(sources) == ["Master", "Replica 0", "Replica 1", "Replica 2", "Router"]
    files = {label: (out, err) for label, out, err in sources}
    assert files["Master"] == ("log.out", "log.err")
    assert files["Replica 2"] == ("replica_2.out", "replica_2.err")
    assert files["Router"] == ("router.out", "router.err")


def test_multi_replica_no_router() -> None:
    assert "Router" not in _labels(_log_sources(2, router="OPENTELA"))


def test_router_requires_multiple_replicas() -> None:
    # A router is only meaningful in front of >1 replica.
    assert "Router" not in _labels(_log_sources(1, router="SGL"))
