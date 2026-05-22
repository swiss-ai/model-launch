import json
import urllib.error
import urllib.request

import pytest

from swiss_ai_model_launch.assets import replica_probe


def _table() -> dict[str, object]:
    # OpenTela registers the served model name on the mesh as a "model=<name>"
    # entry in each llm service's identity_group; "last_seen" is the peer's
    # heartbeat epoch.
    return {
        "/QmHead": {
            "id": "QmHead",
            "service": [{"name": "llm", "identity_group": ["model=swiss-ai/Apertus-70B-aWBl"]}],
            "last_seen": 1771764988,
        },
        "/QmFollower": {
            "id": "QmFollower",
            "service": [{"name": "llm", "identity_group": ["model=swiss-ai/Apertus-70B-aWBl"]}],
            "last_seen": 1771764990,
        },
        "/QmOther": {
            "id": "QmOther",
            "service": [{"name": "llm", "identity_group": ["model=other/model-zzzz"]}],
        },
        "/QmNoService": {"id": "QmNoService", "service": []},
        "/QmBadEntry": "not-a-dict",
    }


def test_peer_ids_for_model_matches_model_label() -> None:
    assert replica_probe.peer_ids_for_model(_table(), "swiss-ai/Apertus-70B-aWBl") == ["QmHead", "QmFollower"]


def test_matching_peers_carries_last_seen() -> None:
    peers = replica_probe.matching_peers(_table(), "swiss-ai/Apertus-70B-aWBl")
    assert peers == [
        {"peer_id": "QmHead", "last_seen": 1771764988},
        {"peer_id": "QmFollower", "last_seen": 1771764990},
    ]


def test_matching_peers_missing_last_seen_is_none() -> None:
    table: dict[str, object] = {"/QmA": {"id": "QmA", "service": [{"identity_group": ["model=m-aaaa"]}]}}
    assert replica_probe.matching_peers(table, "m-aaaa") == [{"peer_id": "QmA", "last_seen": None}]


def test_peer_ids_for_model_excludes_other_models() -> None:
    assert replica_probe.peer_ids_for_model(_table(), "nonexistent/model") == []


def test_peer_ids_for_model_accepts_served_model_name_label_fallback() -> None:
    table: dict[str, object] = {"/QmA": {"id": "QmA", "service": [{"identity_group": ["served_model_name=m-aaaa"]}]}}
    assert replica_probe.peer_ids_for_model(table, "m-aaaa") == ["QmA"]


def test_peer_ids_for_model_falls_back_to_key_without_id() -> None:
    table: dict[str, object] = {"/QmKeyOnly": {"service": [{"identity_group": ["model=m-bbbb"]}]}}
    assert replica_probe.peer_ids_for_model(table, "m-bbbb") == ["QmKeyOnly"]


def test_peer_ids_for_model_dedups_across_services() -> None:
    table: dict[str, object] = {
        "/Qm1": {
            "id": "Qm1",
            "service": [
                {"identity_group": ["model=m-cccc"]},
                {"identity_group": ["model=m-cccc"]},
            ],
        }
    }
    assert replica_probe.peer_ids_for_model(table, "m-cccc") == ["Qm1"]


class _CtxResp:
    def __init__(self, data: bytes = b"") -> None:
        self._data = data

    def __enter__(self) -> "_CtxResp":
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._data


def test_check_replica_healthy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _CtxResp())
    assert replica_probe.check_replica("http://x", "Qm", "m", "k", 1) == "HEALTHY"


def test_check_replica_http_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom(*_a: object, **_k: object) -> object:
        raise urllib.error.HTTPError("http://x", 503, "err", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert replica_probe.check_replica("http://x", "Qm", "m", "k", 1) == "NOT_RESPONDING"


def test_check_replica_transport_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom(*_a: object, **_k: object) -> object:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert replica_probe.check_replica("http://x", "Qm", "m", "k", 1) == "ERROR"


def test_run_probe_reports_each_replica(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    table = _table()

    def _fake_urlopen(request: urllib.request.Request, timeout: float | None = None) -> _CtxResp:
        if request.full_url.endswith("/v1/dnt/table"):
            return _CtxResp(json.dumps(table).encode("utf-8"))
        return _CtxResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    report = replica_probe.run_probe("http://x", "swiss-ai/Apertus-70B-aWBl", "k", 1)
    assert [r["peer_id"] for r in report["replicas"]] == ["QmHead", "QmFollower"]
    assert [r["health"] for r in report["replicas"]] == ["HEALTHY", "HEALTHY"]
    assert [r["last_seen"] for r in report["replicas"]] == [1771764988, 1771764990]
    assert isinstance(report["checked_at"], int)
    assert "table_error" not in report


def test_run_probe_table_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom(*_a: object, **_k: object) -> object:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    report = replica_probe.run_probe("http://x", "m", "k", 1)
    assert report["replicas"] == []
    assert "table_error" in report


def test_main_one_shot_emits_single_report(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("SML_REPLICA_INTERVAL", raising=False)  # default 0 -> one-shot
    monkeypatch.setattr(replica_probe, "run_probe", lambda *a, **k: {"served_model_name": "m", "replicas": []})
    assert replica_probe.main() == 0
    assert capsys.readouterr().out.count(replica_probe.REPORT_BEGIN) == 1


class _StopLoopError(Exception):
    pass


def test_main_loops_when_interval_set(monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("SML_REPLICA_INTERVAL", "1")
    monkeypatch.setattr(replica_probe, "run_probe", lambda *a, **k: {"served_model_name": "m", "replicas": []})

    sleeps = {"n": 0}

    def _fake_sleep(_seconds: float) -> None:
        sleeps["n"] += 1
        if sleeps["n"] >= 2:
            raise _StopLoopError

    monkeypatch.setattr(replica_probe.time, "sleep", _fake_sleep)
    with pytest.raises(_StopLoopError):
        replica_probe.main()
    # one report per loop iteration: 2 iterations before the second sleep aborts
    assert capsys.readouterr().out.count(replica_probe.REPORT_BEGIN) == 2
