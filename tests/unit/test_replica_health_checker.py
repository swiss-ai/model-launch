import ast
import importlib.resources
import json
import subprocess
import urllib.error
import urllib.request

from swiss_ai_model_launch.assets import replica_health_checker as checker


def test_checker_source_is_python36_compatible() -> None:
    """The in-job checker runs under the batch node's own python3, which on some
    HPC hosts is as old as 3.6. Guard (via AST, so the warning docstring's prose
    doesn't trip it) against reintroducing 3.7+ constructs that silently kill the
    checker (a SyntaxError/TypeError → no report → CLI timeout).
    """
    source = importlib.resources.files("swiss_ai_model_launch.assets").joinpath("replica_health_checker.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # `from __future__ import annotations` is 3.7+ (the original breakage).
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            assert all(alias.name != "annotations" for alias in node.names)
        # `subprocess.run(capture_output=/text=)` are 3.7+ kwargs.
        if isinstance(node, ast.Call):
            kwargs = {kw.arg for kw in node.keywords}
            assert "capture_output" not in kwargs
            assert "text" not in kwargs
        # PEP 604 `X | Y` union annotations are 3.10+; use typing.Optional/Union.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            raise AssertionError("PEP 604 union (X | Y) is not 3.6-compatible; use typing.Optional/Union")


class _CtxResp:
    def __init__(self, code: int = 200, data: bytes = b"") -> None:
        self._code = code
        self._data = data

    def __enter__(self) -> "_CtxResp":
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def getcode(self) -> int:
        return self._code

    def read(self) -> bytes:
        return self._data


class _Completed:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_check_health_healthy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _CtxResp(200))
    assert checker.check_health("10.0.0.1", 8080, 1) == "HEALTHY"


def test_check_health_http_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom(*_a: object, **_k: object) -> object:
        raise urllib.error.HTTPError("http://x", 503, "err", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert checker.check_health("10.0.0.1", 8080, 1) == "NOT_RESPONDING"


def test_check_health_transport_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom(*_a: object, **_k: object) -> object:
        raise urllib.error.URLError("down")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert checker.check_health("10.0.0.1", 8080, 1) == "ERROR"


def test_resolve_peer_id_via_srun(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: _Completed("QmSelf\n"))
    assert checker.resolve_peer_id("node001", 8092, 5) == "QmSelf"


def test_resolve_peer_id_no_host() -> None:
    assert checker.resolve_peer_id("", 8092, 5) is None


def test_resolve_peer_id_empty_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: _Completed("   \n"))
    assert checker.resolve_peer_id("node001", 8092, 5) is None


def test_resolve_peer_id_srun_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom(*_a: object, **_k: object) -> object:
        raise subprocess.SubprocessError("srun unavailable")

    monkeypatch.setattr(checker.subprocess, "run", _boom)
    assert checker.resolve_peer_id("node001", 8092, 5) is None


def test_build_report_health_peers_and_last_seen(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(checker, "_http_get", lambda url, timeout: (200, b"") if "10.0.0.1" in url else (503, b""))
    monkeypatch.setattr(checker, "resolve_peer_id", lambda host, port, timeout: {"n0": "Qm1", "n1": "Qm2"}.get(host))
    report = checker.build_report(["10.0.0.1", "10.0.0.2"], ["n0", "n1"], 1, 8080, 8092, 1.0, {}, {}, {}, 1000)

    assert report["checked_at"] == 1000
    first, second = report["replicas"]
    assert first == {
        "node_rank": 0,
        "node_ip": "10.0.0.1",
        "node_host": "n0",
        "peer_id": "Qm1",
        "health": "HEALTHY",
        "last_seen": 1000,
    }
    # 10.0.0.2 has never been healthy: NOT_DEPLOYED, and peer id isn't resolved (only resolved when healthy).
    assert second["health"] == "NOT_DEPLOYED"
    assert second["node_host"] == "n1"  # head node name is reported regardless of health
    assert second["peer_id"] is None
    assert second["last_seen"] is None


def test_build_report_not_deployed_until_first_healthy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(checker, "_http_get", lambda url, timeout: None)  # transport error
    monkeypatch.setattr(checker, "resolve_peer_id", lambda *a, **k: None)
    report = checker.build_report(["10.0.0.1"], ["n0"], 1, 8080, 8092, 1.0, {}, {}, {}, 100)
    assert report["replicas"][0]["health"] == "NOT_DEPLOYED"
    assert report["replicas"][0]["last_seen"] is None


def test_build_report_node_rank_uses_nodes_per_replica(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(checker, "_http_get", lambda url, timeout: (200, b""))
    monkeypatch.setattr(checker, "resolve_peer_id", lambda *a, **k: None)
    report = checker.build_report(["10.0.0.1", "10.0.0.2"], ["n0", "n4"], 4, 8080, 8092, 1.0, {}, {}, {}, 1000)
    assert [r["node_rank"] for r in report["replicas"]] == [0, 4]


def test_build_report_node_host_none_when_hosts_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(checker, "_http_get", lambda url, timeout: (200, b""))
    monkeypatch.setattr(checker, "resolve_peer_id", lambda *a, **k: None)
    # Fewer host names than IPs (or none at all) must not crash; node_host is None.
    report = checker.build_report(["10.0.0.1"], [], 1, 8080, 8092, 1.0, {}, {}, {}, 1000)
    assert report["replicas"][0]["node_host"] is None


def test_build_report_freezes_last_seen_and_caches_peer(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    state = {"healthy": True}
    monkeypatch.setattr(checker, "_http_get", lambda url, timeout: (200, b"") if state["healthy"] else (503, b""))
    monkeypatch.setattr(checker, "resolve_peer_id", lambda *a, **k: "Qm")
    peer_ids: dict[int, str] = {}
    peer_attempts: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    checker.build_report(["10.0.0.1"], ["n0"], 1, 8080, 8092, 1.0, peer_ids, peer_attempts, last_seen, 100)
    assert last_seen[0] == 100
    assert peer_ids[0] == "Qm"

    state["healthy"] = False
    report = checker.build_report(["10.0.0.1"], ["n0"], 1, 8080, 8092, 1.0, peer_ids, peer_attempts, last_seen, 200)
    assert last_seen[0] == 100  # frozen at last-healthy time
    # Was healthy before, so a later failure surfaces the real error (not NOT_DEPLOYED), peer id stays cached.
    assert report["replicas"][0]["health"] == "NOT_RESPONDING"
    assert report["replicas"][0]["last_seen"] == 100
    assert report["replicas"][0]["peer_id"] == "Qm"


def test_build_report_caps_peer_resolution_attempts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(checker, "_http_get", lambda url, timeout: (200, b""))  # always healthy
    calls = {"n": 0}

    def _never_resolves(*_a: object, **_k: object) -> None:
        calls["n"] += 1
        return None

    monkeypatch.setattr(checker, "resolve_peer_id", _never_resolves)
    peer_ids: dict[int, str] = {}
    peer_attempts: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    for tick in range(10):
        checker.build_report(["10.0.0.1"], ["n0"], 1, 8080, 8092, 1.0, peer_ids, peer_attempts, last_seen, tick)
    assert calls["n"] == checker._MAX_PEER_ATTEMPTS  # stops retrying after the cap


def test_write_report_atomically(tmp_path) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / "replica_health.json"
    checker.write_report_atomically({"checked_at": 5, "replicas": []}, str(target))
    assert json.loads(target.read_text()) == {"checked_at": 5, "replicas": []}
    assert not (tmp_path / "replica_health.json.tmp").exists()  # temp file renamed away


class _Ran:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr


def test_cancel_previous_job_success(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    def _ok(cmd, **_k):  # type: ignore[no-untyped-def]
        seen["cmd"] = cmd
        return _Ran(returncode=0)

    monkeypatch.setattr(checker.subprocess, "run", _ok)
    assert checker.cancel_previous_job("4242") is True
    assert seen["cmd"] == ["scancel", "4242"]


def test_cancel_previous_job_nonzero_exit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class _Failed:
        returncode = 1
        stderr = "no such job"

    monkeypatch.setattr(checker.subprocess, "run", lambda *a, **k: _Failed())
    assert checker.cancel_previous_job("4242") is False


def test_cancel_previous_job_subprocess_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _boom(*_a: object, **_k: object) -> object:
        raise subprocess.SubprocessError("scancel missing")

    monkeypatch.setattr(checker.subprocess, "run", _boom)
    assert checker.cancel_previous_job("4242") is False


def _report(*healths: str) -> dict:
    return {"checked_at": 1, "replicas": [{"health": h} for h in healths]}


def test_all_replicas_healthy_true_when_all_present_and_healthy() -> None:
    assert checker.all_replicas_healthy(_report("HEALTHY", "HEALTHY"), 2) is True


def test_all_replicas_healthy_false_when_any_unhealthy_or_missing() -> None:
    assert checker.all_replicas_healthy(_report("HEALTHY", "NOT_RESPONDING"), 2) is False
    assert checker.all_replicas_healthy(_report("HEALTHY"), 2) is False  # one not reported yet


def test_all_replicas_healthy_guards_zero_expected() -> None:
    # No expected replicas -> never "all healthy"; otherwise the empty all() would
    # be vacuously true and the handover would cancel the predecessor immediately,
    # dropping the only allocation.
    assert checker.all_replicas_healthy(_report(), 0) is False
    assert checker.all_replicas_healthy({"replicas": []}, 0) is False
    assert checker.all_replicas_healthy(_report(), -1) is False
