# ruff: noqa: S603, S607  # subprocess invocations against controlled paths/binaries

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from swiss_ai_model_launch.assets import replica_probe
from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealth, replica_checker
from swiss_ai_model_launch.launchers.framework import OCF_BOOTSTRAP_ADDR, OCF_BOOTSTRAP_ADDR_DEV
from swiss_ai_model_launch.launchers.utils import render_helper_sbatch_header


def test_sentinels_match_probe() -> None:
    assert replica_checker.REPORT_BEGIN == replica_probe.REPORT_BEGIN
    assert replica_checker.REPORT_END == replica_probe.REPORT_END


def test_dnt_base_url_prod() -> None:
    assert replica_checker.dnt_base_url_from_bootstrap(OCF_BOOTSTRAP_ADDR) == "http://148.187.108.178:8092"


def test_dnt_base_url_dev() -> None:
    assert replica_checker.dnt_base_url_from_bootstrap(OCF_BOOTSTRAP_ADDR_DEV) == "http://148.187.108.177:8092"


def test_dnt_base_url_invalid() -> None:
    with pytest.raises(ValueError):
        replica_checker.dnt_base_url_from_bootstrap("not-a-multiaddr")


def test_render_probe_script_embeds_config_and_source() -> None:
    api_key = "sk-secret'val"  # contains a single quote to exercise shell-escaping
    script = replica_checker.render_probe_script("vendor/model-aWBl", api_key, "http://1.2.3.4:8092", 7)
    assert f"export SML_SERVED_MODEL_NAME={shlex.quote('vendor/model-aWBl')}" in script
    assert f"export SML_DNT_BASE_URL={shlex.quote('http://1.2.3.4:8092')}" in script
    assert "export SML_REPLICA_TIMEOUT=7" in script
    assert f"export SML_CSCS_API_KEY={shlex.quote(api_key)}" in script
    assert "export SML_REPLICA_INTERVAL=0" in script  # one-shot by default
    assert "def matching_peers(" in script  # probe source is embedded
    assert 'python3 "$probe"' in script


def test_render_probe_script_sets_loop_interval() -> None:
    script = replica_checker.render_probe_script("m", "k", "http://x:8092", refresh_interval_seconds=5)
    assert "export SML_REPLICA_INTERVAL=5" in script


def _report_text(obj: dict[str, object]) -> str:
    return f"some logs\n{replica_checker.REPORT_BEGIN}\n{json.dumps(obj)}\n{replica_checker.REPORT_END}\ntrailing"


def test_parse_report_all_healthy() -> None:
    text = _report_text(
        {"replicas": [{"peer_id": "QmA", "health": "HEALTHY"}, {"peer_id": "QmB", "health": "HEALTHY"}]}
    )
    report = replica_checker.parse_report(text, "m", 2)
    assert report.found == 2
    assert report.replicas[0] == ReplicaHealth("QmA", ModelHealth.HEALTHY)
    assert report.all_healthy
    assert report.complete


def test_parse_report_carries_heartbeat_and_checked_at() -> None:
    text = _report_text(
        {
            "checked_at": 1771765000,
            "replicas": [
                {"peer_id": "QmA", "health": "HEALTHY", "last_seen": 1771764990},
                {"peer_id": "QmB", "health": "HEALTHY"},
            ],
        }
    )
    report = replica_checker.parse_report(text, "m", 2)
    assert report.checked_at == 1771765000
    assert report.replicas[0].last_seen == 1771764990
    assert report.replicas[1].last_seen is None  # absent -> None


def test_parse_report_unknown_health_maps_to_error() -> None:
    text = _report_text({"replicas": [{"peer_id": "QmA", "health": "WAT"}]})
    report = replica_checker.parse_report(text, "m", 1)
    assert report.replicas[0].health == ModelHealth.ERROR
    assert not report.all_healthy


def test_parse_report_complete_requires_expected_count() -> None:
    text = _report_text({"replicas": [{"peer_id": "QmA", "health": "HEALTHY"}]})
    report = replica_checker.parse_report(text, "m", 2)
    assert report.all_healthy  # the replica found is healthy
    assert not report.complete  # but fewer than the expected 2


def test_parse_report_missing_sentinels() -> None:
    report = replica_checker.parse_report("just logs, no report block", "m", 1)
    assert report.table_error is not None
    assert report.replicas == ()
    assert not report.all_healthy


def test_parse_report_malformed_json() -> None:
    text = f"{replica_checker.REPORT_BEGIN}\n{{not valid json]\n{replica_checker.REPORT_END}"
    report = replica_checker.parse_report(text, "m", 1)
    assert report.table_error is not None


def test_parse_report_propagates_table_error() -> None:
    text = _report_text({"replicas": [], "table_error": "connection refused"})
    report = replica_checker.parse_report(text, "m", 2)
    assert report.table_error == "connection refused"
    assert not report.all_healthy


def _full_probe_job_script() -> str:
    header = render_helper_sbatch_header(job_name="sml_test", account="acct", partition="normal", time="00:10:00")
    body = replica_checker.render_probe_script("vendor/model-aWBl", "sk-key", "http://1.2.3.4:8092")
    return header + "\n" + body


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_probe_job_script_bash_syntax(tmp_path: Path) -> None:
    path = tmp_path / "probe.sh"
    path.write_text(_full_probe_job_script())
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not available")
def test_probe_job_script_shellcheck(tmp_path: Path) -> None:
    path = tmp_path / "probe.sh"
    path.write_text(_full_probe_job_script())
    result = subprocess.run(["shellcheck", "-S", "warning", str(path)], capture_output=True)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()


@pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("python3") is None,
    reason="bash and python3 required",
)
def test_probe_job_script_runs_and_emits_parseable_report(tmp_path: Path) -> None:
    # Aim the probe at an unreachable DNT host: it should still print a report
    # carrying a table_error, proving the heredoc + python3 + report wiring works.
    body = replica_checker.render_probe_script("vendor/model-aWBl", "sk-key", "http://127.0.0.1:1", timeout_seconds=1)
    path = tmp_path / "probe.sh"
    path.write_text("#!/bin/bash\n" + body)
    result = subprocess.run(["bash", str(path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = replica_checker.parse_report(result.stdout, "vendor/model-aWBl", 0)
    assert report.table_error is not None
    assert report.replicas == ()
