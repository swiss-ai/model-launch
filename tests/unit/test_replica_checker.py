import json

from swiss_ai_model_launch.cli.healthcheck import ModelHealth, ReplicaHealth, parse_health_report


def test_parse_health_report_all_healthy() -> None:
    text = json.dumps(
        {
            "checked_at": 1771765000,
            "replicas": [
                {"node_rank": 0, "node_ip": "10.0.0.1", "peer_id": "QmA", "health": "HEALTHY", "last_seen": 1771764990},
                {"node_rank": 1, "node_ip": "10.0.0.2", "peer_id": "QmB", "health": "HEALTHY", "last_seen": 1771764991},
            ],
        }
    )
    report = parse_health_report(text, "m", 2)
    assert report.error is None
    assert report.checked_at == 1771765000
    assert report.found == 2
    assert report.all_healthy
    assert report.complete
    assert report.replicas[0] == ReplicaHealth(ModelHealth.HEALTHY, "QmA", 1771764990, 0, "10.0.0.1")


def test_parse_health_report_unknown_health_maps_to_error() -> None:
    report = parse_health_report(json.dumps({"replicas": [{"node_rank": 0, "health": "WAT"}]}), "m", 1)
    assert report.replicas[0].health == ModelHealth.ERROR
    assert not report.all_healthy


def test_parse_health_report_missing_optionals_are_none() -> None:
    report = parse_health_report(json.dumps({"replicas": [{"health": "HEALTHY"}]}), "m", 1)
    replica = report.replicas[0]
    assert replica.peer_id is None
    assert replica.last_seen is None
    assert replica.node_rank is None
    assert replica.node_ip is None


def test_parse_health_report_complete_requires_expected_count() -> None:
    report = parse_health_report(json.dumps({"replicas": [{"node_rank": 0, "health": "HEALTHY"}]}), "m", 2)
    assert report.all_healthy  # the replica present is healthy
    assert not report.complete  # but fewer than the expected 2


def test_parse_health_report_one_unhealthy() -> None:
    text = json.dumps(
        {"replicas": [{"node_rank": 0, "health": "HEALTHY"}, {"node_rank": 1, "health": "NOT_RESPONDING"}]}
    )
    report = parse_health_report(text, "m", 2)
    assert report.found == 2
    assert not report.all_healthy


def test_parse_health_report_malformed_json() -> None:
    report = parse_health_report("{not valid json]", "m", 1)
    assert report.error is not None
    assert report.replicas == ()


def test_parse_health_report_non_dict() -> None:
    report = parse_health_report("[1, 2, 3]", "m", 1)
    assert report.error is not None
    assert report.replicas == ()
