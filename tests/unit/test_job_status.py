from swiss_ai_model_launch.launchers.job_status import JobStatus


def test_known_states_round_trip() -> None:
    for state in ["PENDING", "RUNNING", "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"]:
        assert JobStatus.from_str(state) is JobStatus(state)


def test_cancelled_with_trailing_uid() -> None:
    # sacct reports a cancelled job as "CANCELLED by 12345".
    assert JobStatus.from_str("CANCELLED by 12345") is JobStatus.CANCELLED


def test_lowercase_is_normalised() -> None:
    assert JobStatus.from_str("running") is JobStatus.RUNNING


def test_unrecognised_state_is_unknown() -> None:
    assert JobStatus.from_str("NODE_FAIL") is JobStatus.UNKNOWN
    assert JobStatus.from_str("") is JobStatus.UNKNOWN
