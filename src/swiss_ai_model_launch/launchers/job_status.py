from enum import Enum


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, state: str) -> "JobStatus":
        # SLURM decorates some terminal states, e.g. "CANCELLED by 1234"; the
        # caller already passes the first token, but normalise defensively.
        normalised = state.strip().split()[0].upper() if state.strip() else state
        try:
            return cls(normalised)
        except ValueError:
            return cls.UNKNOWN
