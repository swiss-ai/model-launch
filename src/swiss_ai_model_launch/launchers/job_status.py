from enum import Enum


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, state: str) -> "JobStatus":
        try:
            return cls(state)
        except ValueError:
            return cls.UNKNOWN
