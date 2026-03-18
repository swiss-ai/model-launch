from enum import Enum


class ModelHealth(Enum):
    WAITING = "WAITING"
    HEALTHY = "HEALTHY"
    ERROR = "ERROR"
    NOT_RESPONDING = "NOT_RESPONDING"
