from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

class RunStatus(StrEnum):
    """Enum for run status."""

    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"

@dataclass
class Experiment:
    """Class representing an experiment."""

    experiment_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

@dataclass
class Run:
    """Class representing a run"""

    run_id: str
    experiment_id: str
    status: RunStatus
    started_at: datetime
    run_name: str | None = None
    ended_at: datetime | None = None

@dataclass(frozen=True)
class MetricRecord:
    """Class representing a metric."""

    key: str
    value: float
    step: int
    timestamp: datetime