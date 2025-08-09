# --- Core scheduling domain ---

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class OperationId:
    """Unique identifier for an operation in the job shop."""

    job: int
    machine: int
    lot: int


@dataclass(frozen=True)
class TimeWindow:
    """Time window for an operation, including setup and processing times."""

    setup_start: int
    start: int
    completion: int

    # --- Derived durations ---
    @property
    def proc_duration(self) -> int:
        return self.completion - self.start

    @property
    def setup_duration(self) -> int:
        return self.start - self.setup_start


@dataclass(frozen=True)
class ScheduledOperation:
    """Represents a scheduled operation with its ID, time window, and lot size."""

    id: OperationId
    time: TimeWindow
    lot_size: int
