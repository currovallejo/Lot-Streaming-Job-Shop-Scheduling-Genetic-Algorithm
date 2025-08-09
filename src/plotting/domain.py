from __future__ import annotations
from dataclasses import dataclass


# --- Value Objects ---
@dataclass(frozen=True)
class OperationId:
    job: int
    machine: int
    lot: int


@dataclass(frozen=True)
class TimeWindow:
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
    id: OperationId
    time: TimeWindow
    lot_size: int
