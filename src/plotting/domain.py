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

    # --- Labels (pure functions of identity) ---
    @property
    def product_label(self) -> str:
        return f"P {self.id.job}"

    @property
    def resource_label(self) -> str:
        return f"M {self.id.machine}"

    @property
    def text_label(self) -> str:
        return f"P{self.id.job} - L{self.id.lot}"


# --- Configuration for plotting services ---


@dataclass(frozen=True)
class GanttConfig:
    shift_time: int
    auto_open: bool = True
