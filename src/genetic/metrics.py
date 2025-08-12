"""Metrics collection for Genetic Algorithm (GA) execution.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog
"""

from typing import Protocol, Optional
from .results import GAResult


class MetricsSink(Protocol):
    """Protocol for collecting metrics during GA execution."""

    def on_generation_end(
        self, best_gen: float, best_ever: float
    ) -> None:
        """Called at the end of each generation."""
        ...

    def set_best_solution(self, solution) -> None:
        """Set the best solution found."""
        ...

    def set_final_metrics(self, makespan: float, penalty: Optional[float]) -> None:
        """Set final makespan and penalty metrics."""
        ...

    def add_metadata(self, key: str, value) -> None:
        """Add metadata to the results."""
        ...

    def finalize(self) -> GAResult:
        """Called at the end of the GA run to build final results."""
        ...


class InMemoryMetrics:
    """In-memory implementation of MetricsSink."""

    def __init__(self):
        self.best_fitness_per_gen: list[float] = []
        self.best_ever_per_gen: list[float] = []
        self.best_solution = None
        self.makespan: float = float("inf")
        self.penalty: Optional[float] = None

    def on_generation_end(self, best_gen: float, best_ever: float) -> None:
        """Store the best fitness values for the current generation and the best ever."""
        self.best_fitness_per_gen.append(best_gen)
        self.best_ever_per_gen.append(best_ever)

    def set_best_solution(self, solution) -> None:
        """Store the best solution found during the GA run."""
        self.best_solution = solution

    def set_final_metrics(self, makespan: float, penalty: Optional[float]) -> None:
        """Set final makespan and penalty metrics."""
        self.makespan = makespan
        self.penalty = penalty

    def finalize(self) -> GAResult:
        """Return the collected metrics as a GAResult object."""
        return GAResult(
            best_fitness_per_gen=self.best_fitness_per_gen.copy(),
            best_ever_per_gen=self.best_ever_per_gen.copy(),
            makespan=self.makespan,
            penalty=self.penalty,
            best_solution=self.best_solution,
        )
