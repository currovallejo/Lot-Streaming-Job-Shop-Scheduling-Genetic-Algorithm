from dataclasses import dataclass
from typing import Sequence, Optional, Any


@dataclass(frozen=True)
class GAResult:
    """Results from a genetic algorithm run."""

    best_fitness_per_gen: Sequence[float]
    best_ever_per_gen: Sequence[float]
    makespan: float
    penalty: Optional[float] = None
    best_solution: Optional[Any] = None
    metadata: Optional[dict] = None
