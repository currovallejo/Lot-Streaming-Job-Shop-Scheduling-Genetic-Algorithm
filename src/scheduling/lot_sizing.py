"""
Lot sizing module for Lot Streaming Job Shop Scheduling Problem.

This module implements lot size distribution functionality that transforms genetic
algorithm lot size genes into feasible lot sizes that satisfy job demands. It handles
proportional distribution, residual allocation, and ensures integer lot sizes for
the Lot Streaming Job Shop Scheduling Problem optimization.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejog
Github: https://github.com/currovallejog
"""

from __future__ import annotations
import numpy as np
from typing import Iterable


def distribute_demand(
    lhs: np.ndarray,
    demand: dict,
    jobs: Iterable,
    lots: Iterable,
    n_lots: int,
) -> np.ndarray:
    """
    Distribute the demand across the lot sizes based on the genes provided.
    Args:
        lhs (np.ndarray): Left-hand side of the chromosome representing lot sizes.
        demand (dict): Dictionary containing the demand for each job.
        jobs (Iterable): Iterable of job identifiers.
        lots (Iterable): Iterable of lot identifiers.
        n_lots (int): Number of lots.
    Returns:
        np.ndarray: Array of lot sizes adjusted to meet the demand.
    """
    lot_sizes = np.copy(np.asarray(lhs, dtype=float))

    for j in jobs:
        span = slice(n_lots * j, n_lots * (j + 1))
        total = float(np.sum(lot_sizes[span]))

        if total != 0:
            lot_sizes[span] = lot_sizes[span] / total
            for u in lots:
                lot_sizes[n_lots * j + u] = int(lot_sizes[n_lots * j + u] * demand[j])

            total_pre = int(np.sum(lot_sizes[span]))
            residual = int(demand[j]) - total_pre
            if residual > 0:
                for u in lots:
                    lot_sizes[n_lots * j + u] += 1
                    residual -= 1
                    if residual == 0:
                        break
        else:
            even = int(demand[j]) // n_lots
            lot_sizes[span] = even
            lot_sizes[n_lots * j + (n_lots - 1)] = int(demand[j]) - int(
                np.sum(lot_sizes[span][:-1])
            )

    return lot_sizes.astype(int)
