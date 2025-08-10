"""
Lot sizing: distribute demand into integer lot sizes (legacy behavior).
"""

from __future__ import annotations
import numpy as np
from typing import Iterable


def distribute_demand(
    lot_genes: np.ndarray,
    demand: dict,
    jobs: Iterable,
    lots: Iterable,
    n_lots: int,
) -> np.ndarray:
    lot_sizes = np.copy(np.asarray(lot_genes, dtype=float))

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
