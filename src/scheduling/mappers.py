# src/scheduling/mappers.py
# --- Domain mappers: tensors -> ScheduledOperation list ---

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from ..domain.scheduling import OperationId, TimeWindow, ScheduledOperation
from src.params import JobShopRandomParams


@dataclass
class OperationAssembler:
    """
    Build domain operations from schedule tensors.

    Args:
    - params: JobShopRandomParams to validate dimensions and index conventions.
    """

    params: JobShopRandomParams

    def from_tensors(
        self,
        semi_encoded_solution: list,
        setup_start_times: np.ndarray,
        completion_times: np.ndarray,
    ) -> List[ScheduledOperation]:
        """
        Args:
        - semi_encoded_solution: [lot_sizes, lot_sequence]
        - setup_start_times: [M, J, U]
        - lot_start_times: [M, J, U]
        - completion_times: [M, J, U]

        Returns:
        - ops: flat list of ScheduledOperation
        """
        lot_sizes = np.asarray(semi_encoded_solution[0], dtype=int)
        p_times = self.params.p_times
        n_lots = self.params.n_lots
        set_mju = {
            (m, j, u)
            for m in range(self.params.n_machines)
            for j in range(self.params.n_jobs)
            for u in range(n_lots)
        }

        ops: List[ScheduledOperation] = []
        for m, j, u in set_mju:
            if completion_times[m, j, u] > 0:
                tw = TimeWindow(
                    setup_start=int(setup_start_times[m, j, u]),
                    start=(
                        completion_times[m, j, u]
                        - p_times[m, j] * lot_sizes[n_lots * j + u]
                    ),
                    completion=completion_times[m, j, u],
                )
                ops.append(
                    ScheduledOperation(
                        id=OperationId(job=j, machine=m, lot=u),
                        time=tw,
                        lot_size=int(lot_sizes[n_lots * j + u]),
                    )
                )
        return ops
