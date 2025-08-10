# src/scheduling/mappers.py
# --- Domain mappers: tensors -> ScheduledOperation list ---

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd

from domain.scheduling import OperationId, TimeWindow, ScheduledOperation
from jobshop import JobShopRandomParams


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


def build_schedule_times_df_from_ops(ops: list[ScheduledOperation]) -> pd.DataFrame:
    """
    Build a schedule DataFrame from domain operations.

    Args:
        - ops: list of ScheduledOperation

    Returns:
        - schedule_df: DataFrame with columns
          [job, lot, machine, setup_start_time, start_time, completion_time, lot_size].
    """
    cols = [
        "job",
        "lot",
        "machine",
        "setup_start_time",
        "start_time",
        "completion_time",
        "lot_size",
    ]

    rows = [
        {
            "job": op.id.job,
            "lot": op.id.lot,
            "machine": op.id.machine,
            "setup_start_time": op.time.setup_start,
            "start_time": op.time.start,
            "completion_time": op.time.completion,
            "lot_size": op.lot_size,
        }
        for op in ops
        if op.time.completion > 0
    ]

    df = pd.DataFrame(rows, columns=cols)

    df = df.sort_values(["machine", "job", "lot"]).reset_index(drop=True)
    return df.astype(
        {
            "job": "int64",
            "lot": "int64",
            "machine": "int64",
            "setup_start_time": "int64",
            "start_time": "int64",
            "completion_time": "int64",
            "lot_size": "int64",
        }
    )
