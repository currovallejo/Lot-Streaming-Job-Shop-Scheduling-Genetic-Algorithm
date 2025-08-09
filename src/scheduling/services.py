"""
Application: public decoder classes and orchestration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from .. import params as params_mod
from .lot_sizing import distribute_demand
from ..domain import ScheduledOperation
from .state import (
    StaticData,
    DynamicState,
    Cursor,
    build_routes,
    build_precedences,
    build_time_arrays,
)
from .rules import (
    start_time_no_shifts,
    start_time_with_shifts,
    completion_time_setup_dependent,
    completion_time_setup_independent,
    is_big_lot_duration,
)


class JobShopDecoder:
    """
    Decoder for the Job Shop Scheduling Problem.
    """

    def __init__(self, problem_params: params_mod.JobShopRandomParams):
        self.problem_params = problem_params

    def decode(self, encoded_solution: Any) -> Any:
        """
        Decode the encoded solution (chromosome) to a semi-active schedule.

        Args:
            - encoded_solution: [lot_size_genes, lot_sequence]

        Returns:
            - makespan (int):  max completion time across all machines
            - penalty (int): max penalty for violating shift constraints
            - setup_start  : np.ndarray [M,J,U]
            - completion : np.ndarray [M,J,U]
            - semi : [lot_sizes, lot_sequence]
        """
        p = self.problem_params
        n_lots, n_jobs, n_machines = p.n_lots, p.n_jobs, p.n_machines
        jobs, lots, machines = p.jobs, p.lots, p.machines

        # Lot sizing
        genes = np.asarray(encoded_solution[0], dtype=float)
        lot_sizes = distribute_demand(genes, p.demand, jobs, lots, n_lots)
        lot_sequence = encoded_solution[1]
        semi = [lot_sizes, lot_sequence]

        # DTOs
        static = StaticData(
            sequence=p.seq,
            p_times=p.p_times,
            s_times=p.setup,
            n_lots=n_lots,
            shift_time=p.shift_time,
            is_setup_dependent=p.is_setup_dependent,
            is_shift_constraints=p.shift_constraints,
        )
        setup_start, completion = build_time_arrays(n_machines, n_jobs, n_lots)
        dynamic = DynamicState(
            setup_start=setup_start,
            completion=completion,
            routes=build_routes(p.seq, jobs, lots),
            precedences=build_precedences(machines, p.is_setup_dependent),
            lot_sizes=lot_sizes,
        )

        # Penalty initialization
        penalty = 0
        if static.is_shift_constraints:
            penalty_coeff = int(1e2)
            max_lot_penalty = 0

        # Decode loop
        for j, u in lot_sequence:
            if int(lot_sizes[n_lots * j + u]) == 0:
                continue
            route = dynamic.routes[(j, u)]
            if not route:
                continue
            m = route[0]
            cur = Cursor(job=j, lot=u, machine=m)

            # start
            if not static.is_shift_constraints:
                s_start = start_time_no_shifts(static, dynamic, cur)
            else:
                s_start = start_time_with_shifts(static, dynamic, cur)

            # completion
            if static.is_setup_dependent:
                comp = completion_time_setup_dependent(static, dynamic, cur, s_start)
            else:
                comp = completion_time_setup_independent(static, dynamic, cur, s_start)

            # penalty
            if static.is_shift_constraints and is_big_lot_duration(
                static, dynamic, cur
            ):
                lot_pen = comp % static.shift_time
                if lot_pen > max_lot_penalty:
                    max_lot_penalty = lot_pen
                    penalty = penalty_coeff * max_lot_penalty

            # commit
            dynamic.setup_start[m, j, u] = s_start
            dynamic.completion[m, j, u] = comp
            dynamic.precedences[m].append((j, u))
            route.pop(0)

        makespan = int(dynamic.completion.max()) if dynamic.completion.size else 0
        return makespan, int(penalty), dynamic.setup_start, dynamic.completion, semi


def build_schedule_times_df_from_ops(
    self, ops: list[ScheduledOperation]
) -> pd.DataFrame:
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
