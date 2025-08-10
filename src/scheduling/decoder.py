"""
Chromosome decoder for Lot Streaming Job Shop Scheduling Problem.

This module implements chromosome decoding functionality that transforms genetic
algorithm solutions into feasible job shop schedules. It handles lot size distribution,
operation sequencing, setup times, shift constraints, and calculates makespan and
penalties for the Lot Streaming Job Shop Scheduling Problem optimization.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejog
Github: https://github.com/currovallejog
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from jobshop import JobShopRandomParams
from .lot_sizing import distribute_demand
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


class ChromosomeDecoder:
    """
    Decoder for the Job Shop Scheduling Problem.
    """

    def __init__(self, problem_params: JobShopRandomParams):
        """
        Initialize the decoder with job shop parameters.
        Args:
            problem_params (JobShopRandomParams): Parameters for the job shop problem.
        """
        self.problem_params = problem_params

    def decode(self, encoded_solution: Tuple) -> Tuple:
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
