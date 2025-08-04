"""
Created on Aug 05 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: decoder.py - decodification of the chromosome to the solution
"""

# --------- LIBRARIES ---------
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

# --------- OTHER PYTHON FILES USED ---------
import params
import plot

# --------- DECODER ---------


class SolutionDecoder(ABC):
    """Abstract base class for solution decoding."""

    def __init__(self, problem_params):
        """
        Initialize decoder with problem parameters

        Args:
            problem_params: Problem parameters object
        """
        self.problem_params = problem_params

    @abstractmethod
    def decode(self, encoded_solution: Any) -> Any:
        """
        Decode the encoded solution to a problem-specific solution.

        Args:
            encoded_solution: Encoded solution (e.g., chromosome)

        Returns:
            Decoded solution (e.g., makespan, schedule, etc.)
        """

        pass

    @abstractmethod
    def get_fitness(self, decoded_solution):
        """
        Calculate the fitness of the decoded solution.

        Args:
            decoded_solution: Decoded solution

        Returns:
            Fitness value of the solution
        """
        pass


class JobShopDecoder(SolutionDecoder):
    """
    Decoder for Job Shop Scheduling Problem solutions.
    This class decodes a chromosome into a schedule and calculates the makespan.
    """

    def __init__(self, problem_params: params.JobShopRandomParams):
        """
        Initialize JobShop decoder with problem parameters

        Args:
            problem_params: JobShopRandomParams object containing problem parameters
            chromosome: numpy array representing the chromosome to decode
        """
        super().__init__(problem_params)

    def decode(self, encoded_solution: Any) -> Any:
        """
        Decode the encoded solution (chromosome) to an semi-active schedule.

        Args:
            encoded_solution: Encoded solution (chromosome)

        Returns:
            Tuple containing setup start times, lots start times and completion times.
        """

        # helpers
        def _build_routes() -> Dict[Tuple[int, int], List[int]]:
            """
            Build the routes for each job and lot based on the problem parameters.

            Returns:
                Dictionary with routes for each job and lot.
            """
            return {(job, lot): sequence[job][:] for job in jobs for lot in lots}

        def _build_precedences() -> Dict[int, List[Tuple[int, int]]]:
            """
            Build the precedences for each machine based on the problem parameters.

            Returns:
                Dictionary with precedences for each machine.
            """
            precedences = {}
            for machine in machines:
                if is_setup_dependent:
                    precedences[machine] = [(-1, 0)]
                else:
                    precedences[machine] = []

            return precedences

        def _build_times_container() -> np.ndarray:
            return np.full((n_machines, n_jobs, n_lots), 0)

        def _distribute_demand() -> np.ndarray:
            """
            Distributes a total demand into parts based on the given fractions
            such that the sum of the parts equals the total demand.
            """
            lot_sizes = np.copy(encoded_solution[0])
            for job in jobs:
                total = np.sum(lot_sizes[n_lots * job : n_lots * (job + 1)])

                if total != 0:  # Avoid division by zero
                    lot_sizes[n_lots * job : n_lots * (job + 1)] = (
                        lot_sizes[n_lots * job : n_lots * (job + 1)] / total
                    )

                    for lot in lots:
                        lot_sizes[n_lots * job + lot] = int(
                            lot_sizes[n_lots * job + lot] * demand[job]
                        )
                    total_preliminary = sum(
                        lot_sizes[n_lots * job : n_lots * (job + 1)]
                    )
                    residual = demand[job] - total_preliminary

                    if residual > 0:
                        for lot in lots:
                            lot_sizes[n_lots * job + lot] += 1
                            residual -= 1
                            if residual == 0:
                                break

                else:
                    lot_sizes[n_lots * job : n_lots * (job + 1)] = (
                        int(demand[job]) / n_lots
                    )
                    lot_sizes[n_lots * job + n_lots - 1] = (demand[job]) - sum(
                        lot_sizes[n_lots * job : n_lots * (job)]
                    )

            return lot_sizes

        def _is_empty_machine() -> bool:
            """
            Check if the current machine is empty (no jobs have yet been processed at the current machine) based on the precedences.
            """

            if is_setup_dependent:
                return precedences[current_machine] == [-1]
            else:
                return precedences[current_machine] == []

        def _is_first_machine() -> bool:
            """
            Check if the current machine is the first in the job's route.
            """
            return current_machine == sequence[current_job][0]

        def _completion_time_in_previous_machine() -> int:
            """
            Get the completion time in the previous machine for the current job.
            """
            previous_machine = sequence[current_job][
                sequence[current_job].index(current_machine) - 1
            ]
            return completion_time[previous_machine, current_job, current_lot]

        def _completion_time_in_current_machine() -> int:
            """
            Get the completion time in the current machine for the last processed lot.
            """
            predecessor = precedences[current_machine][-1]
            return completion_time[current_machine, predecessor[0], predecessor[1]]

        def _lot_start_time():

            if _is_first_machine() and _is_empty_machine():
                start_time[current_machine, current_job, current_lot] = 0

            elif _is_first_machine() and not _is_empty_machine():
                start_time[current_machine, current_job, current_lot] = (
                    _completion_time_in_current_machine()
                )

            elif not _is_first_machine() and _is_empty_machine():
                start_time[current_machine, current_job, current_lot] = (
                    _completion_time_in_previous_machine()
                )

            elif not _is_first_machine() and not _is_empty_machine():
                start_time[current_machine, current_job, current_lot] = max(
                    _completion_time_in_current_machine(),
                    _completion_time_in_previous_machine(),
                )

        def _lot_completion_time_setup_dependent() -> int:
            """
            Calculate the completion time of the lot in the current machine with setup dependent times.
            """
            return (
                start_time[current_machine, current_job, current_lot]
                + s_times[
                    current_machine,
                    current_job + 1,
                    precedences[current_machine][-1][0] + 1,
                ]
                + p_times[current_machine, current_job]
                * lot_sizes[n_lots * current_job + current_lot]
            )

        def _lot_completion_time_setup_independent() -> int:
            """
            Calculate the completion time of the lot in the current machine with setup independent times.
            """
            completion_time[current_machine, current_job, current_lot] = (
                start_time[current_machine, current_job, current_lot]
                + s_times[current_machine, current_job]
                + p_times[current_machine, current_job]
                * lot_sizes[n_lots * current_job + current_lot]
            )

        # helpers for shift constraints decoding
        def _lot_processing_time() -> int:
            """
            Calculate the processing time for the lot in the current machine.
            """
            if is_setup_dependent:
                return (
                    s_times[
                        current_machine,
                        current_job + 1,
                        precedences[current_machine][-1][0] + 1,
                    ]
                    + p_times[current_machine, current_job]
                    * lot_sizes[n_lots * current_job + current_lot]
                )
            else:
                return (
                    s_times[current_machine, current_job]
                    + p_times[current_machine, current_job]
                    * lot_sizes[n_lots * current_job + current_lot]
                )

        def _is_big_lot_size() -> bool:
            """
            Check if the lot size fits within the shift capacity.
            """
            return _lot_processing_time() > shift_time

        def _fit_within_predecessor_shift() -> bool:

            if is_setup_dependent:
                return (
                    _completion_time_in_current_machine() % shift_time
                    + s_times[
                        current_machine,
                        current_job + 1,
                        precedences[current_machine][-1][0] + 1,
                    ]
                    + p_times[current_machine, current_job]
                    * lot_sizes[n_lots * current_job + current_lot]
                    <= shift_time
                )

            else:
                return (
                    _completion_time_in_current_machine() % shift_time
                    + s_times[current_machine, current_job]
                    + p_times[current_machine, current_job]
                    * lot_sizes[n_lots * current_job + current_lot]
                    <= shift_time
                )

        def _fit_within_previous_shift() -> bool:
            """
            Check if the lot can fit within the previous same job lot shift time.
            """

            if is_setup_dependent:
                return (
                    _completion_time_in_previous_machine() % shift_time
                    + s_times[
                        current_machine,
                        current_job + 1,
                        precedences[current_machine][-1][0] + 1,
                    ]
                    + p_times[current_machine, current_job]
                    * lot_sizes[n_lots * current_job + current_lot]
                    <= shift_time
                )

            else:
                return (
                    _completion_time_in_previous_machine() % shift_time
                    + s_times[current_machine, current_job]
                    + p_times[current_machine, current_job]
                    * lot_sizes[n_lots * current_job + current_lot]
                    <= shift_time
                )

        def _lot_start_time_with_shifts():
            """
            Calculate the start time of the lot considering shift constraints.
            """

            if _is_big_lot_size():
                _lot_start_time()
            else:
                if _is_first_machine() and _is_empty_machine():
                    start_time[current_machine, current_job, current_lot] = 0

                elif _is_first_machine() and not _is_empty_machine():
                    last_completion = _completion_time_in_current_machine()
                    if _fit_within_predecessor_shift():
                        start_time[current_machine, current_job, current_lot] = (
                            last_completion
                        )

                    else:
                        last_shift = last_completion // shift_time
                        start_time[current_machine, current_job, current_lot] = (
                            shift_time * (last_shift + 1)
                        )

                elif not _is_first_machine() and _is_empty_machine():
                    last_completion = _completion_time_in_previous_machine()
                    if _fit_within_previous_shift():
                        start_time[current_machine, current_job, current_lot] = (
                            last_completion
                        )
                    else:
                        start_time[current_machine, current_job, current_lot] = (
                            shift_time * (last_completion // shift_time + 1)
                        )

                elif not _is_first_machine() and not _is_empty_machine():
                    last_completion_current_machine = (
                        _completion_time_in_current_machine()
                    )
                    last_completion_previous_machine = (
                        _completion_time_in_previous_machine()
                    )

                    if (
                        last_completion_previous_machine
                        >= last_completion_current_machine
                    ):
                        if _fit_within_previous_shift():
                            start_time[current_machine, current_job, current_lot] = (
                                last_completion_previous_machine
                            )
                        else:
                            start_time[current_machine, current_job, current_lot] = (
                                shift_time
                                * (last_completion_previous_machine // shift_time + 1)
                            )

                    else:
                        if _fit_within_predecessor_shift():
                            start_time[current_machine, current_job, current_lot] = (
                                last_completion_current_machine
                            )
                        else:
                            start_time[current_machine, current_job, current_lot] = (
                                shift_time
                                * (last_completion_current_machine // shift_time + 1)
                            )

        # cache problem parameters
        n_lots = self.problem_params.n_lots
        n_jobs = self.problem_params.n_jobs
        n_machines = self.problem_params.n_machines
        jobs = self.problem_params.jobs
        lots = self.problem_params.lots
        machines = self.problem_params.machines
        sequence = self.problem_params.seq
        p_times = self.problem_params.p_times
        s_times = self.problem_params.setup
        demand = self.problem_params.demand
        is_setup_dependent = self.problem_params.is_setup_dependent
        is_shift_constraints = self.problem_params.shift_constraints
        shift_time = self.problem_params.shift_time

        # Get the semi-encoded solution
        lot_sizes = _distribute_demand()
        lot_sequence = encoded_solution[1]
        semi_encoded_solution = [lot_sizes, lot_sequence]

        # Initialize containers for calculated data
        routes = _build_routes()
        precedences = _build_precedences()
        start_time = np.full((n_machines, n_jobs, n_lots), 0)
        completion_time = np.full((n_machines, n_jobs, n_lots), 0)
        makespan = 0

        # set penalty for shift constraints (lot sizes bigger than shift time)
        if is_shift_constraints:
            penalty_coefficient = 1e2
            max_lot_penalty = 0
            penalty = 0

        # decode logic
        for i, job_lot in enumerate(lot_sequence):  # For each lot
            if (
                lot_sizes[n_lots * job_lot[0] + job_lot[1]] != 0
            ):  # If the lot is not empty
                current_job = job_lot[0]
                current_lot = job_lot[1]
                current_machine = routes[(current_job, current_lot)][0]

                # Calculate the start time of the lot
                if is_shift_constraints:
                    _lot_start_time_with_shifts()
                else:
                    _lot_start_time()

                # Calculate the completion time of the lot
                if is_setup_dependent:
                    completion_time[current_machine, current_job, current_lot] = (
                        _lot_completion_time_setup_dependent()
                    )
                else:
                    completion_time[current_machine, current_job, current_lot] = (
                        _lot_completion_time_setup_independent()
                    )
                
                # Calculate penalty if shift constraints are considered
                if is_shift_constraints and _is_big_lot_size():
                    lot_penalty = (
                        completion_time[current_machine, current_job, current_lot]
                        % shift_time
                    )
                    if lot_penalty > max_lot_penalty:
                        max_lot_penalty = lot_penalty
                        penalty = penalty_coefficient * max_lot_penalty

                # Update precedences
                precedences[current_machine].append((current_job, current_lot))

                # Update lot route
                routes[(current_job, current_lot)].pop(0)

                # get makespan
                makespan = np.max(completion_time)

        return makespan, penalty, start_time, completion_time, semi_encoded_solution

    def get_fitness(self, decoded_solution: Any) -> float:
        # Implement the fitness calculation specific to Job Shop Scheduling
        pass


def decode_chromosome(chromosome, params: params.JobShopRandomParams):
    """
    Decodes a chromosome to a solution

    Args:
        chromosome: numpy array with the chromosome
        params: object of class JobShopRandomParams
        shifts: boolean to indicate if shift constraints are considered
        seq_dep_setup: boolean to indicate if setup times are sequence dependent

    Returns:
        y: numpy array with setup start times of all lots
        c: numpy array with completion times of all lots
        makespan: integer with makespan of the solution
    """

    shifts = params.shift_constraints
    seq_dep_setup = params.is_setup_dependent

    # Functions for active scheduling
    def distribute_demand():
        """
        Distributes a total demand into parts based on the given fractions
        such that the sum of the parts equals the total demand.

        Args:

        Returns:
            chromosome_lhs_m: numpy array with the demand distributed into lots
                (LHS of the chromosome modified)
        """
        chromosome_lhs_m = np.copy(chromosome[0])
        for job in params.jobs:
            total = np.sum(chromosome_lhs_m[n_lots * job : n_lots * (job + 1)])

            if total != 0:  # Avoid division by zero
                chromosome_lhs_m[n_lots * job : n_lots * (job + 1)] = (
                    chromosome_lhs_m[n_lots * job : n_lots * (job + 1)] / total
                )

                for lot in params.lots:
                    chromosome_lhs_m[n_lots * job + lot] = int(
                        chromosome_lhs_m[n_lots * job + lot] * params.demand[job]
                    )
                total_preliminary = sum(
                    chromosome_lhs_m[n_lots * job : n_lots * (job + 1)]
                )
                residual = params.demand[job] - total_preliminary

                if residual > 0:
                    for lot in params.lots:
                        chromosome_lhs_m[n_lots * job + lot] += 1
                        residual -= 1
                        if residual == 0:
                            break

            else:
                chromosome_lhs_m[n_lots * job : n_lots * (job + 1)] = (
                    int(params.demand[job]) / n_lots
                )
                chromosome_lhs_m[n_lots * job + n_lots - 1] = (
                    params.demand[job]
                ) - sum(chromosome_lhs_m[n_lots * job : n_lots * (job)])

        return chromosome_lhs_m

    def is_empty_machine() -> bool:
        if seq_dep_setup:
            if precedences[current_machine] == [
                -1
            ]:  # if is first lot processed in the machine
                precedences[current_machine]
                empty_machine = True
            else:
                empty_machine = False
        else:
            if precedences[current_machine] == []:  # if is first lot in the machine
                empty_machine = True
            else:
                empty_machine = False
        return empty_machine

    def is_first_machine() -> bool:
        if (
            current_machine == params.seq[current_job][0]
        ):  # if is first machine in the job route
            first_machine = True
        else:
            first_machine = False
        return first_machine

    def completion_in_previous_machine():
        previous_machine = params.seq[current_job][
            params.seq[current_job].index(current_machine) - 1
        ]  # previous machine in the job route
        next_start_time = c[previous_machine, current_job, current_lot]
        return next_start_time

    def completion_in_current_machine():
        predecessor = precedences[current_machine][
            -1
        ]  # predecessor in the current machine
        next_start_time = c[current_machine, predecessor[0], predecessor[1]]
        return next_start_time

    def lot_start_time():
        if is_first_machine() and is_empty_machine():
            y[current_machine, current_job, current_lot] = 0

        elif is_first_machine() and not is_empty_machine():
            y[current_machine, current_job, current_lot] = (
                completion_in_current_machine()
            )

        elif not is_first_machine() and is_empty_machine():
            y[current_machine, current_job, current_lot] = (
                completion_in_previous_machine()
            )

        elif not is_first_machine() and not is_empty_machine():
            y[current_machine, current_job, current_lot] = max(
                completion_in_current_machine(), completion_in_previous_machine()
            )

    # Functions for shift constraints

    def is_big_lotsize() -> bool:
        if seq_dep_setup:
            lot_size_time = (
                params.setup[
                    current_machine,
                    current_job + 1,
                    precedences[current_machine][-1][0] + 1,
                ]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
            )
        else:
            lot_size_time = (
                params.setup[current_machine, current_job]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
            )
        if lot_size_time > params.shift_time:
            return True
        else:
            return False

    def fit_within_predecessor_shift():
        if seq_dep_setup:
            if (
                completion_in_current_machine() % params.shift_time
                + params.setup[
                    current_machine,
                    current_job + 1,
                    precedences[current_machine][-1][0] + 1,
                ]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False
        else:
            if (
                completion_in_current_machine() % params.shift_time
                + params.setup[current_machine, current_job]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False

    def fit_within_previous_shift():
        if seq_dep_setup:
            if (
                completion_in_previous_machine() % params.shift_time
                + params.setup[
                    current_machine,
                    current_job + 1,
                    precedences[current_machine][-1][0] + 1,
                ]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False
        else:
            if (
                completion_in_previous_machine() % params.shift_time
                + params.setup[current_machine, current_job]
                + params.p_times[current_machine, current_job]
                * chromosome_lhs_m[n_lots * current_job + current_lot]
                <= params.shift_time
            ):
                return True
            else:
                return False

    def lot_start_time_with_shifts():
        if is_big_lotsize():
            lot_start_time()
        else:
            if is_first_machine() and is_empty_machine():
                y[current_machine, current_job, current_lot] = 0

            elif is_first_machine() and not is_empty_machine():
                last_completion = completion_in_current_machine()
                if fit_within_predecessor_shift():
                    y[current_machine, current_job, current_lot] = last_completion
                else:
                    last_shift = last_completion // params.shift_time
                    y[current_machine, current_job, current_lot] = params.shift_time * (
                        last_shift + 1
                    )

            elif not is_first_machine() and is_empty_machine():
                last_completion = completion_in_previous_machine()
                if fit_within_previous_shift():
                    y[current_machine, current_job, current_lot] = last_completion
                else:
                    y[current_machine, current_job, current_lot] = params.shift_time * (
                        last_completion // params.shift_time + 1
                    )

            elif not is_first_machine() and not is_empty_machine():
                last_completion_current_machine = completion_in_current_machine()
                last_completion_previous_machine = completion_in_previous_machine()

                if last_completion_previous_machine >= last_completion_current_machine:
                    if fit_within_previous_shift():
                        y[current_machine, current_job, current_lot] = (
                            last_completion_previous_machine
                        )
                    else:
                        y[current_machine, current_job, current_lot] = (
                            params.shift_time
                            * (
                                last_completion_previous_machine // params.shift_time
                                + 1
                            )
                        )

                else:
                    if fit_within_predecessor_shift():
                        y[current_machine, current_job, current_lot] = (
                            last_completion_current_machine
                        )
                    else:
                        y[current_machine, current_job, current_lot] = (
                            params.shift_time
                            * (last_completion_current_machine // params.shift_time + 1)
                        )

    # Initialize variables
    n_jobs = len(params.jobs)
    n_machines = len(params.machines)
    n_lots = len(params.lots)
    chromosome_lhs = chromosome[0]
    chromosome_rhs = chromosome[1]

    # decode left hand side of the chromosome if it is a float
    if type(chromosome_lhs[0]) is np.float64:
        chromosome_lhs_m = distribute_demand()
    else:
        chromosome_lhs_m = np.copy(chromosome_lhs)

    chromosome_mod = [chromosome_lhs_m, chromosome_rhs]

    # Do a dictionary to track route of each lot
    routes = {
        (job, lot): params.seq[job][:] for job in params.jobs for lot in params.lots
    }

    # Dictionary to track precedence in scheduling
    precedences = {}
    for machine in params.machines:
        if seq_dep_setup:
            precedences[machine] = [(-1, 0)]
        else:
            precedences[machine] = []

    # Arrays to store times of all sublots
    y = np.full((n_machines, n_jobs, n_lots), 0)  # setup start time
    c = np.full((n_machines, n_jobs, n_lots), 0)  # completion time

    # Schedule the jobs and get the makespan
    makespan = 0
    penalty_coefficient = 1e2
    max_lot_penalty = 0
    penalty = 0
    for i, job_lot in enumerate(chromosome_rhs):  # For each lot
        if (
            chromosome_lhs_m[n_lots * job_lot[0] + job_lot[1]] != 0
        ):  # If the lot is not empty
            current_job = job_lot[0]
            current_lot = job_lot[1]
            current_machine = routes[(current_job, current_lot)][0]

            # Calculate the start time and completion of the lot
            if shifts:
                lot_start_time_with_shifts()
            else:
                lot_start_time()

            # Calculate the completion time of the lot
            if seq_dep_setup:
                c[current_machine, current_job, current_lot] = (
                    y[current_machine, current_job, current_lot]
                    + params.setup[
                        current_machine,
                        current_job + 1,
                        precedences[current_machine][-1][0] + 1,
                    ]
                    + params.p_times[current_machine, current_job]
                    * chromosome_lhs_m[n_lots * current_job + current_lot]
                )
            else:
                c[current_machine, current_job, current_lot] = (
                    y[current_machine, current_job, current_lot]
                    + params.setup[current_machine, current_job]
                    + params.p_times[current_machine, current_job]
                    * chromosome_lhs_m[n_lots * current_job + current_lot]
                )

            # Update makespan
            if c[current_machine, current_job, current_lot] > makespan:
                makespan = c[current_machine, current_job, current_lot]

            if is_big_lotsize():
                lot_penalty = (
                    c[current_machine, current_job, current_lot] % params.shift_time
                )
                if lot_penalty > max_lot_penalty:
                    max_lot_penalty = lot_penalty
                    penalty = penalty_coefficient * max_lot_penalty

            # Update precedences
            precedences[current_machine].append((current_job, current_lot))

            # Update lot route
            routes[(current_job, current_lot)].pop(0)

    return makespan, penalty, y, c, chromosome_mod


def get_chromosome_start_times(chromosome_mod, params, c):
    """
    Calculates start times of all lots

    Args:
        chromosome: dataframe with solution parameters
        params: object of class JobShopRandomParams
        c: completion time of each lot

    Returns:
        x: numpy array with start time of each lot
    """
    chromosome_lhs_m = chromosome_mod[0]
    n_machines, n_jobs, n_lots = (
        len(params.machines),
        len(params.jobs),
        len(params.lots),
    )
    triple_mju = {
        (m, j, u)
        for m in range(n_machines)
        for j in range(n_jobs)
        for u in range(n_lots)
    }

    x = np.full((n_machines, n_jobs, n_lots), 0)  # start time
    for m, j, u in triple_mju:
        if c[m, j, u] > 0:
            x[m, j, u] = (
                c[m, j, u] - params.p_times[m, j] * chromosome_lhs_m[n_lots * j + u]
            )

    return x


def build_chromosome_results_df(chromosome_mod, y, x, c):
    """
    Builds a dataframe to show chromosome solution results

    Args:
        y: numpy array with setup start times of all lots
        c: numpy array with completion times of all lots
        x: numpy array with start time of each lot

    Returns:
        df: dataframme with solution results
    """
    # Reshape the 3D array to a 2D array
    # Shape will be (num_machines * num_jobs * num_lots, 1)
    num_machines, num_jobs, num_lots = y.shape
    s_start_time_2d = y.reshape(num_machines * num_jobs * num_lots, 1)
    start_time_2d = x.reshape(num_machines * num_jobs * num_lots, 1)
    completion_time_2d = c.reshape(num_machines * num_jobs * num_lots, 1)

    # Create a DataFrame
    df = pd.DataFrame(s_start_time_2d, columns=["setup_start_time"])

    # Add additional columns
    df["start_time"] = start_time_2d
    df["completion_time"] = completion_time_2d

    # Generate additional columns for machine, job, and lot
    df["machine"] = np.repeat(np.arange(num_machines), num_jobs * num_lots)
    df["job"] = np.tile(np.repeat(np.arange(num_jobs), num_lots), num_machines)
    df["lot"] = np.tile(np.arange(num_lots), num_machines * num_jobs)

    # Add the lot_size column based on the job and lot indices
    chromosome_lhs_m = chromosome_mod[0]
    df["lot_size"] = df.apply(
        lambda row: chromosome_lhs_m[(row["job"] * num_lots) + row["lot"]], axis=1
    )

    # Reorder columns if needed
    df = df[
        [
            "machine",
            "job",
            "lot",
            "setup_start_time",
            "start_time",
            "completion_time",
            "lot_size",
        ]
    ]

    # Filter out rows where completion_time is 0
    df_filtered = df[df["completion_time"] != 0]
    df_filtered = df_filtered.reset_index(drop=True)

    # Display the DataFrame
    print(df_filtered)

    return df_filtered


def get_dataframe_results(chromosome, params, shifts=False, seq_dep_setup=False):
    """
    Get the results of a chromosome in a dataframe

    Args:
        chromosome: numpy array with the chromosome
        params: object of class JobShopRandomParams
        shifts: boolean to indicate if shift constraints are considered
        seq_dep_setup: boolean to indicate if setup times are sequence dependent

    Returns:
        df_results: dataframe with the results of the chromosome
    """
    # Decode the chromosome
    _, _, y, c, chromosome_mod = decode_chromosome(chromosome, params)

    # Get start time of each lot
    x = get_chromosome_start_times(chromosome_mod, params, c)

    # Build dataframe with chromosome solution results
    df_results = build_chromosome_results_df(chromosome_mod, y, x, c)

    return df_results


def main():
    # Generate Job Shop Random Parameters
    my_params = params.JobShopRandomParams(config_path="config/jobshop/js_params_01.yaml")
    my_params.print_jobshop_params()

    # Example chromosome (could be random generated but for testing purposes is fixed)
    chromosome = [
        np.array(
            [
                0.33893225,
                0.39426217,
                0.567908,
                0.67242174,
                0.37249561,
                0.85690449,
                0.62416912,
                0.51083418,
                0.21353647,
            ]
        ),
        [
            (1, 2),
            (2, 2),
            (0, 0),
            (0, 2),
            (0, 0),
            (1, 1),
            (1, 0),
            (0, 0),
            (1, 1),
            (0, 1),
            (1, 0),
            (0, 2),
            (2, 1),
            (1, 2),
            (0, 1),
            (2, 0),
            (0, 2),
            (0, 1),
        ],
    ]

    # Decode the chromosome
    decoder = JobShopDecoder(my_params)
    makespan, penalty, y, c, chromosome_mod = decoder.decode(chromosome)
    
    # decode_chromosome(
    #     chromosome, my_params, shifts=True, seq_dep_setup=True
    # )
    print("makespan: \n", makespan)
    print("penalty: \n", penalty)
    print("setup start times: \n", y)
    print("completion times: \n", c)

    # Get start time of each lot
    x = get_chromosome_start_times(chromosome_mod, my_params, c)
    print("Start times: \n", x)

    # Build dataframe with chromosome solution results
    df_results = build_chromosome_results_df(chromosome_mod, y, x, c)

    # Plot gantt
    demand = my_params.demand
    plot.gantt(df_results, my_params, demand)
    print("nice job")


if __name__ == "__main__":
    main()
