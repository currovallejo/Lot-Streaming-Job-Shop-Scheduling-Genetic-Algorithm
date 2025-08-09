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

# --------- src/ MODULES ---------
from ..import params

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
            Tuple containing
            - makespan (int): The total time to complete all jobs
            - penalty (int): Penalty for shift constraints
            - start_time (np.ndarray): SETUP Start times of each lot on each machine
            - completion_time (np.ndarray): Completion times of each lot on each machine
            - semi_encoded_solution (List[np.ndarray]): Semi-encoded solution containing lot sizes and lot sequence
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
            return (
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
        penalty = 0

        # set penalty for shift constraints (lot sizes bigger than shift time)
        if is_shift_constraints:
            penalty_coefficient = 1e2
            max_lot_penalty = 0

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

    def get_fitness(self, encoded_solution: Any) -> float:
        # Implement the fitness calculation specific to Job Shop Scheduling
        return self.decode(encoded_solution)[0] + self.decode(encoded_solution)[1]

    def get_lot_processing_start_times(
        self, semi_encoded_solution: list, completion_time: np.ndarray
    ) -> np.ndarray:
        """
        Get the start times of each lot based on the encoded solution.

        Args:
            semi_encoded_solution: chromosome containing lot sizes (int) and lot sequence
            completion_time: numpy array with completion times of each lot

        Returns:
            start_times: numpy array with start times of each lot
        """
        lot_sizes = semi_encoded_solution[0]
        p_times = self.problem_params.p_times
        n_machines, n_jobs, n_lots = (
            len(self.problem_params.machines),
            len(self.problem_params.jobs),
            len(self.problem_params.lots),
        )
        set_mju = {
            (m, j, u)
            for m in range(n_machines)
            for j in range(n_jobs)
            for u in range(n_lots)
        }

        lot_start_times = np.full((n_machines, n_jobs, n_lots), 0)  # start time
        for m, j, u in set_mju:
            if completion_time[m, j, u] > 0:
                lot_start_times[m, j, u] = (
                    completion_time[m, j, u] - p_times[m, j] * lot_sizes[n_lots * j + u]
                )

        return lot_start_times

    def build_schedule_times_df(
        self,
        semi_encoded_solution: list,
        setup_start_times: np.ndarray,
        lot_start_times,
        completion_times: np.ndarray,
    ) -> pd.DataFrame:
        """
        Build a DataFrame with the schedule times for each lot.

        Args:
            semi_encoded_solution: chromosome containing lot sizes (int) and lot sequence
            setup_start_times: numpy array with setup start times of each lot
            lot_start_times: numpy array with start times of each lot
            completion_times: numpy array with completion times of each lot

        Returns:
            schedule_df: DataFrame with columns for job, lot, machine, setup start time, start time, and completion time
        """
        n_jobs = self.problem_params.n_jobs
        n_lots = self.problem_params.n_lots
        n_machines = self.problem_params.n_machines
        lot_sizes = semi_encoded_solution[0]

        # flatten the 3D arrays to 2D arrays
        setup_start_times_flat = setup_start_times.reshape(
            n_machines * n_jobs * n_lots, 1
        )
        lot_start_times_flat = lot_start_times.reshape(n_machines * n_jobs * n_lots, 1)
        completion_times_flat = completion_times.reshape(
            n_machines * n_jobs * n_lots, 1
        )

        df = pd.DataFrame(setup_start_times_flat, columns=["setup_start_time"])
        df["start_time"] = lot_start_times_flat
        df["completion_time"] = completion_times_flat
        df["machine"] = np.repeat(np.arange(n_machines), n_jobs * n_lots)
        df["job"] = np.tile(np.repeat(np.arange(n_jobs), n_lots), n_machines)
        df["lot"] = np.tile(np.arange(n_lots), n_machines * n_jobs)
        df["lot_size"] = df.apply(
            lambda row: lot_sizes[(row["job"] * n_lots) + (row["lot"])], axis=1
        )

        # Reorder columns
        schedule_df = df[
            [
                "job",
                "lot",
                "machine",
                "setup_start_time",
                "start_time",
                "completion_time",
                "lot_size",
            ]
        ]

        # Filter out rows where completion time is zero
        schedule_df = schedule_df[schedule_df["completion_time"] > 0].reset_index(
            drop=True
        )

        return schedule_df

    def get_schedule_df_from_solution(self, encoded_solution: list) -> pd.DataFrame:
        """
        Get the schedule DataFrame from the encoded solution.

        Args:
            encoded_solution: Encoded solution (chromosome)

        Returns:
            schedule_df: DataFrame with the schedule times for each lot
        """
        _, _, setup_start_times, completion_times, semi_encoded_solution = self.decode(
            encoded_solution
        )
        lot_start_times = self.get_lot_processing_start_times(
            semi_encoded_solution, completion_times
        )
        return self.build_schedule_times_df(
            semi_encoded_solution, setup_start_times, lot_start_times, completion_times
        )
