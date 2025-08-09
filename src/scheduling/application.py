from __future__ import annotations
import pandas as pd
from .services import JobShopDecoder, build_schedule_times_df_from_ops
from .mappers import OperationAssembler
from ..domain import ScheduledOperation
from ..params import JobShopRandomParams
from ..shared.types import Chromosome


class Scheduler:
    """
    Scheduler class for managing scheduling operations.
    """

    def __init__(self, problem_params: JobShopRandomParams):
        self.problem_params = problem_params
        self._assembler = OperationAssembler(problem_params)
        self._decoder = JobShopDecoder(problem_params)

    def decode(self, encoded_solution: Chromosome) -> tuple:
        """
        Decode the encoded solution to a semi-active schedule.

        Args:
            - encoded_solution: Chromosome containing lot sizes and sequence.

        Returns:
            - makespan (int): The total time to complete all jobs
            - penalty (int): Penalty for shift constraints
            - start_time (np.ndarray): SETUP Start times of each lot on each machine
            - completion_time (np.ndarray): Completion times of each lot on each machine
            - semi_encoded_solution (List[np.ndarray]): Semi-encoded solution containing lot sizes and lot sequence
        """
        solution_decoded = self._decoder.decode(encoded_solution)
        (
            self.makespan,
            self.penalty,
            self.start_time,
            self.completion_time,
            self.semi_encoded_solution,
        ) = solution_decoded
        return solution_decoded

    def get_fitness(self, encoded_solution: Chromosome) -> int:
        """
        Calculate the fitness of the current schedule.

        Returns:
            - fitness (int): The fitness value based on makespan and penalty.
        """
        solution_decoded = self._decoder.decode(encoded_solution)
        return solution_decoded[0] + solution_decoded[1]

    def build_operations(
        self, encoded_solution: Chromosome
    ) -> list[ScheduledOperation]:
        """
        Build domain operations from the semi-encoded solution.

        Returns:
            - List of ScheduledOperation
        """
        solution_decoded = self._decoder.decode(encoded_solution)
        return self._assembler.from_tensors(
            semi_encoded_solution=solution_decoded[4],
            setup_start_times=solution_decoded[2],
            completion_times=solution_decoded[3],
        )

    def build_schedule_df(self, ops: list[ScheduledOperation]) -> pd.DataFrame:
        """
        Build a schedule DataFrame from domain operations.

        Args:
            - ops: List of ScheduledOperation

        Returns:
            - DataFrame containing the schedule times.
        """
        return build_schedule_times_df_from_ops(ops)
