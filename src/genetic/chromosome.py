"""
Chromosome generation module for Lot Streaming Job Shop Scheduling Problem.

This module implements chromosome generators for the genetic algorithm that solves
the Lot Streaming Job Shop Scheduling Problem. It provides classes to generate
random chromosomes consisting of left-hand side (lot sizes) and right-hand side
(job-lot operation sequences) components used in the evolutionary optimization process.

Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
"""

import random
import numpy as np

from jobshop import JobShopRandomParams
from shared.types import Chromosome


class LHSGenerator:
    """Generate the left-hand side of the chromosome, which is a numpy array
    representing the size of each lot.
    """

    def __init__(self, params: JobShopRandomParams):
        self.params = params

    def generate(self) -> np.ndarray:
        """Generate a numpy array with random float values between 0 and 1,
        representing the size of each lot.
        """
        lhs = np.array(
            [random.random() for job in self.params.jobs for lot in self.params.lots]
        )
        return lhs


class RHSGenerator:
    """Generate the right-hand side of the chromosome, which is a list of tuples
    representing (job, lot) pairs for each machine operation.
    """

    def __init__(self, params: JobShopRandomParams):
        self.params = params

    def generate(self) -> list[tuple[int, int]]:
        """Generate a shuffled list of (job, lot) tuples for each machine operation."""
        rhs = [
            (job, lot)
            for job in self.params.jobs
            for lot in self.params.lots
            for machine in self.params.seq[job]
        ]
        random.shuffle(rhs)
        return rhs


class ChromosomeGenerator:
    """Generate a random chromosome for the Lot Streaming Job Shop Scheduling
    Problem.
    """

    def __init__(self, params: JobShopRandomParams):
        self.params = params
        self.lhs_generator = LHSGenerator(params)
        self.rhs_generator = RHSGenerator(params)

    def generate(self) -> Chromosome:
        """Generate a random chromosome consisting of a left-hand side (LHS) and
        right-hand side (RHS).

        Returns:
            Chromosome: A list containing the LHS (numpy array) and RHS (list of tuples)
        """
        lhs = self.lhs_generator.generate()
        rhs = self.rhs_generator.generate()

        return (lhs, rhs)
