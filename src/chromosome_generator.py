"""
Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: chromosome-generator.py - generation of random chromosomes
"""

# --------- LIBRARIES ---------
import random
import numpy as np


# --------- src/ MODULES ---------
from . import params

# --------- CHROMOSOME GENERATOR ---------


def generate_chromosome(
    params: params.JobShopRandomParams,
) -> list[np.ndarray, list[tuple[int, int]]]:
    """Generate a random chromosome for the Job Shop Scheduling Problem.
    Args:
        params (JobShopRandomParams): Parameters for the job shop problem.

    Returns:
        list: A chromosome represented as a list containing two elements:
            - [0] (numpy.ndarray): Left-hand side with random float values between 0
            and 1, representing the size of each lot.
            - [1] (list): Right-hand side with shuffled tuples of (job, lot) for each
            machine operation
    """
    # Generate chromosome left-hand side
    chromosome_lhs = np.array(
        [random.random() for job in params.jobs for lot in params.lots]
    )

    # Generate chromosome right-hand side (sublots)
    chromosome_rhs = [
        (job, lot)
        for job in params.jobs
        for lot in params.lots
        for machine in params.seq[job]
    ]

    random.shuffle(chromosome_rhs)

    # Concatenate both sides as a list for modification permission
    chromosome = [chromosome_lhs, chromosome_rhs]

    return chromosome


# --------- MAIN ---------
def main():
    """Working example of the chromosome generator"""

    my_random_params = params.JobShopRandomParams(
        n_machines=3, n_jobs=3, n_lots=3, seed=4
    )
    chromosome = generate_chromosome(my_random_params)
    print(chromosome)


if __name__ == "__main__":
    main()
