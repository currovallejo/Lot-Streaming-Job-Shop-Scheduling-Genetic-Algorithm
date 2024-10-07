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


# --------- OTHER PYTHON FILES USED ---------
import params

# --------- CHROMOSOME GENERATOR ---------


def generate_chromosome(params):
    """
    Generates a random chromosome given the parameters of the problem

    Args:
        params: object of class JobShopRandomParams
        demand: dictionary with the demand of each job

    Returns:
        chromosome: numpy array with the chromosome
    """
    # Generate chromosome left-hand side
    chromosome_lhs = np.array([
        random.random()
        for job in params.jobs
        for lot in params.lots
    ])

    # Generate chromosome right-hand side (sublots)
    chromosome_rhs = [
        (job, lot)
        for job in params.jobs
        for lot in params.lots
        for machine in params.seq[job]
    ]

    random.shuffle(chromosome_rhs)

    # Concatenate both sides
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
