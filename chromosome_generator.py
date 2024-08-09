"""
Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: chromosome-generator.py - generation of random chromosomes
"""

#--------- LIBRARIES ---------
import numpy as np
import random

#--------- OTHER PYTHON FILES USED ---------
import params

#--------- CHROMOSOME GENERATOR ---------

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
    chromosome_lhs = []

        # Create a random number generator instance
    rng = np.random.default_rng()

    for job in params.jobs:
        chromosome_lhs_j = []
        for lot in params.lots:
                chromosome_lhs_j.append(random.random())
        
        chromosome_lhs.append(np.array(chromosome_lhs_j))
    chromosome_lhs = np.array(chromosome_lhs)
    
    # Generate chromosome right-hand side
    sublots = []
    for j in params.jobs:
        for u in params.lots:
            for m in params.seq[j]:
                sublots.append(np.array([j,u]))

    random.shuffle(sublots)    
    chromosome_rhs = np.array(sublots)

    # Concatenate both sides 
    chromosome =  [chromosome_lhs, chromosome_rhs]

    return chromosome

#--------- MAIN ---------
def main():
    my_random_params = params.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)
    chromosome = generate_chromosome(my_random_params)
    print(chromosome)

if __name__=="__main__":
    main()