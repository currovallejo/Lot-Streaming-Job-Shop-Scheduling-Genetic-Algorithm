"""
Created on Aug 09 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: GA.py - Genetic Algorithm
"""
#--------- LIBRARIES ---------
import numpy as np
import random


#--------- OTHER PYTHON FILES USED ---------
import chromosome_generator
import params


# --------- GENETIC OPERATORS ---------
def spc1_crossover(ind1, ind2):
    """
    Performs SPC-1 crossover on the LHS segment of two individuals.
    
    Args:
        ind1: First individual (list of two numpy arrays).
        ind2: Second individual (list of two numpy arrays).
    
    Returns:
        Two new individuals with the LHS segment crossed over.
    """
    # Ensure both individuals have the same shape in the LHS segment
    lhs1, rhs1 = ind1
    lhs2, rhs2 = ind2

    # Check dimensions
    if lhs1.shape != lhs2.shape:
        raise ValueError("LHS segments of both individuals must have the same shape.")
    
    num_jobs = lhs1.shape[0]  # Number of jobs (rows)
    num_lots = lhs1.shape[1]  # Number of lots (columns)
    
    # Generate a random crossover point
    crossover_point = random.randint(1, num_lots*num_jobs - 1)
    print(crossover_point)
    
    # Convert the flat index to 2D coordinates
    row, col = divmod(crossover_point, num_lots)
    print(row, col)
    
    # Create new LHS segments by swapping parts before the crossover point
    new_lhs1 = np.copy(lhs1)
    new_lhs2 = np.copy(lhs2)
    
    new_lhs1[:row, :col], new_lhs2[:row, :col] = lhs2[:row, :col], lhs1[:row, :col]
    
    # Create new individuals with swapped LHS segments
    offspring1 = [new_lhs1, rhs1]
    offspring2 = [new_lhs2, rhs2]
    
    return offspring1, offspring2

def main():
    # Define parameters
    my_params = params.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)
    
    # Generate two random individuals
    ind1 = chromosome_generator.generate_chromosome(my_params)
    ind2 = chromosome_generator.generate_chromosome(my_params)
    
    # Perform SPC-1 crossover
    offspring1, offspring2 = spc1_crossover(ind1, ind2)
    
    # Print the results
    print("Parent 1:")
    print(ind1[0])
    print("\nParent 2:")
    print(ind2[0])
    print("\nOffspring 1:")
    print(offspring1[0])
    print("\nOffspring 2:")
    print(offspring2[0])

if __name__ == "__main__":
    main()