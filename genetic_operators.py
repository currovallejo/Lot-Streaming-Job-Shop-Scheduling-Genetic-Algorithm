"""
Created on Aug 09 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: genetic_operators.py - Genetic Operators for GA
"""

# --------- LIBRARIES ---------
import numpy as np
import random
from deap import tools

# --------- OTHER PYTHON FILES USED ---------
import chromosome_generator
import params


# --------- GENETIC OPERATORS ---------
# Cross-over operators
def spc1_crossover(ind1, ind2):
    """
    Perform SPC-1 crossover by exchanging the portions to the left of the crossover
    point

    Args:
        ind1: First individual (1D numpy array).
        ind2: Second individual (1D numpy array).

    Returns:
        Two individuals after crossover.
    """
    # Generate a random crossover point
    crossover_point = random.randint(
        1, len(ind1) - 1
    )  # Avoid 0 to ensure at least some crossover

    # Perform the crossover
    # Swap segments from the start to the crossover point
    new_ind1 = np.copy(ind1)
    new_ind2 = np.copy(ind2)

    new_ind1[:crossover_point] = ind2[:crossover_point]
    new_ind2[:crossover_point] = ind1[:crossover_point]

    return new_ind1, new_ind2


def spc2_crossover(ind1, ind2):
    """
    Perform SPC-2 crossover by exchanging the portions to the right of the crossover
    point

    Args:
        ind1 (np.array): The first parent individual.
        ind2 (np.array): The second parent individual.

    Returns:
        np.array, np.array: Two offspring individuals resulting from the crossover.
    """
    # Determine the crossover point
    crossover_point = np.random.randint(1, len(ind1))

    # Copy the individuals to create offspring
    new_ind1 = np.copy(ind1)
    new_ind2 = np.copy(ind2)

    # Perform the crossover by exchanging the portions to the right of the crossover
    # point
    new_ind1[crossover_point:] = ind2[crossover_point:]
    new_ind2[crossover_point:] = ind1[crossover_point:]

    return new_ind1, new_ind2


def cxPartialyMatched_mod(ind1, ind2):
    """
    Executes a partially matched crossover (PMX) on the input individuals.
    Individuals are expected to be list of tuples.
    Individuals are transformed to list of integers for the crossover and then back to
    list of tuples.

    Args:
        ind1: The first individual participating in the crossover.
        ind2: The second individual participating in the crossover.

    Returns:
        Two individuals after crossover.
    """
    # Convert the individual from list of tuples to list of integers
    parent1, parent2 = tuple_lists_to_int_lists(ind1, ind2)

    # Perform the crossover
    offspring1, offspring2 = tools.cxPartialyMatched(parent1, parent2)

    # Convert the offsprings from list of integers to list of tuples
    offspring1, offspring2 = int_lists_to_tuple_lists(offspring1, offspring2)

    return offspring1, offspring2


def cxOrdered_mod(ind1, ind2):
    # Convert the individual from list of tuples to list of integers
    parent1, parent2 = tuple_lists_to_int_lists(ind1, ind2)

    # Perform the crossover
    offspring1, offspring2 = tools.cxOrdered(parent1, parent2)

    # Convert the offsprings from list of integers to list of tuples
    offspring1, offspring2 = int_lists_to_tuple_lists(offspring1, offspring2)

    return offspring1, offspring2


def tuple_lists_to_int_lists(ind1, ind2):
    parent1 = []
    parent2 = []
    for i in range(len(ind1)):
        parent1.append(3 * ind1[i][0] + ind1[i][1])
        parent2.append(3 * ind2[i][0] + ind2[i][1])

    return parent1, parent2


def int_lists_to_tuple_lists(offspring1, offspring2):
    for i in range(len(offspring1)):
        offspring1[i] = (offspring1[i] // 3, offspring1[i] % 3)
        offspring2[i] = (offspring2[i] // 3, offspring2[i] % 3)

    return offspring1, offspring2


# def cxPartialyMatched_mod(ind1, ind2):
#     """MODIFICATION FOR LOT STREAMING JOB SHOP SCHEDULING PROBLEM GA
#     Executes a partially matched crossover (PMX) on the input individuals.
#     The two individuals are modified in place. This crossover expects
#     :term:`sequence` individuals of indices, the result for any other type of
#     individuals is unpredictable.

#     Args:
#         :param ind1: The first individual participating in the crossover.
#         :param ind2: The second individual participating in the crossover.

#     Returns:
#         A tuple of two individuals.

#     """
#     # Step 1: Initialize offspring
#     size = len(ind1)
#     offspring1 = [None] * size
#     offspring2 = [None] * size

#     # Step 2: Choose crossover points
#     cxpoint1 = random.randint(0, size)
#     cxpoint2 = random.randint(0, size - 1)
#     if cxpoint2 >= cxpoint1:
#         cxpoint2 += 1
#     else:
#         # Swap the two cx points
#         cxpoint1, cxpoint2 = cxpoint2, cxpoint1

#     print(cxpoint1, cxpoint2)

#     # Step 3: Copy segments between crossover points
#     offspring1[cxpoint1:cxpoint2+1] = ind1[cxpoint1:cxpoint2+1]
#     offspring2[cxpoint1:cxpoint2+1] = ind2[cxpoint1:cxpoint2+1]

#     # Step 4: Mapping for fixing the offspring
#     def create_mapping(from_parent, to_parent):
#         mapping = {}
#         for i in range(cxpoint1, cxpoint2 + 1):
#             mapping[from_parent[i]] = to_parent[i]
#         return mapping

#     def repair_offspring(offspring, parent, mapping):
#         for i in range(size):
#             if offspring[i] is None:
#                 value = parent[i]
#                 while value in mapping:
#                     value = mapping[value]
#                 offspring[i] = value

#     # Create mappings
#     mapping1 = create_mapping(ind1, ind2)
#     mapping2 = create_mapping(ind2, ind1)

#     # Repair offspring
#     repair_offspring(offspring1, ind2, mapping1)
#     repair_offspring(offspring2, ind1, mapping2)

#     return offspring1, offspring2

# def cxPartialyMatched_mo(ind1, ind2):
#     size = len(ind1)
#     p1 = {}
#     p2 = {}

#     # Create mappings based on tuple values
#     for i in range(size):
#         p1[ind1[i]] = i
#         p2[ind2[i]] = i

#     # Initialize offspring arrays
#     child1 = [None] * size
#     child2 = [None] * size

#     # Fill in the offspring based on the crossover logic
#     for i in range(size):
#         if ind1[i] in p2:
#             child1[i] = ind2[p2[ind1[i]]]
#         if ind2[i] in p1:
#             child2[i] = ind1[p1[ind2[i]]]

#     return child1, child2


# Mutation operators
def sstm_mutation(chromosome, mutation_prob, delta_max):
    """
    Apply Sublot Step Mutation (SStM) mutation operator to a chromosome.

    Args:
        chromosome (np.array): The chromosome array where mutation will be applied.
        mutation_prob (float): Probability of mutation for each gene.
        delta_max (float): Maximum step size for mutation.

    Returns:
        np.array: The mutated chromosome.
    """
    # Ensure the chromosome is a NumPy array
    chromosome = np.asarray(chromosome)

    # Iterate over each gene in the chromosome
    for i in range(len(chromosome)):
        # Determine if the gene will be mutated
        if np.random.rand() < mutation_prob:
            # Calculate the step amount
            delta = delta_max * np.random.rand()

            # Decide mutation direction (up or down)
            if np.random.rand() < 0.5:
                # Increase the value
                chromosome[i] = min(1, chromosome[i] + delta)
            else:
                # Decrease the value
                chromosome[i] = max(0, chromosome[i] - delta)

    return chromosome


def main():
    # Define parameters
    my_params = params.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)

    # Generate two random individuals
    ind1 = chromosome_generator.generate_chromosome(my_params)
    ind2 = chromosome_generator.generate_chromosome(my_params)

    # Perform SPC-1 crossover
    offspring1, offspring2 = cxPartialyMatched_mod(ind1[1], ind2[1])

    # Print the results
    print("Parent 1:")
    print(ind1[1])
    print("\nParent 2:")
    print(ind2[1])
    print("\nOffspring 1:")
    print(offspring1)
    print("\nOffspring 2:")
    print(offspring2)


if __name__ == "__main__":
    main()
