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


def cxJobLevel(ind1, ind2):
    """
    cxJobLevel operator for chromosomes where each gene is a tuple (job, lot).
    The operator copies all lots of a chosen job from parent1 to child1 and parent2 to
    child2
    while keeping their positions. The remaining genes are filled from the other parent
    without duplicating the job.

    Args:
        parent1: list of tuples (job, lot), representing a parent chromosome.
        parent2: list of tuples (job, lot), representing a parent chromosome.

    Returns:
        child1: list of tuples (job, lot), representing the first offspring.
        child2: list of tuples (job, lot), representing the second offspring.
    """

    # Step 1: Choose a job from parent 1 arbitrarily
    chosen_job1 = random.choice([gene[0] for gene in ind1])

    # Step 2: Copy all operations of the chosen job from parent 1 to child 1,
    # preserving their positions
    child1 = [None] * len(ind1)
    for i, gene in enumerate(ind1):
        if gene[0] == chosen_job1:
            child1[i] = gene

    # Step 3: Complete child 1 with the remaining operations from parent 2, preserving
    # the order and avoiding duplicates
    child1_jobs = set(gene for gene in child1 if gene is not None)
    child1_index = 0
    for gene in ind2:
        while child1_index < len(child1) and child1[child1_index] is not None:
            child1_index += 1
        if gene not in child1_jobs:
            child1[child1_index] = gene
            child1_index += 1

    # Step 4: Repeat the process for child 2, but choose the job from parent 2 and fill
    # the rest with operations from parent 1
    chosen_job2 = random.choice([gene[0] for gene in ind2])
    child2 = [None] * len(ind2)
    for i, gene in enumerate(ind2):
        if gene[0] == chosen_job2:
            child2[i] = gene

    child2_jobs = set(gene for gene in child2 if gene is not None)
    child2_index = 0
    for gene in ind1:
        while child2_index < len(child2) and child2[child2_index] is not None:
            child2_index += 1
        if gene not in child2_jobs:
            child2[child2_index] = gene
            child2_index += 1

    return child1, child2


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


def mutShuffleIndexes_mod(individual):
    """
    Shuffle the indexes of an individual with a given probability.

    Args:
        individual (np.array): The individual to be mutated.
        indpb (float): The probability of shuffling each index.

    Returns:
        np.array: The mutated individual.
    """
    individual_listed = []
    for i in range(len(individual)):
        individual_listed.append(3 * individual[i][0] + individual[i][1])

    # Perform the mutation
    mutation, *_ = tools.mutShuffleIndexes(individual_listed, indpb=0.2)

    for i in range(len(mutation)):
        mutation[i] = (mutation[i] // 3, mutation[i] % 3)

    return mutation


def main():

    # Example usage cxJobLevel
    parent1 = [
        (1, 1),
        (2, 1),
        (1, 2),
        (3, 1),
        (2, 2),
        (1, 1),
        (1, 1),
        (2, 1),
        (1, 2),
        (3, 1),
        (2, 2),
    ]
    parent2 = [
        (2, 1),
        (1, 1),
        (3, 1),
        (1, 2),
        (2, 1),
        (1, 2),
        (3, 1),
        (2, 2),
        (2, 2),
        (1, 1),
        (1, 1),
    ]

    child1, child2 = cxJobLevel(parent1, parent2)
    print("Parent 1:", parent1)
    print("Parent 2:", parent2)
    print("Child 1:", child1)
    print("Child 2:", child2)

    # Define parameters
    my_params = params.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)

    # Generate two random individuals
    ind1 = chromosome_generator.generate_chromosome(my_params)
    ind2 = chromosome_generator.generate_chromosome(my_params)

    # Perform SPC-1 crossover
    offspring1, offspring2 = cxJobLevel(ind1[1], ind2[1])

    # Print the results
    print("Parent 1:", ind1[1])
    print("Parent 2:", ind2[1])
    print("Child 1:", offspring1)
    print("Child 2:", offspring2)


if __name__ == "__main__":
    main()
