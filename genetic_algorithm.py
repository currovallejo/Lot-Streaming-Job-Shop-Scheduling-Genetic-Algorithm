"""
Created on Aug 08 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: genetic_algorithm.py - Genetic Algorithm
"""

# --------- LIBRARIES ---------
from deap import base, creator, tools
import time

# --------- OTHER PYTHON FILES USED ---------
import chromosome_generator
import params
import decoder
import plot
import copy
from genetic_operators import *


# --------- GENETIC OPERATORS ---------
# Cross-over operators
def spc1_crossover_lhs(ind1, ind2):
    new_ind1 = copy.deepcopy(ind1)
    new_ind2 = copy.deepcopy(ind2)
    new_ind1_lhs, new_ind2_lhs = spc1_crossover(ind1[0], ind2[0])
    new_ind1[0] = new_ind1_lhs
    new_ind2[0] = new_ind2_lhs
    return new_ind1, new_ind2


def spc2_crossover_lhs(ind1, ind2):
    new_ind1 = copy.deepcopy(ind1)
    new_ind2 = copy.deepcopy(ind2)
    new_ind1_lhs, new_ind2_lhs = spc2_crossover(ind1[0], ind2[0])
    new_ind1[0] = new_ind1_lhs
    new_ind2[0] = new_ind2_lhs
    return new_ind1, new_ind2


def cxPartiallyMatched_rhs(ind1, ind2):
    new_ind1 = copy.deepcopy(ind1)
    new_ind2 = copy.deepcopy(ind2)
    rhs_1 = ind1[1]
    rhs_2 = ind2[1]
    new_ind1_rhs, new_ind2_rhs = cxPartialyMatched_mod(rhs_1, rhs_2)
    new_ind1[1] = new_ind1_rhs
    new_ind2[1] = new_ind2_rhs
    return new_ind1, new_ind2


def cxOrdered_rhs(ind1, ind2):
    new_ind1 = copy.deepcopy(ind1)
    new_ind2 = copy.deepcopy(ind2)
    rhs_1 = ind1[1]
    rhs_2 = ind2[1]
    new_ind1_rhs, new_ind2_rhs = cxOrdered_mod(rhs_1, rhs_2)
    new_ind1[1] = new_ind1_rhs
    new_ind2[1] = new_ind2_rhs
    return new_ind1, new_ind2


def cxJobLevel_rhs(ind1, ind2):
    new_ind1 = copy.deepcopy(ind1)
    new_ind2 = copy.deepcopy(ind2)
    rhs_1 = ind1[1]
    rhs_2 = ind2[1]
    new_ind1_rhs, new_ind2_rhs = cxJobLevel(rhs_1, rhs_2)
    new_ind1[1] = new_ind1_rhs
    new_ind2[1] = new_ind2_rhs
    return new_ind1, new_ind2


# Mutation operators
def sstm_mutation_lhs(ind):
    new_ind = copy.deepcopy(ind)
    new_ind_lhs = sstm_mutation(ind[0], 0.2, 0.5)
    new_ind[0] = new_ind_lhs
    return new_ind


def mutShuffleIndexes_rhs(individual):
    new_ind = copy.deepcopy(individual)
    rhs = individual[1]
    new_rhs = mutShuffleIndexes_mod(rhs)
    new_ind[1] = new_rhs
    return new_ind


# --------- GENETIC ALGORITHM ---------


def create_individual_factory(params):
    def create_individual():
        return chromosome_generator.generate_chromosome(params)

    return create_individual


def run(
    params: params.JobShopRandomParams,
    population_size: int,
    num_generations: int,
    plotting=True,
):
    """
    Genetic algorithm for LSJSP problem developed with DEAP package

    Args:
        params: parameters of the problem
        population_size: number of individuals in the population
        num_generations: number of generations
        shifts: boolean, True if shifts are considered
        seq_dep_setup: boolean, True if sequence dependent setup times are considered
        plotting: boolean, True if Gantt chart is plotted

    Returns:
        best_makespan: best fitness value found
        df_results: parameters of best solution found
    """

    # --------- DEAP SETUP ---------
    # Step 1: Define the fitness function
    shifts = params.shift_constraints
    seq_dep_setup = params.is_setup_dependent
    js_decoder = decoder.JobShopDecoder(params)

    def evalFitness(individual):
        # return as a tuple needed by DEAP
        return (js_decoder.get_fitness(encoded_solution=individual),)

    # Step 2: Create the individual and population
    # Define the fitness class for minimization
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # Define the individual class
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create the toolbox
    toolbox = base.Toolbox()

    # Factory function for individual creation
    create_individual = create_individual_factory(params)

    # Register the individual creation function
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, create_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Step 3: Register genetic operators
    # Crossover
    toolbox.register("spc1", spc1_crossover_lhs)
    toolbox.register("spc2", spc2_crossover_lhs)
    toolbox.register("PMX", cxPartiallyMatched_rhs)
    toolbox.register("OX", cxOrdered_rhs)
    toolbox.register("JL", cxJobLevel_rhs)

    # Mutation
    toolbox.register("sstm", sstm_mutation_lhs)
    toolbox.register("MSI", mutShuffleIndexes_rhs)

    # Selection
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluation
    toolbox.register("evaluate", evalFitness)

    # Step 4: Create the population
    start_population = time.time()
    population = toolbox.population(n=population_size)
    end_population = time.time()
    print(
        "Population of ",
        population_size,
        " individuals created in ",
        end_population - start_population,
        " seconds",
    )

    # Step 5: Define the genetic algorithm
    # Probabilities for genetic operators
    SPC1, SPC2, PMX, OX, JL = 0.8, 0.8, 0.8, 0.8, 0.8
    SSTM, MSI = 0.2, 0.2

    # Diversification threshold for population diversity
    diversification_threshold = 10  # 30 generations without improvement
    no_improvement_generations = 0
    best_fitness = float("inf")  # tracking

    for gen in range(num_generations):
        start = time.time()
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # LHS crossover
            if np.random.rand() < SPC1:
                toolbox.spc1(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
            if np.random.rand() < SPC2:
                toolbox.spc2(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
            # RHS crossover
            if np.random.rand() < PMX:
                toolbox.PMX(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
            if np.random.rand() < OX:
                toolbox.OX(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
            if np.random.rand() < JL:
                toolbox.JL(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # LHS mutation
            if np.random.rand() < SSTM:
                toolbox.sstm(mutant)
                del mutant.fitness.values
            # RHS mutation
            if np.random.rand() < MSI:
                toolbox.MSI(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness (new individuals)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the population by the offspring
        population[:] = offspring

        end = time.time()

        if shifts:
            print(
                "Generation: ",
                gen,
                "Best fitness (makespan + penalty): ",
                min([ind.fitness.values[0] for ind in population]),
                "Time elapsed: ",
                end - start,
            )
        else:
            print(
                "Generation: ",
                gen,
                "Best fitness (makespan): ",
                min([ind.fitness.values[0] for ind in population]),
                "Time elapsed: ",
                end - start,
            )

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        current_best_fitness = min(fits)

        # Check for improvement
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            no_improvement_generations = 0
            SSTM, MSI = 0.2, 0.2
        else:
            no_improvement_generations += 1
            if SSTM < 0.8:
                SSTM = SSTM * 1.05
                MSI = MSI * 1.05

        # Diversification step
        if no_improvement_generations >= diversification_threshold:
            diversification_threshold += 5
            print(
                "\n+++++++++++++++\nDiversification triggered at generation",
                gen,
                "new diversification threshold:",
                diversification_threshold,
                "\n+++++++++++++++\n",
            )
            # Delete half of the population
            half_population_size = len(population) // 2
            population = sorted(population, key=lambda ind: ind.fitness.values[0])
            population = population[:half_population_size]
            # Introduce new random solutions
            new_individuals = toolbox.population(
                n=population_size - half_population_size
            )
            population.extend(new_individuals)
            no_improvement_generations = 0

        end = time.time()
        print(f"Generation {gen} completed in {end - start} seconds")

    # Step 6: Get the best individual
    best_individual = tools.selBest(population, 1)[0]

    if plotting:
        if shifts:
            print("Best fitness (makespan + shift constraint penalty): ", best_fitness)
        else:
            print("Best fitness (makespan): ", best_fitness)

        df_results = decoder.get_dataframe_results(
            best_individual, params, shifts, seq_dep_setup
        )
        plot.gantt(df_results, params, shifts=shifts, seq_dep_setup=seq_dep_setup)

    return best_fitness, best_individual


def main():
    # --------- PARAMETERS AND CONSTRAINTS (CHANGE FOR DIFFERENT SCENARIOS) ---------
    # contraints
    shifts_constraint = True
    sequence_dependent = True

    # Parameters
    n_machines = 3  # number of machines
    n_jobs = 3  # number of jobs
    n_lots = 3  # number of lots
    seed = 5  # seed for random number generator
    demand = {i: 50 for i in range(0, n_jobs + 1)}  # demand of each job

    # Create parameters object
    my_params = params.JobShopRandomParams(
        config_path="config/jobshop/js_params_01.yaml"
    )
    my_params.demand = demand  # demand of each job
    my_params.print_jobshop_params(save_to_excel=False)

    # --------- GENETIC ALGORITHM (CHANGE FOR DIFFERENT GA CONFIGURATIONS) ---------
    # Genetic algorithm
    start = time.time()
    current_fitness, best_individual = run(
        my_params,
        population_size=100,
        num_generations=100,
        plotting=True,
    )
    end = time.time()
    print("Time elapsed: ", end - start, "seconds")
    print("Current fitness: ", current_fitness)
    if my_params.shift_constraints:
        js_decoder = decoder.JobShopDecoder(my_params)
        makespan, penalty, y, c, chromosome_mod = js_decoder.decode(best_individual)
        print("Makespan: ", makespan)
        print("Penalty: ", penalty)
        print(best_individual)
        print(chromosome_mod)


if __name__ == "__main__":
    main()
