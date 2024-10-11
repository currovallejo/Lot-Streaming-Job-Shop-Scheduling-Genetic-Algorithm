"""
Created on Aug 08 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: genetic_algorithm_POO.py - migration of the genetic algorithm to a class-based structure

*FAILED* - GA runs slower
"""

# --------- LIBRARIES ---------
from deap import base, creator, tools
import time

# --------- OTHER PYTHON FILES USED ---------
import decoder
import chromosome_generator
import params as parameters
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


class GA:
    """
    Genetic Algorithm class for the Lot Streaming Job Shop Scheduling Problem (LSJSP).

    Attributes:
        params (dict): Parameters for the genetic algorithm.
        population_size (int): Size of the population.
        num_generations (int): Number of generations.
        shifts (bool): Whether to consider shifts in the scheduling.
        seq_dep_setup (bool): Whether to consider sequence-dependent setup times.
        toolbox (deap.base.Toolbox): DEAP toolbox for genetic algorithm operations.
    """

    def __init__(
        self,
        params,
        population_size=100,
        num_generations=50,
        diversification_step=5,
        shifts=False,
        seq_dep_setup=False,
    ):
        self.params = params
        self.population_size = population_size
        self.num_generations = num_generations
        self.diversification_step = diversification_step
        self.shifts = shifts
        self.seq_dep_setup = seq_dep_setup
        self.toolbox = base.Toolbox()

    def initialize_DEAP(self):
        """
        This method sets up the DEAP (Distributed Evolutionary Algorithms in Python) toolbox
        for use in the genetic algorithm. It defines the fitness and individual classes,
        registers various genetic operators (crossover, mutation, selection), and creates
        the initial population of individuals.
        The following components are registered in the DEAP toolbox:
        - Fitness class for minimization.
        - Individual class.
        - Individual creation function.
        - Population creation function.
        - Evaluation function.
        - Crossover genetic operators: SPC1, SPC2, PMX, OX, JL.
        - Mutation genetic operators: SSTM, MSI.
        - Selection function: Tournament selection with a tournament size of 3.
        Additionally, this method creates the initial population.

        Functions defined:
            create_individual(self): Creates an individual for the genetic algorithm.

        Attributes initialized:
            self.population (list): The initial population of individuals.
            self.gen_opt_probabilities (dict): Probabilities for genetic operators.
        """

        def create_individual_factory():
            """
            Creates a factory function for generating individuals.

            Returns:
                function: A function that generates a chromosome using the parameters.
            """

            def create_individual():
                return chromosome_generator.generate_chromosome(self.params)

            return create_individual

        def evaluate_individual(individual, params, shifts, seq_dep_setup):
            """
            Default evaluation function to evaluate the fitness of an individual.

            Args:
                individual: The individual to evaluate.

            Returns:
                Fitness: The fitness value of the individual.
            """
            decoded_chromosome = decoder.decode_chromosome(
                individual, params, shifts, seq_dep_setup
            )
            fitness = decoded_chromosome[0]  # makespan

            if shifts:
                shift_penalty = decoded_chromosome[1]
                fitness += shift_penalty  # makespan + shift_penalty

            return (fitness,)

        # Define the fitness class for minimization
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Define the individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Register the individual creation function
        create_individual = create_individual_factory()
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, create_individual
        )

        # Register the population creation function
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register the evaluation function
        self.toolbox.register(
            "evaluate",
            evaluate_individual,
            params=self.params,
            shifts=self.shifts,
            seq_dep_setup=self.seq_dep_setup,
        )

        # Register the crossover genetic operators
        self.toolbox.register("spc1", spc1_crossover_lhs)
        self.toolbox.register("spc2", spc2_crossover_lhs)
        self.toolbox.register("PMX", cxPartiallyMatched_rhs)
        self.toolbox.register("OX", cxOrdered_rhs)
        self.toolbox.register("JL", cxJobLevel_rhs)

        # Register the mutation function
        self.toolbox.register("sstm", sstm_mutation_lhs)
        self.toolbox.register("MSI", mutShuffleIndexes_rhs)

        # Probabilities for genetic operators
        self.gen_opt_probabilities = {
            "SPC1": 0.8,
            "SPC2": 0.8,
            "PMX": 0.8,
            "OX": 0.8,
            "JL": 0.8,
            "SSTM": 0.2,
            "MSI": 0.2,
        }

        # Register the selection function
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Create the population
        start_population = time.time()
        self.population = self.toolbox.population(n=self.population_size)
        end_population = time.time()
        print(
            "Population of ",
            self.population_size,
            " individuals created in ",
            end_population - start_population,
            " seconds",
        )

    def run(self, show_evolution=True):
        """
        Runs the genetic algorithm for the LSJSP problem.

        Returns:
            best_makespan: The best fitness value found.
            df_results: The parameters of the best solution found.

        Attributes initialized:
            self.fitness_scores (list): The best fitness score of the population in each generation.
        """
        # Diversification threshold for population diversity
        no_improvement_generations = 0
        diversification_threshold = 10
        best_fitness = float("inf")
        self.fitness_scores = []

        for gen in range(self.num_generations):
            start = time.time()
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # LHS crossover
                if np.random.rand() < self.gen_opt_probabilities["SPC1"]:
                    self.toolbox.spc1(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                if np.random.rand() < self.gen_opt_probabilities["SPC2"]:
                    self.toolbox.spc2(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                # RHS crossover
                if np.random.rand() < self.gen_opt_probabilities["PMX"]:
                    self.toolbox.PMX(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                if np.random.rand() < self.gen_opt_probabilities["OX"]:
                    self.toolbox.OX(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                if np.random.rand() < self.gen_opt_probabilities["JL"]:
                    self.toolbox.JL(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # LHS mutation
                if np.random.rand() < self.gen_opt_probabilities["SSTM"]:
                    self.toolbox.sstm(mutant)
                    del mutant.fitness.values
                # RHS mutation
                if np.random.rand() < self.gen_opt_probabilities["MSI"]:
                    self.toolbox.MSI(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness (new individuals)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace the population by the offspring
            self.population[:] = offspring
            end = time.time()
            if self.shifts:
                print(
                    "Generation: ",
                    gen,
                    "Best fitness (makespan + penalty): ",
                    min([ind.fitness.values[0] for ind in self.population]),
                    "Time elapsed: ",
                    end - start,
                )
            else:
                print(
                    "Generation: ",
                    gen,
                    "Best fitness (makespan): ",
                    min([ind.fitness.values[0] for ind in self.population]),
                    "Time elapsed: ",
                    end - start,
                )

            # Gather all the fitnesses in one list and print the stats
            current_fitness_scores = [ind.fitness.values[0] for ind in self.population]
            current_best_fitness = min(current_fitness_scores)
            self.fitness_scores.append(current_fitness_scores)

            # Check for improvement
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1

            # Diversification step
            if no_improvement_generations >= diversification_threshold:
                diversification_threshold += self.diversification_step
                print(
                    "\n+++++++++++++++\nDiversification triggered at generation",
                    gen,
                    "new diversification threshold:",
                    diversification_threshold,
                    "\n+++++++++++++++\n",
                )
                # Delete half of the population
                half_population_size = len(self.population) // 2
                self.population = sorted(
                    self.population, key=lambda ind: ind.fitness.values[0]
                )
                self.population = self.population[:half_population_size]
                # Introduce new random solutions
                new_individuals = self.toolbox.population(
                    n=self.population_size - half_population_size
                )
                self.population.extend(new_individuals)
                no_improvement_generations = 0

            # Get the best individual
            best_individual = tools.selBest(self.population, 1)[0]

        if self.shifts:
            print("Best fitness (makespan + shift constraint penalty): ", best_fitness)
        else:
            print("Best fitness (makespan): ", best_fitness)

        df_results = decoder.get_dataframe_results(
            best_individual, self.params, self.shifts, self.seq_dep_setup
        )

        return best_fitness, best_individual, df_results


def main():
    # --------- PARAMETERS AND CONSTRAINTS (CHANGE FOR DIFFERENT SCENARIOS) ---------
    # contraints
    shifts_constraint = False
    sequence_dependent = False

    # Parameters
    n_machines = 3  # number of machines
    n_jobs = 3  # number of jobs
    n_lots = 4  # number of lots
    seed = 4  # seed for random number generator
    demand = {i: 100 for i in range(0, n_jobs + 1)}  # demand of each job

    # Create parameters object
    params = parameters.JobShopRandomParams(
        n_machines=n_machines, n_jobs=n_jobs, n_lots=n_lots, seed=seed
    )
    params.demand = demand  # demand of each job
    params.printParams(sequence_dependent=sequence_dependent, save_to_excel=False)

    # --------- GENETIC ALGORITHM (CHANGE FOR DIFFERENT GA CONFIGURATIONS) ---------
    # Genetic algorithm
    start = time.time()
    # Create genetic algorithm object
    genetic_algorithm = GA(
        params,
        population_size=100,
        num_generations=50,
        shifts=shifts_constraint,
        seq_dep_setup=sequence_dependent,
    )
    # Initialize DEAP toolbox and create initial population
    genetic_algorithm.initialize_DEAP()
    # Run genetic algorithm
    best_fitness, best_individual, df_results = genetic_algorithm.run(
        show_evolution=True
    )
    end = time.time()
    # Print results
    print("Time elapsed: ", end - start, "seconds")
    if shifts_constraint:
        makespan, penalty, y, c, chromosome_mod = decoder.decode_chromosome(
            best_individual,
            params,
            shifts=shifts_constraint,
            seq_dep_setup=sequence_dependent,
        )
        print("Makespan: ", makespan)
        print("Penalty: ", penalty)
    else:
        print("Makespan: ", best_fitness)

    # Plotting fitness evolution
    plot.solution_evolution(genetic_algorithm.fitness_scores)
    # Plotting Gantt chart
    plot.gantt(
        df_results,
        params,
        show=True,
        version=0,
        shifts=shifts_constraint,
        seq_dep_setup=sequence_dependent,
    )


if __name__ == "__main__":
    main()
