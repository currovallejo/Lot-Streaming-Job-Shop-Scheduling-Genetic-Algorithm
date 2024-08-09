"""
Created on Aug 08 2024

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: GA.py - Genetic Algorithm
"""
#--------- LIBRARIES ---------
from deap import base, creator, tools, algorithms

#--------- OTHER PYTHON FILES USED ---------
import chromosome_generator
import params
import decoder
from genetic_operators import *



#--------- GENETIC ALGORITHM ---------

def create_individual_factory(params, demand):
    def create_individual():
        return chromosome_generator.generate_chromosome(params, demand)
    return create_individual

def lsjsp_ga(params, demand):
    """
    Genetic algorithm for LSJSP problem developed with DEAP package

    Args:

    Returns:
        best_makespan: best fitness value found
        df_results: parameters of best solution found
    """
    #--------- DEAP SETUP ---------
    # Step 1: Define the fitness function
    def evalFitness(individual, params):
        decoded_chromosome = decoder.decode_chromosome(individual, params)
        makespan = decoded_chromosome[0]
        return makespan
    
    # Step 2: Create the individual and population
    # Define the fitness class for minimization
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) 
    
    # Define the individual class
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Create the toolbox
    toolbox = base.Toolbox()

    # Factory function for individual creation
    create_individual = create_individual_factory(params, demand)

    # Register the individual creation function
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Step 3: Register genetic operators





def main():
    my_params = params.JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=4)
    demand = {i:50 for i in range(0,11)}
    ind_1 = chromosome_generator.generate_chromosome(my_params, demand)
    ind_2 = chromosome_generator.generate_chromosome(my_params, demand)
    ind_12 = spc1_crossover(ind_1, ind_2)
    print('ind1: \n',ind_1[0])
    print('ind2: \n',ind_2[0])
    print('ind12: \n',ind_12[0][0],'\n', ind_12[1][0])

if __name__ == "__main__":
    main()