"""
Created on Aug 08 2024
Refactored on Aug 2025

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: genetic_algorithm.py - Genetic Algorithm
"""

# --------- LIBRARIES ---------
from deap import base, creator, tools
import time
import numpy as np
from typing import Tuple, Callable

# --------- src/ MODULES ---------
from .chromosome import ChromosomeGenerator
from scheduling import Scheduler
from jobshop import JobShopRandomParams
from .operators import LotStreamingOperators
from shared.types import Chromosome
from shared.utils import load_config, timed

# --------- GENETIC ALGORITHM ---------


class GeneticAlgorithm:
    def __init__(self, problem_params: JobShopRandomParams, logger, config_path: str):
        self.problem_params = problem_params
        self.logger = logger
        self.config = load_config(config_path)
        self.population_size = self.config["population_size"]
        self.num_generations = self.config["num_generations"]
        self.cx_probs: dict[str, float] = self.config["crossover"]
        self.mut_probs: dict[str, float] = self.config["mutation"]
        self.tournament_size = self.config["selection"]["tournament_size"]
        self.diversification = self.config["diversification"]["enabled"]
        self.diversification_threshold = self.config["diversification"]["threshold"]
        # Genetic operators setup
        operators = LotStreamingOperators()
        operators.build_master_ops_dict(problem_params)
        operators.build_inverted_master_ops_dict()
        self._cx_funcs = {
            "spc1": operators.spc1_lhs,
            "spc2": operators.spc2_lhs,
            "pmx": operators.pmx_rhs,
            "ox": operators.ox_rhs,
            "job_level": operators.cx_job_level_rhs,
        }
        self._mut_funcs = {
            "sstm": operators.sstm_lhs,
            "msi": operators.shuffle_rhs,
        }

    @timed
    def run(self) -> Tuple[float, list, Chromosome]:
        """
        Run the genetic algorithm for the Lot Streaming Job Shop Scheduling Problem.
        Returns:
            best_fitness: Best makespan value found.
            fitness_history: List of best fitness values per generation.
            best_individual: Chromosome of the best solution found.
        """
        # Initialize DEAP toolbox
        toolbox = self._setup_deap()
        # Create the initial population
        population = toolbox.population(n=self.population_size)
        # Initialize best fitness tracking for plotting
        best_fitness = float("inf")
        best_fitness_history = []
        self.no_improve_gen = 0  # Counter for generations without improvement

        for gen in range(self.num_generations):
            start = time.time()

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            self._apply_crossover(offspring)
            self._apply_mutation(offspring)

            # Evaluate the individuals that have evolved
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            # Replace the population by the offspring
            population[:] = offspring

            # Track the best fitness of the current population
            best_population_fitness = min(ind.fitness.values[0] for ind in population)

            # Append the best fitness of this generation to the history
            best_fitness_history.append(best_population_fitness)

            # Increase mutation rates if no improvement
            self._update_mutation_rates(best_fitness, best_population_fitness)

            # Update the best fitness and individual if found a better one
            if best_population_fitness < best_fitness:
                best_fitness = best_population_fitness
                best_individual = tools.selBest(population, 1)[0]

            # Diversify population if needed
            if (
                self.diversification
                and self.no_improve_gen >= self.diversification_threshold
            ):
                self.logger.info(
                    f"Diversification triggered at gen {gen}, new threshold {self.diversification_threshold}"
                )
                population = self._diversify_population(population, toolbox)

            end = time.time()
            self.logger.info(f"Generation {gen} completed in {end - start} seconds")
            self.logger.info(
                f"Best fitness of this population: {best_population_fitness}"
            )

        return best_fitness, best_fitness_history, best_individual

    def _eval_fitness(self, individual: Chromosome) -> Tuple[float]:
        """
        Evaluate the fitness of an individual.
        Returns:
            A tuple containing the fitness value (makespan).
        """
        self.scheduler = Scheduler(self.problem_params)
        return (self.scheduler.get_fitness(encoded_solution=individual),)

    def _create_individual_factory(self) -> Callable:
        """
        Factory function to create an individual.
        Returns:
            A function that generates a new individual.
        """
        chromosome_generator = ChromosomeGenerator(self.problem_params)
        return chromosome_generator.generate

    def _setup_deap(self) -> base.Toolbox:
        """
        Set up DEAP for the genetic algorithm.
        Returns:
            toolbox: The DEAP toolbox configured with the problem parameters.
        """
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self._create_individual_factory(),
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        toolbox.register("evaluate", self._eval_fitness)
        return toolbox

    def _apply_crossover(self, offspring: list[Chromosome]):
        """
        Apply crossover to the offspring.
        Args:
            offspring: List of offspring individuals.
        """
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            for name, func in self._cx_funcs.items():
                p = self.cx_probs.get(name) or 0.0
                if p > 0 and np.random.rand() < p:
                    child1, child2 = func(ind1, ind2)
                    ind1[:], ind2[:] = child1, child2
                    # invalidate fitness so it’ll be re-evaluated
                    del ind1.fitness.values
                    del ind2.fitness.values

    def _apply_mutation(self, offspring: list[Chromosome]):
        """
        Apply mutation to the offspring.
        Args:
            offspring: List of offspring individuals.
        """
        for ind in offspring:
            for name, func in self._mut_funcs.items():
                p = self.mut_probs.get(name, 0.0)
                if p > 0 and np.random.rand() < p:
                    mutant = func(ind)
                    ind[:] = mutant
                    del ind.fitness.values

    def _diversify_population(
        self, population: list[Chromosome], toolbox: base.Toolbox
    ) -> list[Chromosome]:
        """
        Kills half of the population and introduces new random individuals
        Args:
            population: Current population of individuals.
            no_improvement_generations: Number of generations without improvement.
        """
        self.diversification_threshold *= 2  # Increase threshold for next time
        # Delete half of the population
        half = self.population_size // 2
        population.sort(key=lambda ind: ind.fitness.values[0])
        new_pop = population[:half]
        # Fill the remainder with new individuals
        n_new = self.population_size - half
        new_pop.extend(toolbox.population(n=n_new))
        # Reset no_improve_gen counter
        self.no_improve_gen = 0
        return new_pop

    def _update_mutation_rates(
        self, best_fitness: float, best_population_fitness: float
    ):
        """
        Reset or grow the mutation probabilities based on improvement.
        """
        init_sstm = self.config["mutation"]["sstm"]
        cap = self.config.get("mutation_cap", 0.8)
        factor = self.config.get("mutation_growth", 1.05)

        if best_population_fitness < best_fitness and self.no_improve_gen > 0:
            # reset everything
            best_fitness = best_population_fitness
            self.no_improve_gen = 0
            new_rate = init_sstm
        else:
            # no improvement
            self.no_improve_gen += 1
            # grow up to cap, otherwise reset
            grown = self.mut_probs["sstm"] * factor
            new_rate = grown if grown < cap else init_sstm

        self.mut_probs["sstm"] = new_rate
        self.mut_probs["msi"] = new_rate
