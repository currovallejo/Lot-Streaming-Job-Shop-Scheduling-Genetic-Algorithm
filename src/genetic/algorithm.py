"""
Genetic Algorithm implementation for Lot Streaming Job Shop Scheduling Problem.

This module implements a genetic algorithm using the DEAP framework to solve the
Lot Streaming Job Shop Scheduling Problem. The algorithm optimizes job scheduling
to minimize makespan through evolutionary operators including crossover, mutation,
selection, and population diversification.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog
"""

from deap import base, creator, tools
import time
import numpy as np
from typing import Tuple, Callable, Optional
from logging import Logger

from .chromosome import ChromosomeGenerator
from scheduling import Scheduler
from jobshop import JobShopRandomParams
from .operators import LotStreamingOperators
from shared.types import Chromosome
from shared.utils import load_config, timed
from .metrics import MetricsSink
from .results import GAResult

# --------- GENETIC ALGORITHM ---------


class GeneticAlgorithm:
    def __init__(
        self,
        problem_params: JobShopRandomParams,
        logger: Logger,
        config_path: str,
        metrics_sink: MetricsSink,
    ):
        """
        Initialize the genetic algorithm with problem parameters, logger, and
        configuration path.
        Args:
            problem_params (JobShopRandomParams): Parameters for the job shop problem.
            logger (Logger): Logger instance for logging information.
            config_path (str): Path to the configuration file.
        """

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
            "job_level": operators.cx_job_level_rhs,
        }
        self._mut_funcs = {
            "sstm": operators.sstm_lhs,
            "msi": operators.shuffle_rhs,
        }
        self.metrics_sink = metrics_sink

    @timed
    def run(self) -> GAResult:
        """Run the genetic algorithm
        Returns:
            GAResult: The result of the genetic algorithm run, containing metrics and best solution.
        """
        # Initialize DEAP toolbox
        toolbox = self._setup_deap()
        # Create the initial population
        population = toolbox.population(n=self.population_size)
        # Initialize best fitness tracking for plotting
        best_fitness = float("inf")
        best_individual = None
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

            # Increase mutation rates if no improvement
            self._update_mutation_rates(best_fitness, best_population_fitness)

            # Update best solution when found
            if best_population_fitness < best_fitness:
                best_fitness = best_population_fitness
                best_individual = tools.selBest(population, 1)[0]
                self.metrics_sink.set_best_solution(best_individual)

            # Report metrics at the end of each generation
            self.metrics_sink.on_generation_end(best_population_fitness, best_fitness)

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

        # Compute final makespan and penalty for the best solution
        makespan, penalty = self._compute_final_metrics(best_individual)
        self.metrics_sink.set_final_metrics(makespan, penalty)

        return self.metrics_sink.finalize()

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

    def _compute_final_metrics(self, best_individual) -> Tuple[float, Optional[float]]:
        """
        Compute final makespan and penalty for the best individual.

        Args:
            best_individual: The best individual found during the GA run

        Returns:
            Tuple of (makespan, penalty) where penalty is None if shift_constraints is False
        """
        if best_individual is None:
            return float("inf"), None

        # Use existing scheduler instance to decode the solution
        decoded_result = self.scheduler.decode(best_individual)
        makespan = decoded_result[0]  # First element is makespan

        # Only compute penalty if shift constraints are enabled
        penalty = decoded_result[1] if self.problem_params.shift_constraints else None

        return makespan, penalty
