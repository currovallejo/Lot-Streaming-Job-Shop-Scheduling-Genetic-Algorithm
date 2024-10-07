"""
Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro

Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH GA
Script: chromosome.py - migration of chromosome_generator.py and decoder.py to a class
"""

# --------- LIBRARIES ---------
import random
import numpy as np

# --------- OTHER PYTHON FILES USED ---------
import params


class Chromosome:
    """
    Represents a chromosome in the job-shop lot streaming scheduling problem.
    Handles the creation, decoding, and evaluation of chromosomes based on the problem's parameters.
    """

    def __init__(self, parameters):
        """
        Initializes the Chromosome object with the parameters required to generate a chromosome.

        Args:
            params: An instance of JobShopRandomParams, containing jobs, lots, and sequences.
        """
        self.jobs = parameters.jobs
        self.lots = parameters.lots
        self.seq = parameters.seq
        self.n_jobs = len(self.jobs)
        self.n_lots = len(self.lots)
        self.n_machines = len(parameters.machines)
        self.chromosome_lhs = None
        self.chromosome_rhs = None

    def generate(self):
        """
        Generates a random chromosome.

        Returns:
            A tuple of numpy array (left-hand side) and a shuffled list (right-hand side) representing the chromosome.
        """
        # Generate chromosome left-hand side (numeric values)
        self.chromosome_lhs = np.array(
            [random.random() for job in self.jobs for lot in self.lots]
        )

        # Generate chromosome right-hand side (sublots)
        self.chromosome_rhs = [
            (job, lot)
            for job in self.jobs
            for lot in self.lots
            for machine in self.seq[job]
        ]

        # Shuffle right-hand side to randomize sublots
        random.shuffle(self.chromosome_rhs)

        return self.chromosome_lhs, self.chromosome_rhs
    
    def distribute_demand(self):
        """
        Distributes the total demand into parts based on the generated chromosome fractions
        such that the sum of the parts equals the total demand for each job.

        Returns:
            numpy array with the demand distributed into lots (LHS of the chromosome modified).
        """
        if self.chromosome_lhs is None:
            raise ValueError("Chromosome LHS not generated yet. Call the generate method first.")

        chromosome_lhs_m = np.copy(self.chromosome_lhs)

        # Distribute demand for each job
        for job in self.jobs:
            total = np.sum(chromosome_lhs_m[self.n_lots*job:self.n_lots*(job+1)])

            if total != 0:  # Avoid division by zero
                chromosome_lhs_m[self.n_lots*job:self.n_lots*(job+1)] /= total

                for lot in self.lots:
                    chromosome_lhs_m[self.n_lots*job+lot] = int(chromosome_lhs_m[self.n_lots*job+lot] * self.demand[job])
                
                total_preliminary = sum(chromosome_lhs_m[self.n_lots*job:self.n_lots*(job+1)])
                residual = self.demand[job] - total_preliminary

                # Distribute the residual if there is any
                if residual > 0:
                    for lot in self.lots:
                        chromosome_lhs_m[self.n_lots*job+lot] += 1
                        residual -= 1
                        if residual == 0:
                            break
            else:
                # Handle case where total is 0
                chromosome_lhs_m[self.n_lots*job:self.n_lots*(job+1)] = int(self.demand[job]) / self.n_lots
                chromosome_lhs_m[self.n_lots*job+self.n_lots-1] = self.demand[job] - sum(chromosome_lhs_m[self.n_lots*job:self.n_lots*(job+1)-1])

        return chromosome_lhs_m

    def decode(self):
        """
        Decodes the chromosome into a more interpretable schedule (or any other meaningful structure).

        Returns:
            Decoded representation of the chromosome (for now, simply returns the right-hand side).
        """
        # Placeholder for decoding logic - this can be expanded based on your decoding strategy
        return self.chromosome_rhs

    def calculate_fitness(self):
        """
        Calculates the fitness of the chromosome based on some fitness function.

        Returns:
            A fitness value for the chromosome.
        """
        # Placeholder for the fitness calculation
        fitness_value = (
            random.random()
        )  # Replace with your actual fitness function logic
        return fitness_value
