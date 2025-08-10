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
import copy
from deap import tools
from typing import Tuple, List

# --------- src/ MODULES ---------
from .. import jobshop
from ..shared.types import Chromosome, RHS

# --------- GENETIC OPERATORS ---------


class SimpleCrossoverOperators:
    """Crossover custom operators for 1D arrays"""

    def spc1(self, ind1: list, ind2: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply SPC-1 crossover operator

        SPC-1 (Single Point Crossover 1) selects a single cut point in the sequence
        of genes, then exchanges the tail segments of the two parents to create
        offspring. This preserves the relative order of genes before and after
        the crossover point.
        """
        cp = random.randint(1, len(ind1) - 1)
        new_ind1, new_ind2 = np.copy(ind1), np.copy(ind2)
        new_ind1[:cp], new_ind2[:cp] = ind2[:cp], ind1[:cp]
        return new_ind1, new_ind2

    def spc2(self, ind1: list, ind2: list) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply SPC-2 crossover operator

        SPC-2 (Two-Point Crossover) selects two cut points in the gene sequence,
        exchanges the segment between them between parents, and leaves the
        outer segments intact. This introduces more mixing while preserving
        gene order outside the crossover region.
        """
        cp = random.randint(1, len(ind1) - 1)
        new_ind1, new_ind2 = np.copy(ind1), np.copy(ind2)
        new_ind1[cp:], new_ind2[cp:] = ind2[cp:], ind1[cp:]
        return new_ind1, new_ind2


class SimpleMutationOperators:
    """Mutation custom operators for 1D arrays"""

    def sstm(self, ind: list, gen_mutation_prob=0.2, delta_max=0.5) -> np.ndarray:
        """
        Apply Sublot Step Mutation (SStM) to a real-valued chromosome.

        Each gene in the input array `ind` is considered for mutation with
        independent probability `gen_mutation_prob`. When a gene is selected,
        its value is perturbed by a random step Δ ∈ [0, delta_max], chosen uniformly.
        The direction of the step (positive or negative) is equally likely.
        After mutation, each gene is clipped to the [0, 1] interval.

        Args:
            ind: The LHS chromosome to mutate.
            mutation_prob (float): Probability of mutation for each gene.
            delta_max (float): Maximum step size for mutation.
        Returns:
            mutant: The mutated LHS chromosome.
        """
        mutant = np.copy(ind)
        for i in range(len(mutant)):
            if np.random.rand() < gen_mutation_prob:
                delta = delta_max * np.random.rand()
                if np.random.rand() < 0.5:
                    mutant[i] = min(1, mutant[i] + delta)
                else:
                    mutant[i] = max(0, mutant[i] - delta)
        return mutant


class LotStreamingOperators:
    """Genetic operators for Lot Streaming Job Shop chromosomes.

    A chromosome consists of:
    - Left-hand side (LHS): a NumPy array representing lot sizes.
    - Right-hand side (RHS): a list of (job, lot) tuples representing operation
    sequences.
    """

    def __init__(self):
        """ "Initialize custom crossover and mutation operators for LHS (1D array)."""
        self.flat_crossover_operators = SimpleCrossoverOperators()
        self.flat_mutation_operators = SimpleMutationOperators()

    # ---- CONVERSION METHODS FOR RHS (Job Sequences) ---------

    def build_master_ops_dict(
        self, problem_params: jobshop.JobShopRandomParams
    ) -> None:
        """
        Populate the master_ops attribute  with a mapping from
        operation-index → (job, lot), based on problem parameters.

        This mapping allows decoding flat operations 1D lists back into the RHS
        chromosome format.

        Args:
            problem_params: Contains
            - jobs: Iterable of job IDs
            - lots: Iterable of lot IDs
            - seq:  Dictionary mapping job IDs to processing routes (list of machines).

        Side Effects:
            Sets `self.master_ops: Dict[int, Tuple[int, int]]`.
        """
        jobs = problem_params.jobs
        lots = problem_params.lots
        seq = problem_params.seq

        ops_iter = (
            (job, lot) for job in jobs for lot in lots for _ in range(len(seq[job]))
        )

        self.master_ops = dict(enumerate(ops_iter))

    def build_inverted_master_ops_dict(self) -> None:
        """
        Populate the inverted_master_ops attribute with a mapping from
        (job, lot) → list of operation indices.

        This inverted index allows decoding flat operations 1D lists back into the
        original list of (job, lot) tuples, supporting crossover and mutation operators
        that require flat representations without duplicated values.

        Side Effects:
            Sets `self.inverted_master_ops: Dict[Tuple[int, int], List[int]]`.
        """
        value_to_keys: dict[Tuple[int, int], list[int]] = {}
        for op_idx, job_lot in self.master_ops.items():
            value_to_keys.setdefault(job_lot, []).append(op_idx)
        self.inverted_master_ops = value_to_keys

    def _rhs_to_ops_list(self, rhs: RHS) -> list[int]:
        """Given a sequence of (job, lot) tuples, consume one operation-ID
        from inverted_master_ops per occurrence, in order.
        Args:
            rhs (list[tuple[int, int]]): Right-hand side of the chromosome.
        Returns:
            list[int]: List of operation indices corresponding to the RHS.
                i.e. RHS converted to a flat list of unique operation indices.
        """
        inv = self.inverted_master_ops
        # Build one iterator per job_lot
        ops_iters = {jl: iter(indices) for jl, indices in inv.items()}
        # Consume one operation per job_lot in the RHS
        return [next(ops_iters[jl]) for jl in rhs]

    def _ops_list_to_rhs(self, ops: list[int]) -> RHS:
        """
        Decode a flat list of operation IDs into the RHS sequence of (job, lot) tuples.

        Args:
            ops: Sequence of integer operation IDs.

        Returns:
            A list of (job, lot) tuples corresponding to each op-ID in order.
        """
        master_ops = self.master_ops
        return [master_ops[op] for op in ops]

    def _rhs_crossover_func(
        self, rhs1: RHS, rhs2: RHS, deap_crossover_func
    ) -> tuple[RHS, RHS]:
        """
        Applies a DEAP-style crossover function to RHS chromosomes.

        This method converts structured RHS (list of tuples) to a flat representation,
        applies the crossover, and converts the result back.

        Args:
            rhs1: RHS of the first chromosome (list of (job, lot) tuples).
            rhs2: RHS of the second chromosome (list of (job, lot) tuples).
            deap_crossover_func: DEAP-style crossover function accepting flat lists.

        Returns:
            A tuple containing two new RHS lists after crossover.
        """
        rhs1_ops = self._rhs_to_ops_list(rhs1)
        rhs2_ops = self._rhs_to_ops_list(rhs2)
        child1, child2 = deap_crossover_func(rhs1_ops, rhs2_ops)
        new_rhs1 = self._ops_list_to_rhs(child1)
        new_rhs2 = self._ops_list_to_rhs(child2)
        return new_rhs1, new_rhs2

    def _rhs_mutation_func(self, rhs: list, deap_mutation_func) -> list:
        """Generic mutation function for right-hand side
            Made to handle deap mutation functions that expect flat lists
        Args:
            rhs (list): Right-hand side of the chromosome.
            mutation_func: Mutation function to apply.
        Returns:
            list: New right-hand side after mutation.
        """
        rhs_ops = self._rhs_to_ops_list(rhs)
        child, *_ = deap_mutation_func(rhs_ops, indpb=0.2)
        new_rhs = self._ops_list_to_rhs(child)
        return new_rhs

    # --------- GENERIC METHODS ---------

    def _crossover_template(
        self,
        ind1: Chromosome,
        ind2: Chromosome,
        crossover_func,
        target_idx: int,
    ) -> Tuple[Chromosome, Chromosome]:
        """Generic crossover method for chromosomes
        Args:
            ind1 (tuple): First individual (chromosome).
            ind2 (tuple): Second individual (chromosome).
            crossover_func: Crossover function to apply.
            target_idx (int): Idx of the chromosome part (lhs / rhs) to apply crossover.
        Returns:
            tuple: New individuals after crossover.
        """
        new_ind1, new_ind2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
        new_ind1[target_idx], new_ind2[target_idx] = crossover_func(
            ind1[target_idx], ind2[target_idx]
        )
        return new_ind1, new_ind2

    def _mutation_template(
        self,
        ind: Chromosome,
        mutation_func,
        target_idx: int,
    ) -> list:
        """Generic mutation method for chromosomes
        Args:
            ind (tuple): Individual (chromosome).
            mutation_func: Mutation function to apply.
            target_index (int): Index of the chromosome part to apply mutation.
        Returns:
            tuple: New individual after mutation.
        """
        new_ind = copy.deepcopy(ind)
        new_ind[target_idx] = mutation_func(ind[target_idx])
        if type(new_ind[target_idx]) is tuple:
            new_ind[target_idx] = new_ind[target_idx][0]
        return new_ind

    # --------- LHS CROSSOVER METHODS (Lot Sizes) ---------
    def spc1_lhs(
        self, ind1: Chromosome, ind2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """Perform SPC-1 crossover on the left-hand side of two chromosomes."""
        return self._crossover_template(
            ind1, ind2, self.flat_crossover_operators.spc1, 0
        )

    def spc2_lhs(
        self, ind1: Chromosome, ind2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """Perform SPC-2 crossover on the left-hand side of two chromosomes."""
        return self._crossover_template(
            ind1, ind2, self.flat_crossover_operators.spc2, 0
        )

    # --------- RHS CROSSOVER METHODS (Job Sequences) ---------

    def pmx_rhs(
        self, ind1: Chromosome, ind2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        PMX (Partially Matched Crossover) crossover for right-hand side of chromosomes.
            For more information, see DEAP package documentation
        """
        return self._crossover_template(
            ind1,
            ind2,
            lambda rhs1, rhs2: self._rhs_crossover_func(
                rhs1, rhs2, tools.cxPartialyMatched
            ),
            target_idx=1,
        )

    def ox_rhs(
        self, ind1: Chromosome, ind2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        OX crossover for right-hand side of chromosomes.
            For more information, see DEAP package documentation
        """
        return self._crossover_template(
            ind1,
            ind2,
            lambda rhs1, rhs2: self._rhs_crossover_func(rhs1, rhs2, tools.cxOrdered),
            target_idx=1,
        )

    def cx_job_level_rhs(
        self, ind1: Chromosome, ind2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Perform a job-level crossover for right-hand side of chromosomes.

        This operator exchanges all operations of a single randomly chosen job
        between parents, preserving their original positions, and then fills the
        remaining slots with the other parent's operations in order, avoiding duplicates.
        """
        return self._crossover_template(
            ind1,
            ind2,
            lambda r1, r2: self._cx_job_level(r1, r2),
            1,
        )

    def _cx_job_level(
        self, rhs1: RHS, rhs2: RHS
    ) -> tuple[list[tuple[int, int] | None], list[tuple[int, int] | None]]:
        """Job Level crossover core logic for right-hand side of chromosomes.
        As it is specific for this problem solution design,
        it is treated specifically"""

        def _fill_rest(new_rhs, jobs_set, other_rhs):
            idx = 0
            for gene in other_rhs:
                while idx < len(new_rhs) and new_rhs[idx] is not None:
                    idx += 1
                if gene not in jobs_set:
                    new_rhs[idx] = gene
                    idx += 1

        # Step 1: Choose a job from parent 1 arbitrarily
        chosen_job1 = random.choice([gene[0] for gene in rhs1])

        # Step 2: Copy all operations of the chosen job from parent 1 to child 1,
        # preserving their positions
        new_rhs1 = [gene if gene[0] == chosen_job1 else None for gene in rhs1]

        # Step 3: Complete child 1 with the remaining operations from parent 2,
        # preserving the order and avoiding duplicates
        new_rhs1_jobs = {g for g in new_rhs1 if g is not None}
        _fill_rest(new_rhs1, new_rhs1_jobs, rhs2)

        # Step 4: Repeat the process for child 2, but choose the job from parent 2
        chosen_job2 = random.choice([gene[0] for gene in rhs2])
        new_rhs2 = [gene if gene[0] == chosen_job2 else None for gene in rhs2]

        # fill the rest with operations from parent 1
        child2_jobs = {g for g in new_rhs2 if g is not None}
        _fill_rest(new_rhs2, child2_jobs, rhs1)

        return new_rhs1, new_rhs2

    # --------- MUTATION METHODS ---------
    def sstm_lhs(self, ind: list) -> List[Chromosome,]:
        """SSTM mutation for left-hand side of chromosomes."""
        return self._mutation_template(ind, self.flat_mutation_operators.sstm, 0)

    def shuffle_rhs(self, ind: list) -> List[Chromosome,]:
        """Shuffle mutation for right-hand side of chromosomes."""
        return self._mutation_template(
            ind,
            lambda rhs: self._rhs_mutation_func(rhs, tools.mutShuffleIndexes),
            1,
        )
