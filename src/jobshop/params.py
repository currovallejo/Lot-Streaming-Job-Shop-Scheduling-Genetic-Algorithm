"""
Job Shop parameter generation and container for Lot Streaming scheduling.

This module defines classes to build and manage job shop parameters with clear
separation between data containers, random generation, and reporting.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejogt
Github: https://github.com/currovallejog
"""

import numpy as np
from dataclasses import dataclass
from typing import Sequence, Dict, Any

from shared import utils
from .types import ProcessingTimes, SetupTimes, JobSequences, Demand
from . import reporting


@dataclass(frozen=True)
class JobShopData:
    """Immutable container for job shop parameters."""

    machines: range
    jobs: range
    lots: range
    p_times: ProcessingTimes
    seq: JobSequences
    setup: SetupTimes
    demand: Demand
    shift_time: int
    is_setup_dependent: bool
    seed: int  # Metadata for reproducibility
    t_span: tuple
    t_span_setup: tuple
    shift_constraints: bool


class JobShopParams:
    """Base class for job-shop parameters (backward compatibility)."""

    def __init__(
        self,
        machines: Sequence,
        jobs: Sequence,
        p_times: ProcessingTimes,
        seq: JobSequences,
        setup: SetupTimes,
        lots: Sequence,
    ):
        """Legacy constructor for backward compatibility."""
        self.machines = machines
        self.jobs = jobs
        self.p_times = p_times
        self.seq = seq
        self.setup = setup
        self.lots = lots


class JobShopRandomParams(JobShopParams):
    """Main facade for generating random job shop parameters from configuration."""

    def __init__(self, config_path: str = "config/jobshop/js_params_01.yaml"):
        """Generate job-shop parameters from YAML config.

        Args:
            config_path: Path to YAML configuration file
        """
        self._config = utils.load_config(config_path)
        self._data = self._generate_jobshop_data(self._config)

        # Expose data attributes for backward compatibility
        super().__init__(
            machines=self._data.machines,
            jobs=self._data.jobs,
            p_times=self._data.p_times,
            seq=self._data.seq,
            setup=self._data.setup,
            lots=self._data.lots,
        )

        # Additional attributes expected by existing code
        self.demand = self._data.demand
        self.shift_time = self._data.shift_time
        self.n_machines = len(self._data.machines)
        self.n_jobs = len(self._data.jobs)
        self.n_lots = len(self._data.lots)
        self.seed = self._data.seed
        self.t_span = self._data.t_span
        self.t_span_setup = self._data.t_span_setup
        self.is_setup_dependent = self._data.is_setup_dependent
        self.shift_constraints = self._data.shift_constraints

    @property
    def data(self) -> JobShopData:
        """Access to immutable data container."""
        return self._data

    def _generate_jobshop_data(self, config: Dict[str, Any]) -> JobShopData:
        """Generate job shop data from configuration"""
        # Extract configuration
        job_shop_config = config["job_shop"]
        n_machines = job_shop_config["n_machines"]
        n_jobs = job_shop_config["n_jobs"]
        n_lots = job_shop_config["n_lots"]

        t_span = tuple(config["processing_times"]["t_span"])
        t_span_setup = tuple(config["setup_times"]["t_span_setup"])
        seed = config["random"]["seed"]
        is_setup_dependent = config["setup_times"]["setup_mode"] == "dependent"
        shift_time = config["shift"]["shift_time"]
        shift_constraints = config["shift"]["shift_constraint"]

        # Generate ranges
        machines = range(n_machines)
        jobs = range(n_jobs)
        lots = range(n_lots)

        # Generate random data
        p_times = self._generate_processing_times(machines, jobs, t_span, seed)
        seq = self._generate_job_sequences(machines, jobs, seed)
        setup = self._generate_setup_times(
            machines, jobs, t_span_setup, seed, is_setup_dependent
        )

        # Fixed demand
        demand = {
            i: config["fixed_params"]["demand_per_job"] for i in range(0, n_jobs + 1)
        }

        return JobShopData(
            machines=machines,
            jobs=jobs,
            lots=lots,
            p_times=p_times,
            seq=seq,
            setup=setup,
            demand=demand,
            shift_time=shift_time,
            is_setup_dependent=is_setup_dependent,
            seed=seed,
            t_span=t_span,
            t_span_setup=t_span_setup,
            shift_constraints=shift_constraints,
        )

    def _generate_processing_times(
        self, machines: range, jobs: range, t_span: tuple, seed: int
    ) -> ProcessingTimes:
        """Generate random processing times"""
        np.random.seed(seed)
        times = np.arange(t_span[0], t_span[1])
        return {(m, j): np.random.choice(times) for m in machines for j in jobs}

    def _generate_job_sequences(
        self, machines: range, jobs: range, seed: int
    ) -> JobSequences:
        """Generate random machine sequences for each job"""
        np.random.seed(seed)
        return {j: self._generate_single_job_sequence(machines) for j in jobs}

    def _generate_single_job_sequence(self, machines: range) -> list:
        """Generate a random sequence of machines for one job."""
        sequence_length = np.random.randint(1, len(machines) + 1)
        sequence = np.random.choice(machines, size=sequence_length, replace=False)
        return sequence.astype(int).tolist()

    def _generate_setup_times(
        self,
        machines: range,
        jobs: range,
        t_span_setup: tuple,
        seed: int,
        is_sequence_dependent: bool,
    ) -> SetupTimes:
        """Generate setup times (sequence-dependent or independent)."""
        if is_sequence_dependent:
            return self._generate_sequence_dependent_setup(
                machines, jobs, t_span_setup, seed
            )
        else:
            return self._generate_sequence_independent_setup(
                machines, jobs, t_span_setup, seed
            )

    def _generate_sequence_independent_setup(
        self, machines: range, jobs: range, t_span_setup: tuple, seed: int
    ) -> SetupTimes:
        """Generate sequence-independent setup times."""
        np.random.seed(seed)
        times = np.arange(t_span_setup[0], t_span_setup[1])
        return {(m, j): np.random.choice(times) for m in machines for j in jobs}

    def _generate_sequence_dependent_setup(
        self, machines: range, jobs: range, t_span_setup: tuple, seed: int
    ) -> SetupTimes:
        """Generate sequence-dependent setup times with dummy jobs."""
        np.random.seed(seed)
        times = np.arange(t_span_setup[0], t_span_setup[1])
        setup_times: SetupTimes = {}  # Add explicit type annotation

        # Extended jobs include dummy jobs (0 and n_jobs+2)
        jobs_extended = list(range(0, jobs[-1] + 3))

        for m in machines:
            for j in jobs_extended:
                if j == 0 or j == jobs_extended[-1]:  # dummy jobs
                    for k in jobs_extended:
                        setup_times[(m, j, k)] = 0
                else:
                    setup_times[(m, j, 0)] = 50  # setup from dummy job
                    for k in jobs_extended:
                        if j == k:
                            setup_times[(m, j, k)] = 0  # no setup for same job
                        elif k != 0:
                            setup_times[(m, j, k)] = np.random.choice(times)

        return setup_times

    def print_jobshop_params(self) -> None:
        """Print all job shop parameters (convenience method)."""
        reporting.print_jobshop_params(
            self._data.machines,  # mypy check
            self._data.jobs,  # mypy check
            self._data.lots,  # mypy check
            self.p_times,
            self.seq,
            self.setup,
            self.demand,
            self.shift_time,
            self.is_setup_dependent,
        )
