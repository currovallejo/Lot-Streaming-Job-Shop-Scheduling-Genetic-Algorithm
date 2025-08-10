"""
DEFINITION OF CLASSES TO HANDLE JOBSHOP PARAMETERS
----------------------------------------------------------------------
- Classes, methods and functions are defined to facilitate the handling of jobshop
parameters in the script where the model is defined.
- Based in code written by Bruno Scalia C. F. Leite
"""

import numpy as np
from typing import Iterable
import pandas as pd

from shared import utils


def custom_serializer(obj):
    if isinstance(obj, tuple):
        return str(obj)
    return obj


class JobSequence(list):
    def prev(self, x):
        if self.is_first(x):
            return None
        else:
            i = self.index(x)
            return self[i - 1]

    def next(self, x):
        if self.is_last(x):
            return None
        else:
            i = self.index(x)
            return self[i + 1]

    def is_first(self, x):
        return x == self[0]

    def is_last(self, x):
        return x == self[-1]

    def swap(self, x, y):
        i = self.index(x)
        j = self.index(y)
        self[i] = y
        self[j] = x

    def append(self, __object) -> None:
        if __object not in self:
            super().append(__object)
        else:
            pass


class JobShopParams:
    def __init__(
        self,
        machines: Iterable,
        jobs: Iterable,
        p_times: dict,
        seq: dict,
        setup: dict,
        lots: Iterable,
    ):
        """White label class for job-shop parameters

        Parameters
        ----------
        machines : Iterable
            Set of machines

        jobs : Iterable
            Set of jobs

        p_times : dict
            Processing times indexed by pairs machine, job

        seq : dict
            Sequence of operations (machines) of each job

        demand : dict
            Demand of each job
        """
        self.machines = machines
        self.jobs = jobs
        self.p_times = p_times
        self.seq = seq
        self.setup = setup
        self.lots = lots


class JobShopRandomParams(JobShopParams):
    def __init__(self, config_path: str = "config/jobshop_config.yaml"):
        """Class for generating job-shop parameters from YAML config

        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file, by default "config/jobshop_config.yaml"
        """
        config = utils.load_config(config_path)

        # Extract high-level parameters from config
        n_machines = config["job_shop"]["n_machines"]
        n_jobs = config["job_shop"]["n_jobs"]
        n_lots = config["job_shop"]["n_lots"]

        self.n_machines = n_machines
        self.n_jobs = n_jobs
        self.n_lots = n_lots
        self.t_span = tuple(config["processing_times"]["t_span"])
        self.seed = config["random"]["seed"]
        self.t_span_setup = tuple(config["setup_times"]["t_span_setup"])
        self.is_setup_dependent = config["setup_times"]["setup_mode"] == "dependent"
        self.shift_constraints = config["shift"]["shift_constraint"]

        # Generate random job-shop data
        machines = np.arange(n_machines, dtype=int)
        jobs = np.arange(n_jobs)
        p_times = self._random_times(machines, jobs, self.t_span)
        lots = np.arange(n_lots, dtype=int)
        seq = self._random_sequences(machines, jobs)
        setup = self._random_setup(machines, jobs, self.t_span_setup)

        super().__init__(machines, jobs, p_times, seq, setup, lots)

        # Non random parameters from config
        self.demand = {
            i: config["fixed_params"]["demand_per_job"] for i in range(0, n_jobs + 1)
        }
        self.shift_time = config["shift"]["shift_time"]

    def _random_times(self, machines, jobs, t_span):
        """Generates random processing times for each job on each machine."""
        np.random.seed(self.seed)
        t = np.arange(t_span[0], t_span[1])
        return {(m, j): np.random.choice(t) for m in machines for j in jobs}

    def _random_sequences(self, machines, jobs):
        np.random.seed(self.seed)
        return {j: self._generate_random_sequence(machines) for j in jobs}

    def _generate_random_sequence(self, machines):
        """Generates a random sequence (route) of machines for a job."""
        # Decide on the length of the sequence
        # (can be any number between 1 and len(machines))
        sequence_length = np.random.randint(1, len(machines) + 1)

        # Randomly select machines for the sequence
        sequence = np.random.choice(machines, size=sequence_length, replace=False)
        sequence = sequence.astype(int)

        return JobSequence(sequence)

    def _random_setup(self, machines, jobs, t_span_setup):
        """Generates random setup times for each job on each machine."""
        if self.is_setup_dependent:
            return self._random_sequence_dependent_setup(machines, jobs, t_span_setup)
        else:
            # sequence independent setup times
            np.random.seed(self.seed)
            t = np.arange(
                t_span_setup[0], t_span_setup[1]
            )  # meto en un vector todos los tiempos posibles
            return {(m, j): np.random.choice(t) for m in machines for j in jobs}

    def _random_sequence_dependent_setup(
        self, machines, jobs, t_span_setup
    ):  # sequence dependent setup times
        """
        Generates random SEQUENCE DEPENDENT setup times for each job on each machine.

        2 more jobs are added to the list of jobs_extended: 0 and n_jobs+1 in order to
        fit with the random values generated for solving the MILP model. It is just to
        get the same solution and prove that the GA works.
        2 dummy jobs are strictly needed for the MILP modelling.
        j = successor
        k = predecessor
        """
        np.random.seed(self.seed)
        t = np.arange(
            t_span_setup[0], t_span_setup[1]
        )  # meto en un vector todos los tiempos posibles
        sd_setup_times = {}
        jobs_extended = list(range(0, jobs[-1] + 3))  # add dummy jobs
        for m in machines:
            for j in jobs_extended:
                if j == 0 or j == jobs_extended[-1]:  # for dummy jobs
                    for k in jobs_extended:
                        sd_setup_times[m, j, k] = 0
                else:
                    sd_setup_times[m, j, 0] = 50
                    for k in jobs_extended:
                        if j == k:
                            sd_setup_times[m, j, k] = 0
                        elif k != 0:
                            sd_setup_times[m, j, k] = np.random.choice(t)

        return sd_setup_times

    def print_jobshop_params(self, save_to_excel=False):
        """Prints the parameters of the job-shop problem"""
        print("Job-Shop Parameters:")
        self._print_params_machines()
        self._print_params_jobs()
        self._print_params_lots()
        self._print_params_demand()
        self._print_params_setup_times()
        self._print_params_processing_times()
        self._print_params_sequence()

    def _print_params_machines(self):
        print("[MACHINES]: \n", self.machines, "\n")

    def _print_params_jobs(self):
        print("[JOBS]: \n", self.jobs, "\n")

    def _print_params_lots(self):
        print("[BATCHES]: \n", self.lots, "\n")

    def _print_params_demand(self):
        print("[DEMAND]: \n", self.demand, "\n")

    def _print_params_processing_times(self):
        print(
            "[PROCESS TIMES]the working time associated with each job on each machine is:"
        )
        # Determine the dimensions of the matrix
        max_job = max(key[1] for key in self.p_times.keys())
        max_machine = max(key[0] for key in self.p_times.keys())

        # Create an empty matrix filled with zeros
        matrix = np.zeros((max_job + 1, max_machine + 1), dtype=int)

        # Fill the matrix with the given data
        for key, value in self.p_times.items():
            matrix[key[1]][key[0]] = value

        # Transpose the matrix to have jobs as rows and machines as columns
        transposed_matrix = matrix.T

        # Create a DataFrame with row and column labels
        jobs = [f"Job {i}" for i in range(max_job + 1)]
        machines = [f"Machine {j}" for j in range(max_machine + 1)]

        # Print the DataFrame
        print(pd.DataFrame(transposed_matrix, columns=jobs, index=machines), "\n")

    def _print_params_setup_times(self):
        if self.is_setup_dependent:
            self._print_params_sequence_dependent_setup_times()
        else:
            self._print_params_sequence_independent_setup_times()

    def _print_params_sequence_dependent_setup_times(self):
        print(
            "[SEQ DEPENDENT SETUP TIMES] row(k) = predecessor | column(j) = successor  \n IMPORTANT! indexes are 1 unit more. \n IMPORTANT! For job 0, index in setup dictionary is j=1 (successor) or k=1 (predecessor) \n the setup time associated with each job on each machine is:"
        )
        for m in self.machines:
            print("Machine ", m)
            n_columns = len(self.jobs)
            n_rows = len(self.jobs) + 1
            matrix = np.zeros((n_rows, n_columns), dtype=int)
            for key, value in self.setup.items():
                if key[0] == m:
                    if key[1] != 0 and key[1] != self.jobs[-1] + 2:
                        if key[2] != self.jobs[-1] + 2:
                            matrix[key[2]][key[1] - 1] = value

            # Create a DataFrame with row and column labels
            setup_jobs = [
                f"Job {i}" for i in range(n_columns)
            ]  # columns are successors jobs
            precedence_jobs = [
                f"Job {j-1}" for j in range(n_rows)
            ]  # rows are predecessors jobs

            # Print the DataFrame
            print(pd.DataFrame(matrix, columns=setup_jobs, index=precedence_jobs), "\n")

    def _print_params_sequence_independent_setup_times(self):
        print(
            "[SEQ INDEPENDENT SETUP TIMES] the setup time associated with each job on each machine is:"
        )
        # Determine the dimensions of the matrix
        max_job = max(key[1] for key in self.setup.keys())
        max_machine = max(key[0] for key in self.setup.keys())

        # Create an empty matrix filled with zeros
        matrix = np.zeros((max_machine + 1, max_job + 1), dtype=int)

        # Fill the matrix with the given data
        for key, value in self.setup.items():
            matrix[key[0]][key[1]] = value

        # Create a DataFrame with row and column labels
        jobs = [f"Job {i}" for i in range(max_job + 1)]
        machines = [f"Machine {j}" for j in range(max_machine + 1)]

        # Print the DataFrame
        print(pd.DataFrame(matrix, columns=jobs, index=machines), "\n")

    def _print_params_sequence(self):
        print("[SEQ] the sequence for each job is: ")

        trabajo_list = []
        seq_list = []
        for trabajo in self.seq:
            print(trabajo, "|", self.seq[trabajo])
            trabajo_list.append(trabajo)
            seq_list.append(self.seq[trabajo])
