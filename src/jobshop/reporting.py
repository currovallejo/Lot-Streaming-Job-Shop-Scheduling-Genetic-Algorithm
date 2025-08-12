"""
Reporting functions for job shop parameters.

This module provides pure functions to generate formatted reports and DataFrames
from job shop parameters without side effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union

from .types import ProcessingTimes, SetupTimes, JobSequences


def processing_times_dataframe(p_times: ProcessingTimes) -> pd.DataFrame:
    """Create a DataFrame showing processing times (machines as rows, jobs as columns)."""
    if not p_times:
        return pd.DataFrame()

    max_job = max(key[1] for key in p_times.keys())
    max_machine = max(key[0] for key in p_times.keys())

    matrix = np.zeros((max_machine + 1, max_job + 1), dtype=int)

    for (machine, job), time in p_times.items():
        matrix[machine][job] = time

    jobs = [f"Job {i}" for i in range(max_job + 1)]
    machines = [f"Machine {i}" for i in range(max_machine + 1)]

    return pd.DataFrame(matrix, columns=jobs, index=machines)


def setup_times_dataframe(
    setup: SetupTimes, jobs: range, is_sequence_dependent: bool
) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Create a DataFrame showing setup times.

    Returns:
        - For sequence-independent: pd.DataFrame
        - For sequence-dependent: Dict[int, pd.DataFrame] (one DataFrame per machine)
    """
    if not setup:
        return pd.DataFrame()

    if is_sequence_dependent:
        return _sequence_dependent_setup_dataframe(setup, jobs)
    else:
        return _sequence_independent_setup_dataframe(setup)


def _sequence_independent_setup_dataframe(setup: SetupTimes) -> pd.DataFrame:
    """Create DataFrame for sequence-independent setup times."""
    max_job = max(key[1] for key in setup.keys() if len(key) == 2)
    max_machine = max(key[0] for key in setup.keys() if len(key) == 2)

    matrix = np.zeros((max_machine + 1, max_job + 1), dtype=int)

    for key, value in setup.items():
        if len(key) == 2:  # sequence-independent format
            machine, job = key
            matrix[machine][job] = value

    jobs = [f"Job {i}" for i in range(max_job + 1)]
    machines = [f"Machine {i}" for i in range(max_machine + 1)]

    return pd.DataFrame(matrix, columns=jobs, index=machines)


def _sequence_dependent_setup_dataframe(
    setup: SetupTimes, jobs: range
) -> Dict[int, pd.DataFrame]:
    """Create DataFrames for sequence-dependent setup times (one per machine)."""
    machines = set(key[0] for key in setup.keys() if len(key) == 3)
    result = {}

    for machine in machines:
        n_jobs = len(jobs)
        matrix = np.zeros((n_jobs + 1, n_jobs), dtype=int)

        for key, value in setup.items():
            if len(key) == 3 and key[0] == machine:
                _, successor, predecessor = key
                if (
                    successor != 0
                    and successor != jobs[-1] + 2
                    and predecessor != jobs[-1] + 2
                ):
                    matrix[predecessor][successor - 1] = value

        successor_jobs = [f"Job {i}" for i in range(n_jobs)]
        predecessor_jobs = [f"Job {i-1}" for i in range(n_jobs + 1)]

        result[machine] = pd.DataFrame(
            matrix, columns=successor_jobs, index=predecessor_jobs
        )

    return result


def job_sequences_summary(sequences: JobSequences) -> pd.DataFrame:
    """Create a summary DataFrame of job sequences."""
    data = []
    for job_id, sequence in sequences.items():
        data.append(
            {
                "Job": job_id,
                "Sequence": " -> ".join(f"M{m}" for m in sequence),
                "Length": len(sequence),
            }
        )

    return pd.DataFrame(data)


def generate_summary_report(
    machines: range,
    jobs: range,
    lots: range,
    demand: Dict[int, int],
    shift_time: int,
    is_setup_dependent: bool,
) -> str:
    """Generate a text summary of job shop parameters."""
    lines = [
        "=== Job Shop Parameters Summary ===",
        f"Machines: {len(machines)} (indices: {list(machines)})",
        f"Jobs: {len(jobs)} (indices: {list(jobs)})",
        f"Lots: {len(lots)} (indices: {list(lots)})",
        f"Demand per job: {demand.get(0, 'N/A')}",
        f"Shift time: {shift_time}",
        f"Setup mode: {'Sequence-dependent' if is_setup_dependent else 'Sequence-independent'}",
        "=" * 35,
    ]
    return "\n".join(lines)


def print_jobshop_params(
    machines: range,
    jobs: range,
    lots: range,
    p_times: ProcessingTimes,
    seq: JobSequences,
    setup: SetupTimes,
    demand: Dict[int, int],
    shift_time: int,
    is_setup_dependent: bool,
) -> None:
    """Print all job shop parameters in a formatted way"""

    # Print summary
    summary = generate_summary_report(
        machines, jobs, lots, demand, shift_time, is_setup_dependent
    )
    print(summary)

    # Print processing times
    print("\nProcessing Times:")
    proc_times_df = processing_times_dataframe(p_times)
    print(proc_times_df)

    # Print job sequences
    print("\nJob Sequences:")
    sequences_df = job_sequences_summary(seq)
    print(sequences_df)

    # Print setup times
    print("\nSetup Times:")
    setup_result = setup_times_dataframe(setup, jobs, is_setup_dependent)

    if isinstance(setup_result, dict):
        # Sequence-dependent setup times (one table per machine)
        for machine_id, df in setup_result.items():
            print(f"\nMachine {machine_id}:")
            print(df)
    else:
        # Sequence-independent setup times (single table)
        print(setup_result)

    print("\n" + "=" * 50)
