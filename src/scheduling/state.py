"""
State containers and simple builders (routes, precedences, time arrays).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Iterable
import numpy as np


# --- DTOs --------------------------------------------------------------------


@dataclass(frozen=True)
class StaticData:
    """
    Immutable view of parameters used during decoding.

    Args:
        - sequence: job -> list of machines in order
        - p_times: processing times [M, J]
        - s_times: setup times, either [M, J] (indep) or [M, J+1, J+1] (dep, with +1 shift)
        - n_lots: lots per job
        - shift_time: length of a shift
        - is_setup_dependent: flag
        - is_shift_constraints: flag
    """

    sequence: dict
    p_times: dict
    s_times: dict
    n_lots: int
    shift_time: int
    is_setup_dependent: bool
    is_shift_constraints: bool


@dataclass
class DynamicState:
    """
    Mutable scheduling state.

    Args:
        - setup_start: [M, J, U]
        - completion: [M, J, U]
        - routes: {(job, lot): remaining machine list}
        - precedences: {machine: [(job, lot), ...]}
        - lot_sizes: flattened vector by (job, lot)
    """

    setup_start: np.ndarray
    completion: np.ndarray
    routes: Dict[Tuple[int, int], List[int]]
    precedences: Dict[int, List[Tuple[int, int]]]
    lot_sizes: np.ndarray


@dataclass(frozen=True)
class Cursor:
    """
    Current operation indices.

    Args:
        - job: current job id
        - lot: current lot id
        - machine: current machine id
    """

    job: int
    lot: int
    machine: int


# --- Builders ----------------------------------------------------------------


def build_routes(
    sequence: dict,
    jobs: Iterable,
    lots: Iterable,
) -> Dict[Tuple[int, int], List[int]]:
    return {(j, u): list(sequence[j]) for j in jobs for u in lots}


def build_precedences(
    machines: Iterable,
    is_setup_dependent: bool,
) -> Dict[int, List[Tuple[int, int]]]:
    preds: Dict[int, List[Tuple[int, int]]] = {}
    for m in machines:
        preds[m] = [(-1, 0)] if is_setup_dependent else []
    return preds


def build_time_arrays(
    n_machines: int, n_jobs: int, n_lots: int
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((n_machines, n_jobs, n_lots), 0, dtype=int),
        np.full((n_machines, n_jobs, n_lots), 0, dtype=int),
    )
