"""
Scheduling rules for Lot Streaming Job Shop Scheduling Problem.

This module implements scheduling rules and time calculation functions for the chromosome
decoder. It provides logic for determining start times, completion times, and handling
shift constraints and setup dependencies while building a semi-active schedule.

Author: Francisco Vallejo
LinkedIn: www.linkedin.com/in/franciscovallejog
Github: https://github.com/currovallejog
"""

from __future__ import annotations

from .state import StaticData, DynamicState, Cursor


# --- internal helpers --------------------------------------------------------


def _is_empty_machine(st: StaticData, dyn: DynamicState, m: int) -> bool:
    """indicates if a machine is empty (no operations scheduled)."""
    return (
        (dyn.precedences[m] == [(-1, 0)])
        if st.is_setup_dependent
        else (dyn.precedences[m] == [])
    )


def _is_first_machine(st: StaticData, j: int, m: int) -> bool:
    """indicates if the machine is the first in the job sequence."""
    return m == st.sequence[j][0]


def _completion_prev_machine(st: StaticData, dyn: DynamicState, cur: Cursor) -> int:
    """Calculate the completion time of the previous machine in the job sequence."""
    r = st.sequence[cur.job]
    idx = r.index(cur.machine)
    prev_m = r[idx - 1]
    return int(dyn.completion[prev_m, cur.job, cur.lot])


def _completion_current_machine(dyn: DynamicState, m: int) -> int:
    """Calculate the completion time of the current machine."""
    prev_j, prev_u = dyn.precedences[m][-1]
    return int(dyn.completion[m, prev_j, prev_u])


def lot_processing_duration(st: StaticData, dyn: DynamicState, cur: Cursor) -> int:
    """Calculate the processing duration of the current lot on the current machine."""
    qty = int(dyn.lot_sizes[st.n_lots * cur.job + cur.lot])
    if st.is_setup_dependent:
        prev_j = dyn.precedences[cur.machine][-1][0]  # -1 at start
        return int(
            st.s_times[cur.machine, cur.job + 1, prev_j + 1]
            + st.p_times[cur.machine, cur.job] * qty
        )
    else:
        return int(
            st.s_times[cur.machine, cur.job] + st.p_times[cur.machine, cur.job] * qty
        )


# --- public rules ------------------------------------------------------------


def start_time_no_shifts(st: StaticData, dyn: DynamicState, cur: Cursor) -> int:
    """
    Calculate the start time of the current lot on the current machine without shift constraints.
    Args:
        st (StaticData): Static data containing job shop parameters.
        dyn (DynamicState): Dynamic state containing current scheduling information.
        cur (Cursor): Current cursor position in the job sequence.
    Returns:
        int: The start time for the current lot on the current machine.
    """
    is_first = _is_first_machine(st, cur.job, cur.machine)
    is_empty = _is_empty_machine(st, dyn, cur.machine)

    if is_first and is_empty:
        return 0
    if is_first and not is_empty:
        return _completion_current_machine(dyn, cur.machine)
    if (not is_first) and is_empty:
        return _completion_prev_machine(st, dyn, cur)
    return max(
        _completion_current_machine(dyn, cur.machine),
        _completion_prev_machine(st, dyn, cur),
    )


def completion_time_setup_dependent(
    st: StaticData, dyn: DynamicState, cur: Cursor, s_start: int
) -> int:
    """
    Calculate the completion time of the current lot on the current machine with setup dependency.
    Args:
        st (StaticData): Static data containing job shop parameters.
        dyn (DynamicState): Dynamic state containing current scheduling information.
        cur (Cursor): Current cursor position in the job sequence.
        s_start (int): The start time for the current lot on the current machine.
    Returns:
        int: The completion time for the current lot on the current machine.
    """
    prev_j = dyn.precedences[cur.machine][-1][0]  # -1 at start
    qty = int(dyn.lot_sizes[st.n_lots * cur.job + cur.lot])
    return int(
        s_start
        + st.s_times[cur.machine, cur.job + 1, prev_j + 1]
        + st.p_times[cur.machine, cur.job] * qty
    )


def completion_time_setup_independent(
    st: StaticData, dyn: DynamicState, cur: Cursor, s_start: int
) -> int:
    """
    Calculate the completion time of the current lot on the current machine without setup dependency.
    Args:
        st (StaticData): Static data containing job shop parameters.
        dyn (DynamicState): Dynamic state containing current scheduling information.
        cur (Cursor): Current cursor position in the job sequence.
        s_start (int): The start time for the current lot on the current machine.
    Returns:
        int: The completion time for the current lot on the current machine.
    """
    qty = int(dyn.lot_sizes[st.n_lots * cur.job + cur.lot])
    return int(
        s_start
        + st.s_times[cur.machine, cur.job]
        + st.p_times[cur.machine, cur.job] * qty
    )


def is_big_lot_duration(st: StaticData, dyn: DynamicState, cur: Cursor) -> bool:
    """
    Check if the lot processing duration exceeds the shift time.
    Args:
        st (StaticData): Static data containing job shop parameters.
        dyn (DynamicState): Dynamic state containing current scheduling information.
        cur (Cursor): Current cursor position in the job sequence.
    Returns:
        bool: True if the lot processing duration exceeds the shift time, False otherwise.
    """
    return lot_processing_duration(st, dyn, cur) > int(st.shift_time)


def start_time_with_shifts(st: StaticData, dyn: DynamicState, cur: Cursor) -> int:
    """
    Calculate the start time of the current lot on the current machine with shift constraints.
    Args:
        st (StaticData): Static data containing job shop parameters.
        dyn (DynamicState): Dynamic state containing current scheduling information.
        cur (Cursor): Current cursor position in the job sequence.
    Returns:
        int: The start time for the current lot on the current machine.
    """
    if is_big_lot_duration(st, dyn, cur):
        return start_time_no_shifts(st, dyn, cur)

    is_first = _is_first_machine(st, cur.job, cur.machine)
    is_empty = _is_empty_machine(st, dyn, cur.machine)

    last_on_m = 0 if is_empty else _completion_current_machine(dyn, cur.machine)
    last_on_prev = 0 if is_first else _completion_prev_machine(st, dyn, cur)

    dur = lot_processing_duration(st, dyn, cur)

    if is_first:
        if is_empty:
            return 0
        rem = last_on_m % st.shift_time
        return (
            last_on_m
            if rem + dur <= st.shift_time
            else (last_on_m // st.shift_time + 1) * st.shift_time
        )

    if is_empty:
        rem = last_on_prev % st.shift_time
        return (
            last_on_prev
            if rem + dur <= st.shift_time
            else (last_on_prev // st.shift_time + 1) * st.shift_time
        )

    if last_on_prev >= last_on_m:
        rem = last_on_prev % st.shift_time
        return (
            last_on_prev
            if rem + dur <= st.shift_time
            else (last_on_prev // st.shift_time + 1) * st.shift_time
        )
    else:
        rem = last_on_m % st.shift_time
        return (
            last_on_m
            if rem + dur <= st.shift_time
            else (last_on_m // st.shift_time + 1) * st.shift_time
        )
