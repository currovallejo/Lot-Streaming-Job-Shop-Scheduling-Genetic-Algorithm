# --- Pandas → Domain mappers (validation + translation) ---

from __future__ import annotations

from typing import List
import pandas as pd

from .domain import OperationId, TimeWindow, ScheduledOperation


# --- Public API ---

def map_dataframe(df_results: pd.DataFrame) -> List[ScheduledOperation]:
    """
    Convert the scheduling results DataFrame into immutable domain objects.

    Args:
        df_results: DataFrame containing scheduling results with columns:
            - job
            - machine
            - lot
            - lot_size
            - setup_start_time
            - start_time
            - completion_time
    Returns:
        List of ScheduledOperation objects representing the scheduled operations.
    """
    if df_results is None:
        raise ValueError("df_results is None")

    # --- Validate schema ---
    required = {
        "job",
        "machine",
        "lot",
        "lot_size",
        "setup_start_time",
        "start_time",
        "completion_time",
    }
    missing = required - set(df_results.columns)
    if missing:
        raise ValueError(f"df_results is missing columns: {sorted(missing)}")

    # --- Normalize types (ensure ints) ---
    df = df_results.astype(int).reset_index(drop=True)

    # --- Map rows to domain ---
    ops: List[ScheduledOperation] = []
    for _, r in df.iterrows():
        op = ScheduledOperation(
            id=OperationId(
                job=int(r["job"]),
                machine=int(r["machine"]),
                lot=int(r["lot"]),
            ),
            time=TimeWindow(
                setup_start=int(r["setup_start_time"]),
                start=int(r["start_time"]),
                completion=int(r["completion_time"]),
            ),
            lot_size=int(r["lot_size"]),
        )
        ops.append(op)

    return ops
