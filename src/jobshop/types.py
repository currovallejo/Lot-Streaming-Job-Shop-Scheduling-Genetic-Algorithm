"""
Type aliases for job shop scheduling parameters.
"""

from typing import Dict, List, Tuple, Union

# Basic identifiers
MachineId = int
JobId = int
LotId = int

# Time-related types
ProcessingTime = int
SetupTime = int

# Dictionary types for parameters
ProcessingTimes = Dict[Tuple[MachineId, JobId], ProcessingTime]
JobSequences = Dict[JobId, List[MachineId]]
SetupTimes = Dict[
    Union[Tuple[MachineId, JobId], Tuple[MachineId, JobId, JobId]], SetupTime
]
Demand = Dict[JobId, int]
