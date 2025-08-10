"""Test consistency between legacy and new ChromosomeDecoder implementations."""

import numpy as np

from src.legacy import ChromosomeDecoder
from src.scheduling import Scheduler


def test_decoder_v1_vs_v2_consistency(problem_params, chromosome):
    # Legacy decoder
    legacy = ChromosomeDecoder(problem_params)
    mk1, pen1, y1, c1, semi1 = legacy.decode(chromosome)

    # New decoder
    new = Scheduler(problem_params)
    mk2, pen2, y2, c2, semi2 = new.decode(chromosome)

    # Basic equality checks
    assert mk1 == mk2, "Makespan differs between legacy and v2"
    assert pen1 == pen2, "Penalty differs between legacy and v2"

    # Arrays: setup_start (y) and completion (c)
    np.testing.assert_array_equal(y1, y2, err_msg="Setup-start times differ")
    np.testing.assert_array_equal(c1, c2, err_msg="Completion times differ")

    # Semi-encoded solution (lot sizes)
    assert isinstance(semi2, list) and len(semi2) == 2
    np.testing.assert_array_equal(
        np.asarray(semi1[0], dtype=int),
        np.asarray(semi2[0], dtype=int),
        err_msg="Lot sizes differ",
    )

    # - No negative times
    assert (y2 >= 0).all() and (c2 >= 0).all()
    # - Completion always after (or at) setup-start
    assert (c2 >= y2).all()
