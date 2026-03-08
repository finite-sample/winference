"""
winference: Win rate calibration under non-transitivity.

Two approaches to calibrating win rates when preferences aren't transitive:
  A) Hodge decomposition — separate transitive signal from cyclic residual,
     calibrate only the transitive part.
  B) Heterogeneous groups — find prompt categories where transitivity holds
     within each group, calibrate per group, compose for any target distribution.
"""

from importlib.metadata import PackageNotFoundError, version

from winference.bradley_terry import BradleyTerry
from winference.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_diagram,
)
from winference.groups import GroupCalibrator, GroupTest
from winference.hodge import HodgeDecomposition
from winference.simulate import (
    simulate_heterogeneous,
    simulate_rock_paper_scissors,
    simulate_transitive,
)
from winference.tournament import TournamentGraph

try:
    __version__ = version("winference")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "BradleyTerry",
    "GroupCalibrator",
    "GroupTest",
    "HodgeDecomposition",
    "TournamentGraph",
    "brier_score",
    "expected_calibration_error",
    "log_loss",
    "reliability_diagram",
    "simulate_heterogeneous",
    "simulate_rock_paper_scissors",
    "simulate_transitive",
]
