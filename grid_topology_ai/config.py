from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridConfig:
    """
    Central configuration for the topology switching project.

    We use two types of limits:

    1. Soft limits:
       Violating them is undesirable and should be penalized.

    2. Hard limits:
       Violating them means the state is dangerous.
    """

    # Main experimental network.
    network_name: str = "case118"

    # -----------------------------
    # Voltage limits
    # -----------------------------

    # Soft voltage limits.
    vm_min_soft_pu: float = 0.95
    vm_max_soft_pu: float = 1.05

    # Hard voltage limits.
    vm_min_hard_pu: float = 0.90
    vm_max_hard_pu: float = 1.10

    # -----------------------------
    # Line loading limits
    # -----------------------------

    # Soft line loading limit.
    # Above this value we start penalizing overloads.
    line_loading_soft_limit_percent: float = 100.0

    # Hard line loading limit.
    # Above this value the state is considered dangerous.
    line_loading_hard_limit_percent: float = 120.0

    # Minimum overload required to treat a generated scenario as useful.
    min_overload_for_scenario_percent: float = 100.0


    # -----------------------------
    # Line limit calibration
    # -----------------------------

    # Standard IEEE test cases often have very large thermal limits.
    # As a result, base-case line loading can be only 1-5%.
    #
    # For topology switching learning, this is not useful.
    # We calibrate line limits so that the base-case loading is around
    # this target value.
    target_base_line_loading_percent: float = 55.0

    # Minimum allowed line current limit after calibration.
    # This prevents near-zero-flow lines from receiving unrealistically tiny limits.
    min_line_max_i_ka: float = 0.05

    # Maximum allowed line current limit after calibration.
    # This prevents extremely high-current lines from receiving too large limits.
    max_line_max_i_ka: float = 10.0

    # If True, check_network.py will show both raw and calibrated metrics.
    calibrate_line_limits: bool = True

    # -----------------------------
    # Emergency scenario generation
    # -----------------------------

    # Random load scaling range.
    # A value of 1.20 means all loads are multiplied by 1.20.
    min_load_scale: float = 1.05
    max_load_scale: float = 1.60

    # Maximum number of attempts to find useful emergency scenarios.
    # Not every random outage creates overloads, so we need multiple attempts.
    max_scenario_attempts: int = 1000

    # A scenario is useful if at least one line exceeds this loading.
    useful_scenario_min_loading_percent: float = 100.0

    # -----------------------------
    # Reproducibility
    # -----------------------------

    seed: int = 43

    # -----------------------------
    # Paths
    # -----------------------------

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    states_dir: Path = Path("data/states")