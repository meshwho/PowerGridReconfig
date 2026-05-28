from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from grid_topology_ai.config import GridConfig
from grid_topology_ai.legacy_pandapower.limit_calibrator import calibrate_line_limits_from_base_case
from grid_topology_ai.metrics import GridMetrics, compute_grid_metrics
from grid_topology_ai.network_factory import clone_network, create_network, run_power_flow


@dataclass
class EmergencyScenario:
    """
    One generated emergency scenario.

    This object contains:
    - the pandapower network after the emergency;
    - information about the initial disturbance;
    - metrics describing the emergency state.

    This is not yet a transition.
    A transition will be created later when we apply an action:
        scenario_state -> action -> next_state -> reward
    """

    scenario_id: int
    net: object
    network_name: str
    load_scale: float
    failed_line: int
    metrics: GridMetrics


class ScenarioGenerator:
    """
    Generator of emergency grid states.

    Main idea:
    We start from a calibrated base network and create N-1 outage scenarios.

    Current disturbance model:
    1. Scale all loads by a random factor.
    2. Disconnect one random line.
    3. Run AC power flow.
    4. Keep the scenario only if it produces a useful overload.

    Why this class exists:
    Later we will add more scenario types:
    - generator outage;
    - transformer outage;
    - local load growth;
    - renewable generation uncertainty;
    - combined N-k disturbances.
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.base_net = create_network(config.network_name)

        self._prepare_base_network()

    def _prepare_base_network(self) -> None:
        """
        Prepare the base network.

        Steps:
        1. Run raw power flow.
        2. Calibrate line limits from the base case.
        3. Run power flow again with calibrated limits.

        Important:
        Calibration does not change the physical flows.
        It changes only max_i_ka, therefore loading_percent becomes meaningful
        for our emergency switching task.
        """

        converged = run_power_flow(self.base_net)

        if not converged:
            raise RuntimeError(
                f"Base network {self.config.network_name} did not converge "
                "before line limit calibration."
            )

        if self.config.calibrate_line_limits:
            calibrate_line_limits_from_base_case(self.base_net, self.config)

            converged = run_power_flow(self.base_net)

            if not converged:
                raise RuntimeError(
                    f"Base network {self.config.network_name} did not converge "
                    "after line limit calibration."
                )

    def generate_one(self, scenario_id: int) -> Optional[EmergencyScenario]:
        """
        Try to generate one useful emergency scenario.

        Returns
        -------
        EmergencyScenario or None
            Returns a scenario if the random disturbance creates an overloaded
            but power-flow-convergent state. Otherwise returns None.

        Why None is allowed:
        Many random outages are not interesting. For example, the grid may remain
        safe after an outage. We do not want to train emergency control on boring
        cases where no action is needed.
        """

        net = clone_network(self.base_net)

        load_scale = self._sample_load_scale()
        failed_line = self._sample_line_outage(net)

        self._scale_loads(net, load_scale)
        self._disconnect_line(net, failed_line)

        converged = run_power_flow(net)
        metrics = compute_grid_metrics(net, self.config, converged)

        if not self._is_useful_emergency(metrics):
            return None

        return EmergencyScenario(
            scenario_id=scenario_id,
            net=net,
            network_name=self.config.network_name,
            load_scale=load_scale,
            failed_line=failed_line,
            metrics=metrics,
        )

    def generate_many(self, num_scenarios: int) -> list[EmergencyScenario]:
        """
        Generate a list of useful emergency scenarios.

        The generator may need many attempts because not every random line outage
        produces an overload.
        """

        scenarios: list[EmergencyScenario] = []

        attempts = 0

        while (
            len(scenarios) < num_scenarios
            and attempts < self.config.max_scenario_attempts
        ):
            attempts += 1

            scenario = self.generate_one(scenario_id=len(scenarios))

            if scenario is not None:
                scenarios.append(scenario)

                print(
                    f"Scenario {scenario.scenario_id:04d}: "
                    f"load_scale={scenario.load_scale:.3f}, "
                    f"failed_line={scenario.failed_line}, "
                    f"max_loading={scenario.metrics.max_line_loading_percent:.2f}%, "
                    f"overloaded_lines={scenario.metrics.num_overloaded_lines}"
                )

        if len(scenarios) < num_scenarios:
            print(
                f"Warning: requested {num_scenarios} scenarios, "
                f"but generated only {len(scenarios)} after {attempts} attempts."
            )

        return scenarios

    def _sample_load_scale(self) -> float:
        """
        Sample a random global load scale.

        For the first MVP we scale all loads together.
        Later we will replace this with regional and stochastic load profiles.
        """

        return float(
            self.rng.uniform(
                self.config.min_load_scale,
                self.config.max_load_scale,
            )
        )

    def _sample_line_outage(self, net) -> int:
        """
        Sample one line that is currently in service.

        We only choose from active lines.
        """

        active_lines = net.line.index[net.line["in_service"].astype(bool)].to_numpy()

        if len(active_lines) == 0:
            raise RuntimeError("No active lines available for outage sampling.")

        return int(self.rng.choice(active_lines))

    @staticmethod
    def _scale_loads(net, load_scale: float) -> None:
        """
        Scale all loads in the network.

        We scale both active and reactive load:
        - p_mw
        - q_mvar

        This keeps the load power factor approximately unchanged.
        """

        net.load["p_mw"] = net.load["p_mw"] * load_scale
        net.load["q_mvar"] = net.load["q_mvar"] * load_scale

    @staticmethod
    def _disconnect_line(net, line_id: int) -> None:
        """
        Disconnect one line.

        This represents the initial N-1 contingency.
        """

        net.line.at[line_id, "in_service"] = False

    def _is_useful_emergency(self, metrics: GridMetrics) -> bool:
        """
        Decide whether a generated scenario is useful for training.

        For now a useful scenario must:
        - have converged AC power flow;
        - be acceptable in the hard sense;
        - have at least one soft line overload.

        We do not require voltage to be perfect, because the base IEEE 118 case
        already has small soft voltage violations.
        """

        if not metrics.converged:
            return False

        if not metrics.is_acceptable:
            return False

        if (
            metrics.max_line_loading_percent
            < self.config.useful_scenario_min_loading_percent
        ):
            return False

        if metrics.num_overloaded_lines <= 0:
            return False

        return True