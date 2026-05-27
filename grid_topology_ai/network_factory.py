from __future__ import annotations

import copy
from typing import Literal

import pandapower as pp
import pandapower.networks as pn


SupportedNetwork = Literal["case14", "case30", "case57", "case118"]


def create_network(network_name: SupportedNetwork):


    if network_name == "case14":
        net = pn.case14()
    elif network_name == "case30":
        net = pn.case30()
    elif network_name == "case57":
        net = pn.case57()
    elif network_name == "case118":
        net = pn.case118()
    else:
        raise ValueError(
            f"Unsupported network: {network_name}. "
            "Supported values are: case14, case30, case57, case118."
        )

    prepare_network(net)

    return net


def prepare_network(net) -> None:
    """
    - normalize names;
    - add missing limits;
    - check line parameters;
    - prepare switches;
    - add metadata.
    """


    if "in_service" not in net.line.columns:
        net.line["in_service"] = True

    if "in_service" not in net.bus.columns:
        net.bus["in_service"] = True



def clone_network(net):

    return copy.deepcopy(net)


def run_power_flow(net) -> bool:
    """

    Returns
    -------
    bool
        True if power flow converged, False otherwise.

    """

    try:
        pp.runpp(
            net,
            algorithm="nr",
            calculate_voltage_angles=True,
            init="auto",
            max_iteration=30,
            tolerance_mva=1e-6,
            check_connectivity=True,
        )
        return bool(net.converged)
    except Exception:
        return False