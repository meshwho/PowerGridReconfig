from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from pypower.api import ppoption, runpf
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    TAP,
    T_BUS,
)
import yaml
from pypower.idx_bus import (
    BASE_KV,
    BS,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    QD,
    VA,
    VM,
    VMAX,
    VMIN,
)
from pypower.idx_bus import PQ as BUS_TYPE_PQ
from pypower.idx_bus import PV as BUS_TYPE_PV
from pypower.idx_bus import REF as BUS_TYPE_REF
from pypower.idx_gen import (
    GEN_BUS,
    GEN_STATUS,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
)


def load_yaml_config(path: str | None) -> dict:
    if path is None:
        return {}

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a dictionary: {config_path}")

    return data


def get_nested(config: dict, keys: list[str], default):
    current = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current


def clone_ppc(ppc: dict) -> dict:
    cloned = {}

    for key, value in ppc.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = deepcopy(value)

    return cloned


def run_power_flow(
    ppc: dict,
    pf_alg: int,
    max_iter: int,
    fallback_to_newton: bool = True,
) -> tuple[dict | None, bool]:
    """
    Run PYPOWER power flow safely.

    Fast-decoupled methods may fail with a singular matrix for some emergency
    topologies. In dataset generation this should not crash the whole script:
    failed cases are skipped, or optionally retried with Newton-Raphson.
    """

    try:
        ppopt = ppoption(
            VERBOSE=0,
            OUT_ALL=0,
            PF_ALG=int(pf_alg),
            PF_MAX_IT=int(max_iter),
        )

        result_ppc, success = runpf(ppc, ppopt)

        if bool(success):
            return result_ppc, True

    except Exception:
        pass

    if not fallback_to_newton or int(pf_alg) == 1:
        return None, False

    try:
        ppopt = ppoption(
            VERBOSE=0,
            OUT_ALL=0,
            PF_ALG=1,
            PF_MAX_IT=int(max_iter),
        )

        result_ppc, success = runpf(ppc, ppopt)
        return result_ppc, bool(success)

    except Exception:
        return None, False

def branch_s_mva(branch: np.ndarray) -> np.ndarray:
    pf = branch[:, PF]
    qf = branch[:, QF]
    pt = branch[:, PT]
    qt = branch[:, QT]

    s_from = np.sqrt(pf * pf + qf * qf)
    s_to = np.sqrt(pt * pt + qt * qt)

    return np.maximum(s_from, s_to)


def build_static_branch_limits(
    base_result_ppc: dict,
    target_base_utilization: float,
    min_rate_mva: float,
) -> np.ndarray:
    branch = base_result_ppc["branch"]

    base_s = branch_s_mva(branch)
    raw_rate = branch[:, RATE_A].copy()

    derived_rate = np.maximum(
        base_s / max(float(target_base_utilization), 1e-6),
        float(min_rate_mva),
    )

    rate = np.where(raw_rate > 0.0, raw_rate, derived_rate)
    rate = np.maximum(rate, float(min_rate_mva))

    return rate.astype(float)


def apply_load_randomization(
    ppc: dict,
    rng: np.random.Generator,
    load_scale_min: float,
    load_scale_max: float,
    local_load_noise: float,
) -> None:
    bus = ppc["bus"]

    global_scale = rng.uniform(float(load_scale_min), float(load_scale_max))

    local_scale = rng.normal(
        loc=1.0,
        scale=float(local_load_noise),
        size=bus.shape[0],
    )

    local_scale = np.clip(local_scale, 0.75, 1.35)

    scale = global_scale * local_scale

    bus[:, PD] = bus[:, PD] * scale
    bus[:, QD] = bus[:, QD] * scale


def apply_branch_limits(ppc: dict, limits: np.ndarray) -> None:
    ppc["branch"][:, RATE_A] = limits
    ppc["branch"][:, RATE_B] = limits
    ppc["branch"][:, RATE_C] = limits


def apply_outages(
    ppc: dict,
    outage_branch_positions: list[int],
) -> None:
    for branch_pos in outage_branch_positions:
        ppc["branch"][int(branch_pos), BR_STATUS] = 0.0


def bus_type_one_hot(bus_type: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pq = (bus_type.astype(int) == BUS_TYPE_PQ).astype(float)
    pv = (bus_type.astype(int) == BUS_TYPE_PV).astype(float)
    ref = (bus_type.astype(int) == BUS_TYPE_REF).astype(float)

    return pq, pv, ref


def build_bus_dataframe(
    scenario_id: int,
    load_scenario_idx: float,
    result_ppc: dict,
) -> pd.DataFrame:
    bus = result_ppc["bus"]
    gen = result_ppc["gen"]

    bus_ids = bus[:, BUS_I].astype(int)

    pg_by_bus = {int(bus_id): 0.0 for bus_id in bus_ids}
    qg_by_bus = {int(bus_id): 0.0 for bus_id in bus_ids}

    for gen_row in gen:
        if float(gen_row[GEN_STATUS]) <= 0.0:
            continue

        bus_id = int(gen_row[GEN_BUS])
        pg_by_bus[bus_id] = pg_by_bus.get(bus_id, 0.0) + float(gen_row[PG])
        qg_by_bus[bus_id] = qg_by_bus.get(bus_id, 0.0) + float(gen_row[QG])

    pq, pv, ref = bus_type_one_hot(bus[:, BUS_TYPE])

    return pd.DataFrame(
        {
            "scenario": int(scenario_id),
            "load_scenario_idx": float(load_scenario_idx),
            "bus": bus_ids,
            "Pd": bus[:, PD],
            "Qd": bus[:, QD],
            "Pg": [pg_by_bus[int(bus_id)] for bus_id in bus_ids],
            "Qg": [qg_by_bus[int(bus_id)] for bus_id in bus_ids],
            "Vm": bus[:, VM],
            "Va": bus[:, VA],
            "PQ": pq,
            "PV": pv,
            "REF": ref,
            "vn_kv": bus[:, BASE_KV],
            "GS": bus[:, GS],
            "BS": bus[:, BS],
            "min_vm_pu": bus[:, VMIN],
            "max_vm_pu": bus[:, VMAX],
        }
    )


def build_branch_dataframe(
    scenario_id: int,
    load_scenario_idx: float,
    result_ppc: dict,
) -> pd.DataFrame:
    branch = result_ppc["branch"]

    pf = branch[:, PF]
    qf = branch[:, QF]
    pt = branch[:, PT]
    qt = branch[:, QT]

    s_from = np.sqrt(pf * pf + qf * qf)
    s_to = np.sqrt(pt * pt + qt * qt)
    s_max = np.maximum(s_from, s_to)

    rate_a = branch[:, RATE_A]
    status = branch[:, BR_STATUS]

    loading = np.divide(
        s_max,
        rate_a,
        out=np.zeros_like(s_max, dtype=float),
        where=rate_a > 0.0,
    ) * 100.0

    loading[status <= 0.0] = 0.0
    loading = np.nan_to_num(loading, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(
        {
            "scenario": int(scenario_id),
            "load_scenario_idx": float(load_scenario_idx),
            "idx": np.arange(branch.shape[0], dtype=int),
            "from_bus": branch[:, F_BUS].astype(int),
            "to_bus": branch[:, T_BUS].astype(int),
            "pf": pf,
            "qf": qf,
            "pt": pt,
            "qt": qt,
            "r": branch[:, BR_R],
            "x": branch[:, BR_X],
            "b": branch[:, BR_B],
            "tap": branch[:, TAP],
            "shift": branch[:, SHIFT],
            "rate_a": rate_a,
            "br_status": status,
            "s_from_mva": s_from,
            "s_to_mva": s_to,
            "s_max_mva": s_max,
            "loading_percent": loading,
            "ang_min": branch[:, ANGMIN],
            "ang_max": branch[:, ANGMAX],
        }
    )


def build_gen_dataframe(
    scenario_id: int,
    load_scenario_idx: float,
    result_ppc: dict,
) -> pd.DataFrame:
    gen = result_ppc["gen"]

    return pd.DataFrame(
        {
            "scenario": int(scenario_id),
            "load_scenario_idx": float(load_scenario_idx),
            "idx": np.arange(gen.shape[0], dtype=int),
            "bus": gen[:, GEN_BUS].astype(int),
            "p_mw": gen[:, PG],
            "q_mvar": gen[:, QG],
            "max_q_mvar": gen[:, QMAX],
            "min_q_mvar": gen[:, QMIN],
            "max_p_mw": gen[:, PMAX],
            "min_p_mw": gen[:, PMIN],
            "in_service": gen[:, GEN_STATUS],
        }
    )


def make_base_case(vmin: float, vmax: float) -> dict:
    net = pn.case118()

    pp.runpp(
        net,
        calculate_voltage_angles=True,
        init="flat",
        numba=False,
    )

    ppc = deepcopy(net["_ppc"])

    ppc["bus"][:, VMIN] = float(vmin)
    ppc["bus"][:, VMAX] = float(vmax)

    return ppc


def choose_outages(
    rng: np.random.Generator,
    active_branch_positions: np.ndarray,
    p_two_outages: float,
) -> list[int]:
    outage_count = 2 if rng.random() < float(p_two_outages) else 1
    outage_count = min(outage_count, len(active_branch_positions))

    selected = rng.choice(
        active_branch_positions,
        size=outage_count,
        replace=False,
    )

    return [int(x) for x in selected]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GridFM-compatible raw parquet files from pandapower case118."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output raw directory containing bus_data.parquet, "
            "branch_data.parquet, gen_data.parquet. "
            "Can also be provided in YAML config as output_dir."
        ),
    )
    parser.add_argument("--num-scenarios", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=10000)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pf-alg", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--max-iter", type=int, default=30)

    parser.add_argument("--vmin", type=float, default=0.94)
    parser.add_argument("--vmax", type=float, default=1.06)

    parser.add_argument("--load-scale-min", type=float, default=0.95)
    parser.add_argument("--load-scale-max", type=float, default=1.35)
    parser.add_argument("--local-load-noise", type=float, default=0.05)

    parser.add_argument(
        "--target-base-utilization",
        type=float,
        default=0.70,
        help=(
            "Used only when MATPOWER/PYPOWER branch rate is zero. "
            "rate_a is derived as base_flow / target_base_utilization."
        ),
    )

    parser.add_argument("--min-rate-mva", type=float, default=40.0)

    parser.add_argument(
        "--p-two-outages",
        type=float,
        default=0.15,
        help="Probability of generating a two-branch outage instead of a single outage.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file with generation parameters.",
    )

    parser.add_argument(
        "--min-save-loading",
        type=float,
        default=105.0,
        help="Save only scenarios whose solved emergency state has max loading at least this value.",
    )

    args = parser.parse_args()

    config = load_yaml_config(args.config)

    if args.config is not None:
        args.output_dir = get_nested(config, ["output_dir"], args.output_dir)

        args.num_scenarios = int(
            get_nested(config, ["num_scenarios"], args.num_scenarios)
        )
        args.max_attempts = int(
            get_nested(config, ["max_attempts"], args.max_attempts)
        )
        args.seed = int(
            get_nested(config, ["seed"], args.seed)
        )

        args.pf_alg = int(
            get_nested(config, ["pf_alg"], args.pf_alg)
        )
        args.max_iter = int(
            get_nested(config, ["max_iter"], args.max_iter)
        )

        args.vmin = float(
            get_nested(config, ["voltage", "vmin"], args.vmin)
        )
        args.vmax = float(
            get_nested(config, ["voltage", "vmax"], args.vmax)
        )

        args.load_scale_min = float(
            get_nested(
                config,
                ["load_randomization", "load_scale_min"],
                args.load_scale_min,
            )
        )
        args.load_scale_max = float(
            get_nested(
                config,
                ["load_randomization", "load_scale_max"],
                args.load_scale_max,
            )
        )
        args.local_load_noise = float(
            get_nested(
                config,
                ["load_randomization", "local_load_noise"],
                args.local_load_noise,
            )
        )

        args.target_base_utilization = float(
            get_nested(
                config,
                ["branch_limits", "target_base_utilization"],
                args.target_base_utilization,
            )
        )
        args.min_rate_mva = float(
            get_nested(
                config,
                ["branch_limits", "min_rate_mva"],
                args.min_rate_mva,
            )
        )

        args.p_two_outages = float(
            get_nested(
                config,
                ["outages", "p_two_outages"],
                args.p_two_outages,
            )
        )

        args.min_save_loading = float(
            get_nested(
                config,
                ["filter", "min_save_loading"],
                args.min_save_loading,
            )
        )

        if args.output_dir is None:
            raise ValueError(
                "Output directory is not specified. "
                "Provide --output-dir or set output_dir in YAML config."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("=" * 100)
    print("Generating synthetic case118 raw dataset")
    print("=" * 100)
    print(f"Output dir:              {output_dir}")
    print(f"Requested scenarios:      {args.num_scenarios}")
    print(f"Max attempts:             {args.max_attempts}")
    print(f"Seed:                     {args.seed}")
    print(f"PF algorithm:             {args.pf_alg}")
    print(f"Load scale range:         {args.load_scale_min} - {args.load_scale_max}")
    print(f"Local load noise:         {args.local_load_noise}")
    print(f"Min save loading:         {args.min_save_loading}%")
    print(f"Two-outage probability:   {args.p_two_outages}")
    print(f"Voltage limits:           {args.vmin} - {args.vmax}")

    base_ppc = make_base_case(
        vmin=args.vmin,
        vmax=args.vmax,
    )

    base_result_ppc, base_success = run_power_flow(
        ppc=clone_ppc(base_ppc),
        pf_alg=args.pf_alg,
        max_iter=args.max_iter,
        fallback_to_newton=True,
    )

    if not base_success or base_result_ppc is None:
        raise RuntimeError("Base case power flow did not converge.")

    static_limits = build_static_branch_limits(
        base_result_ppc=base_result_ppc,
        target_base_utilization=args.target_base_utilization,
        min_rate_mva=args.min_rate_mva,
    )

    active_branch_positions = np.flatnonzero(
        base_ppc["branch"][:, BR_STATUS] > 0.0
    )

    bus_frames: list[pd.DataFrame] = []
    branch_frames: list[pd.DataFrame] = []
    gen_frames: list[pd.DataFrame] = []

    accepted = 0
    attempts = 0
    failed_pf = 0
    weak_cases = 0

    while accepted < int(args.num_scenarios) and attempts < int(args.max_attempts):
        attempts += 1

        ppc = clone_ppc(base_ppc)

        apply_branch_limits(ppc, static_limits)

        apply_load_randomization(
            ppc=ppc,
            rng=rng,
            load_scale_min=args.load_scale_min,
            load_scale_max=args.load_scale_max,
            local_load_noise=args.local_load_noise,
        )

        outage_positions = choose_outages(
            rng=rng,
            active_branch_positions=active_branch_positions,
            p_two_outages=args.p_two_outages,
        )

        apply_outages(ppc, outage_positions)

        result_ppc, success = run_power_flow(
            ppc=ppc,
            pf_alg=args.pf_alg,
            max_iter=args.max_iter,
            fallback_to_newton=True,
        )

        if not success or result_ppc is None:
            failed_pf += 1
            continue

        # Keep the static branch limits in the solved result.
        result_ppc["branch"][:, RATE_A] = static_limits
        result_ppc["branch"][:, RATE_B] = static_limits
        result_ppc["branch"][:, RATE_C] = static_limits

        branch_df = build_branch_dataframe(
            scenario_id=accepted,
            load_scenario_idx=float(attempts),
            result_ppc=result_ppc,
        )

        active = branch_df["br_status"] > 0.0
        max_loading = float(branch_df.loc[active, "loading_percent"].max())

        if max_loading < float(args.min_save_loading):
            weak_cases += 1
            continue

        bus_df = build_bus_dataframe(
            scenario_id=accepted,
            load_scenario_idx=float(attempts),
            result_ppc=result_ppc,
        )

        gen_df = build_gen_dataframe(
            scenario_id=accepted,
            load_scenario_idx=float(attempts),
            result_ppc=result_ppc,
        )

        bus_frames.append(bus_df)
        branch_frames.append(branch_df)
        gen_frames.append(gen_df)

        print(
            f"Accepted scenario {accepted:05d} | "
            f"attempt={attempts:05d} | "
            f"max_loading={max_loading:8.3f}% | "
            f"outages={outage_positions}"
        )

        accepted += 1

    if accepted == 0:
        raise RuntimeError("No scenarios were generated. Try lowering --min-save-loading.")

    bus_data = pd.concat(bus_frames, ignore_index=True)
    branch_data = pd.concat(branch_frames, ignore_index=True)
    gen_data = pd.concat(gen_frames, ignore_index=True)

    bus_path = output_dir / "bus_data.parquet"
    branch_path = output_dir / "branch_data.parquet"
    gen_path = output_dir / "gen_data.parquet"

    bus_data.to_parquet(bus_path, index=False)
    branch_data.to_parquet(branch_path, index=False)
    gen_data.to_parquet(gen_path, index=False)

    print("\n" + "=" * 100)
    print("Generation summary")
    print("=" * 100)
    print(f"Accepted scenarios:       {accepted}")
    print(f"Attempts:                 {attempts}")
    print(f"Failed power flows:        {failed_pf}")
    print(f"Weak cases skipped:        {weak_cases}")
    print(f"Saved:                     {bus_path}")
    print(f"Saved:                     {branch_path}")
    print(f"Saved:                     {gen_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()