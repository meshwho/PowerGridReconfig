from pathlib import Path

import yaml

from scripts.data.build_balanced_gridfm_dataset import make_gridfm_config_text


def test_gridfm_config_uses_branch_topology_element(tmp_path: Path) -> None:
    config_text = make_gridfm_config_text(
        network_name="case118_ieee",
        network_source="pglib",
        data_dir=tmp_path,
        scenarios=10,
        seed=42,
        num_processes=1,
        sigma=0.2,
        global_range=0.5,
        max_scaling_factor=2.0,
        step_size=0.1,
        start_scaling_factor=1.0,
        topology_variants=3,
        topology_k=1,
        generation_perturbation_type="none",
        generation_perturbation_sigma=0.0,
        admittance_perturbation_type="none",
        admittance_perturbation_sigma=0.0,
    )

    config = yaml.safe_load(config_text)

    assert config["topology_perturbation"]["elements"] == ["branch"]
