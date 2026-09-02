import importlib


CANONICAL_MODULES = (
    "grid_topology_ai.physics.constraints",
    "grid_topology_ai.physics.objective",
    "grid_topology_ai.physics.utility",
    "grid_topology_ai.physics.redispatch",
    "grid_topology_ai.physics.lodf",
)


def test_canonical_physics_modules_import() -> None:
    for module_name in CANONICAL_MODULES:
        assert importlib.import_module(module_name) is not None
