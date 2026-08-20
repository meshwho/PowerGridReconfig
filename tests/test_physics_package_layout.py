import importlib
import importlib.util


CANONICAL_MODULES = (
    "grid_topology_ai.physics.constraints",
    "grid_topology_ai.physics.objective",
    "grid_topology_ai.physics.utility",
    "grid_topology_ai.physics.redispatch",
    "grid_topology_ai.physics.lodf",
)

LEGACY_MODULES = (
    "grid_topology_ai.physical_constraints",
    "grid_topology_ai.physical_objective",
    "grid_topology_ai.grid_utility",
    "grid_topology_ai.redispatch",
    "grid_topology_ai.lodf",
)


def test_canonical_physics_modules_import() -> None:
    for module_name in CANONICAL_MODULES:
        assert importlib.import_module(module_name) is not None


def test_legacy_physics_modules_are_absent() -> None:
    for module_name in LEGACY_MODULES:
        assert importlib.util.find_spec(module_name) is None
