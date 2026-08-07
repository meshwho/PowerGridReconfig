from __future__ import annotations

import os
import tempfile
from importlib.metadata import version
from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

if os.name == "nt":
    from scripts.data.gridfm_compat_cli import _windows_safe_correct_network
    from gridfm_datakit import network as _network

    _work_root = Path(os.environ.get("TEMP", tempfile.gettempdir()))

    def _correct_network(network_path: str, force: bool = False) -> str:
        return _windows_safe_correct_network(
            network_path,
            network_module=_network,
            work_root=_work_root,
            force=force,
        )

    _network.correct_network = _correct_network

from gridfm_datakit.generate import (
    generate_power_flow_data,
    generate_power_flow_data_distributed,
)
from gridfm_datakit import powsybl

__version__ = version("gridfm-datakit")

__all__ = [
    "generate_power_flow_data",
    "generate_power_flow_data_distributed",
    "powsybl",
]
