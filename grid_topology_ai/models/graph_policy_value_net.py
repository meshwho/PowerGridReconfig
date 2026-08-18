from __future__ import annotations


class GraphPolicyValueNet:
    """Removed fixed-cardinality Graph V1 model."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "Graph V1 has been removed. Use GraphPolicyValueNetV2."
        )
