from pathlib import Path


def test_generation_policy_target_static_boundaries() -> None:
    text = Path("grid_topology_ai/self_play/generation.py").read_text(encoding="utf-8")

    forbidden = (
        "make_one_hot_policy(",
        "policy_target = {int(selected_action_id): 1.0}",
        "policy_target = {selected_action_id: 1.0}",
    )
    assert all(token not in text for token in forbidden)
    assert "policy_target_source" in text
    assert "mcts_visit_distribution" in text
    assert "gate_overrode_mcts_selection" in text
