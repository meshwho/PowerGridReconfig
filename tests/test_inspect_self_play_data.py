from __future__ import annotations

import pandas as pd

from scripts.inspect_self_play_data import format_sample_row, load_json_dict


def test_strict_row_without_legacy_reward_columns_formats() -> None:
    row = pd.Series(
        {
            "scenario_id": 1,
            "step": 0,
            "state_id": "strict",
            "solved": True,
            "termination_reason": "solved",
        }
    )
    text = format_sample_row(row)
    assert "selected_action=n/a" in text
    assert "step_reward=n/a" in text
    assert load_json_dict(None) == {}
