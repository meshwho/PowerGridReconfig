from __future__ import annotations

from pathlib import Path

from grid_topology_ai.config import EvaluationConfig
from grid_topology_ai.self_play import stages
from grid_topology_ai.self_play.artifacts import save_json


def test_run_evaluate_resolves_config_pf_alg(tmp_path: Path, monkeypatch) -> None:
    captured = []

    def fake_evaluate(request):
        captured.append(request)
        request.output_csv.parent.mkdir(parents=True, exist_ok=True)
        request.output_csv.write_text("scenario_id,solved\n1,true\n", encoding="utf-8")
        save_json({"solve_rate": 1.0, "pf_alg": 3}, request.output_json)
        return {"solve_rate": 1.0, "pf_alg": 3}

    monkeypatch.setattr(stages, "evaluate_checkpoint", fake_evaluate)
    stages.run_evaluate(
        project_root=tmp_path,
        checkpoint=tmp_path / "candidate.pt",
        eval_csv=tmp_path / "eval.csv",
        eval_raw_dir=tmp_path / "raw",
        output_dir=tmp_path / "eval",
        config=EvaluationConfig(pf_alg=3),
    )

    assert captured[0].pf_alg is None
    assert captured[0].resolved_pf_alg == 3
