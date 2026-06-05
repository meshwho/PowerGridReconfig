$ErrorActionPreference = "Stop"

cd "D:\проекты\PowerGridReconfig"

$env:TEMP="D:\temp_gridfm"
$env:TMP="D:\temp_gridfm"
$env:JULIA_DEPOT_PATH="D:\julia_depot"
$env:Path = "C:\Users\timof\AppData\Local\Programs\Julia-1.12.6\bin;$env:Path"

New-Item -ItemType Directory -Force -Path "D:\temp_gridfm" | Out-Null
New-Item -ItemType Directory -Force -Path "data\self_play\impact_teacher_balanced_v1_hard_lodf_k100" | Out-Null

& "D:\проекты\PowerGridReconfig\.venv311\Scripts\python.exe" -u -m scripts.self_play.generate_impact_teacher_parallel_fast `
  data/datasets/case118_balanced_v1/raw `
  --transitions data/datasets/case118_balanced_v1/transitions/transitions_train_hard.csv `
  --output-dir data/self_play/impact_teacher_balanced_v1_hard_lodf_k100 `
  --depth 6 `
  --beam-width 30 `
  --candidate-pool 220 `
  --top-k 100 `
  --use-lodf-screening `
  --lodf-screen-top-k 100 `
  --lodf-min-candidate-count 8 `
  --max-steps 6 `
  --max-teacher-steps 6 `
  --pf-alg 3 `
  --pf-max-iter 30 `
  --num-workers auto `
  --auto-worker-cpu-mode logical `
  --auto-worker-cpu-fraction 0.85 `
  --auto-worker-max 6 `
  --batch-size 2 `
  --clear-caches-every 2 `
  --max-worker-memory-mb 1200 `
  --min-free-system-memory-mb 1536 `
  --auto-worker-memory-mb 950 `
  --auto-worker-memory-reserve-mb 2048 `
  --max-tasks-per-child 0 `
  --add-handoff-example `
  --quiet-success 2>&1 | Tee-Object -FilePath "data\self_play\impact_teacher_balanced_v1_hard_lodf_k100\run.log"