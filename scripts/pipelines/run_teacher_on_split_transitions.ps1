param(
    [switch]$Force,
    [switch]$SkipSimple,
    [switch]$SkipMedium,
    [switch]$SkipHard
)

$ErrorActionPreference = "Stop"

# ======================================================================================
# Runtime/cache locations
# ======================================================================================

$TempGridFM = "D:\temp_gridfm"
$JuliaDepot = "D:\julia_depot"
$JuliaBin = "C:\Users\timof\AppData\Local\Programs\Julia-1.12.6\bin"

New-Item -ItemType Directory -Force -Path $TempGridFM | Out-Null
New-Item -ItemType Directory -Force -Path $JuliaDepot | Out-Null

$env:TEMP = $TempGridFM
$env:TMP = $TempGridFM
$env:JULIA_DEPOT_PATH = $JuliaDepot

if (Test-Path $JuliaBin) {
    $env:Path = "$JuliaBin;$env:Path"
}
else {
    Write-Warning "Julia bin directory not found: $JuliaBin"
}

# ======================================================================================
# Main paths
# ======================================================================================

$RawDir = "data/datasets/case118_balanced_v1/raw"

# Ты уже разделил transitions вручную - здесь просто указываем готовые файлы.
$TransitionsSimple = "data/datasets/case118_balanced_v1/transitions/transitions_train_simple.csv"
$TransitionsMedium = "data/datasets/case118_balanced_v1/transitions/transitions_train_medium.csv"
$TransitionsHard = "data/datasets/case118_balanced_v1/transitions/transitions_train_hard.csv"

# Output folders.
$OutSimple = "data/self_play/impact_teacher_balanced_v1_simple"
$OutMedium = "data/self_play/impact_teacher_balanced_v1_medium"
$OutHard = "data/self_play/impact_teacher_balanced_v1_hard_strong"

# ======================================================================================
# Memory-safe settings
# ======================================================================================

$MaxWorkerMemoryMb = 1000
$MaxTasksPerChild = 0

# Можно увеличить workers, если RAM держится стабильно.
$SimpleWorkers = 8
$MediumWorkers = 7
$HardWorkers = 6

$SimpleBatchSize = 3
$MediumBatchSize = 2
$HardBatchSize = 2

# ======================================================================================
# Helpers
# ======================================================================================

function Assert-LastCommandOk {
    param([string]$Message)

    if ($LASTEXITCODE -ne 0) {
        throw "$Message failed with exit code $LASTEXITCODE"
    }
}

function Remove-OutputIfForce {
    param([string]$Path)

    if ($Force -and (Test-Path $Path)) {
        Write-Host "Force enabled - removing: $Path"
        Remove-Item -Recurse -Force $Path
    }
}

function Run-TeacherIfNeeded {
    param(
        [string]$Name,
        [string]$Transitions,
        [string]$OutputDir,
        [int]$Depth,
        [int]$BeamWidth,
        [int]$CandidatePool,
        [int]$TopK,
        [int]$MaxSteps,
        [int]$MaxTeacherSteps,
        [int]$NumWorkers,
        [int]$BatchSize,
        [int]$ClearCachesEvery
    )

    $ExamplesPath = "$OutputDir/examples.csv"

    if (!(Test-Path $Transitions)) {
        throw "$Name transitions file not found: $Transitions"
    }

    if ((Test-Path $ExamplesPath) -and -not $Force) {
        Write-Host ""
        Write-Host "===================================================================================================="
        Write-Host "$Name teacher already exists - skipping"
        Write-Host "===================================================================================================="
        Write-Host "Examples: $ExamplesPath"
        return
    }

    Remove-OutputIfForce $OutputDir

    Write-Host ""
    Write-Host "===================================================================================================="
    Write-Host "Running $Name teacher"
    Write-Host "===================================================================================================="
    Write-Host "Raw dir:     $RawDir"
    Write-Host "Transitions: $Transitions"
    Write-Host "Output dir:  $OutputDir"
    Write-Host "Depth:       $Depth"
    Write-Host "Beam width:  $BeamWidth"
    Write-Host "Pool:        $CandidatePool"
    Write-Host "Top-K:       $TopK"
    Write-Host "Workers:     $NumWorkers"
    Write-Host "Batch size:  $BatchSize"

    python -m scripts.self_play.generate_impact_teacher_parallel_fast `
      $RawDir `
      --transitions $Transitions `
      --output-dir $OutputDir `
      --depth $Depth `
      --beam-width $BeamWidth `
      --candidate-pool $CandidatePool `
      --top-k $TopK `
      --max-steps $MaxSteps `
      --max-teacher-steps $MaxTeacherSteps `
      --pf-alg 3 `
      --pf-max-iter 30 `
      --num-workers $NumWorkers `
      --batch-size $BatchSize `
      --clear-caches-every $ClearCachesEvery `
      --max-worker-memory-mb $MaxWorkerMemoryMb `
      --max-tasks-per-child $MaxTasksPerChild `
      --add-handoff-example `
      --quiet-success

    Assert-LastCommandOk "$Name teacher"
}

# ======================================================================================
# Start
# ======================================================================================

Write-Host "===================================================================================================="
Write-Host "Running teacher generation on already split transitions"
Write-Host "===================================================================================================="
Write-Host "TEMP:             $env:TEMP"
Write-Host "TMP:              $env:TMP"
Write-Host "JULIA_DEPOT_PATH: $env:JULIA_DEPOT_PATH"
Write-Host "Raw dir:          $RawDir"
Write-Host "Force:            $Force"
Write-Host ""

if (!(Test-Path $RawDir)) {
    throw "Raw directory not found: $RawDir"
}

# ======================================================================================
# Simple teacher
# ======================================================================================

if (-not $SkipSimple) {
    Run-TeacherIfNeeded `
        -Name "simple" `
        -Transitions $TransitionsSimple `
        -OutputDir $OutSimple `
        -Depth 4 `
        -BeamWidth 10 `
        -CandidatePool 60 `
        -TopK 30 `
        -MaxSteps 5 `
        -MaxTeacherSteps 5 `
        -NumWorkers $SimpleWorkers `
        -BatchSize $SimpleBatchSize `
        -ClearCachesEvery $SimpleBatchSize
}

# ======================================================================================
# Medium teacher
# ======================================================================================

if (-not $SkipMedium) {
    Run-TeacherIfNeeded `
        -Name "medium" `
        -Transitions $TransitionsMedium `
        -OutputDir $OutMedium `
        -Depth 5 `
        -BeamWidth 20 `
        -CandidatePool 160 `
        -TopK 70 `
        -MaxSteps 5 `
        -MaxTeacherSteps 5 `
        -NumWorkers $MediumWorkers `
        -BatchSize $MediumBatchSize `
        -ClearCachesEvery $MediumBatchSize
}

# ======================================================================================
# Hard teacher - strong settings
# ======================================================================================

if (-not $SkipHard) {
    Run-TeacherIfNeeded `
        -Name "hard_strong" `
        -Transitions $TransitionsHard `
        -OutputDir $OutHard `
        -Depth 6 `
        -BeamWidth 30 `
        -CandidatePool 220 `
        -TopK 100 `
        -MaxSteps 6 `
        -MaxTeacherSteps 6 `
        -NumWorkers $HardWorkers `
        -BatchSize $HardBatchSize `
        -ClearCachesEvery $HardBatchSize
}

Write-Host ""
Write-Host "===================================================================================================="
Write-Host "All requested teacher runs finished"
Write-Host "===================================================================================================="
Write-Host "Simple examples: $OutSimple/examples.csv"
Write-Host "Medium examples: $OutMedium/examples.csv"
Write-Host "Hard examples:   $OutHard/examples.csv"
Write-Host ""