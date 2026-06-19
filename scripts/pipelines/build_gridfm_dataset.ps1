param(
    [ValidateSet("smoke", "eval", "bootstrap")]
    [string]$Profile = "smoke",

    [switch]$Resume,

    [string]$GridFMCommandTemplate = 'python -m gridfm_datakit.cli generate "{config}"',

    [string]$JuliaBin = "",

    [string]$TempGridFM = "D:\temp_gridfm",

    [string]$JuliaDepot = "D:\julia_depot",

    [int]$NumProcesses = 4
)

$ErrorActionPreference = "Stop"

# ======================================================================================
# Runtime/cache locations
# ======================================================================================

New-Item -ItemType Directory -Force -Path $TempGridFM | Out-Null
New-Item -ItemType Directory -Force -Path $JuliaDepot | Out-Null

$env:TEMP = $TempGridFM
$env:TMP = $TempGridFM
$env:JULIA_DEPOT_PATH = $JuliaDepot

if ($JuliaBin -ne "") {
    if (Test-Path $JuliaBin) {
        $env:Path = "$JuliaBin;$env:Path"
    }
    else {
        Write-Warning "Julia bin directory not found: $JuliaBin"
    }
}

# ======================================================================================
# Project data layout
# ======================================================================================

$DataRoot = "data"

$GeneratedRoot = "$DataRoot/gridfm_generated"
$TransitionsRoot = "$DataRoot/gridfm_transitions"
$ScratchRoot = "$DataRoot/_scratch"
$DatasetsRoot = "$DataRoot/datasets"
$SelfPlayRoot = "$DataRoot/self_play"
$TrainingRoot = "$DataRoot/training"

New-Item -ItemType Directory -Force -Path $GeneratedRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TransitionsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ScratchRoot | Out-Null
New-Item -ItemType Directory -Force -Path $DatasetsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $SelfPlayRoot | Out-Null
New-Item -ItemType Directory -Force -Path $TrainingRoot | Out-Null

# ======================================================================================
# Shared network settings
# ======================================================================================

$NetworkName = "case118_ieee"
$NetworkSource = "pglib"
$RawNetworkDirName = $NetworkName

# ======================================================================================
# Shared physical / scenario settings
# ======================================================================================

$Sigma = 0.2
$GlobalRange = 0.5
$MaxScalingFactor = 2.0
$StepSize = 0.1
$StartScalingFactor = 1.0
$TopologyVariants = 10
$TopologyK = 1
$GenerationPerturbationType = "none"
$GenerationPerturbationSigma = 0.0

$AdmittancePerturbationType = "none"
$AdmittancePerturbationSigma = 0.0

# ======================================================================================
# Shared selection thresholds
# ======================================================================================

$MinLoading = 105
$MaxLoading = 260

$SimpleMinLoading = 105
$SimpleMaxLoading = 120
$SimpleMaxHard = 0
$SimpleMaxOverloaded = 2

$MediumMinLoading = 120
$MediumMaxLoading = 150
$MediumMaxHard = 1
$MediumMaxOverloaded = 5

$HardMinLoading = 150
$HardMinHard = 2

$TrainFraction = 0.80
$SplitSeed = 42

# ======================================================================================
# Profile-specific settings
# ======================================================================================

$AllowPartial = $false

if ($Profile -eq "smoke") {
    # Temporary small run only to check that GridFM generation works.
    $DatasetName = "case118_smoke_v1"

    # Smoke data goes into _scratch, not into permanent dataset folders.
    $OutputRoot = "$ScratchRoot/$DatasetName"
    $TransitionExportRoot = "$ScratchRoot/$DatasetName/transitions_export"

    $TargetTotal = 50
    $ChunkSize = 200
    $MaxChunks = 3
    $SeedStart = 10000

    $GenerationPerturbationType = "cost_perturbation"
    $GenerationPerturbationSigma = 0.10

    $AdmittancePerturbationType = "random_perturbation"
    $AdmittancePerturbationSigma = 0.02

    $SimpleFraction = 0.25
    $MediumFraction = 0.50
    $HardFraction = 0.25

    $AllowPartial = $true
}
elseif ($Profile -eq "eval") {
    # Fixed evaluation set.
    # Important: do not use this for training or self-play generation.
    $DatasetName = "case118_eval_v1"

    # Raw GridFM data stays here:
    $OutputRoot = "$GeneratedRoot/$DatasetName"

    # Stable transitions and manifest copy go here:
    $TransitionExportRoot = "$TransitionsRoot/$DatasetName"

    $TargetTotal = 500
    $ChunkSize = 1000
    $MaxChunks = 20
    $SeedStart = 90000

    $GenerationPerturbationType = "cost_perturbation"
    $GenerationPerturbationSigma = 0.10

    $AdmittancePerturbationType = "random_perturbation"
    $AdmittancePerturbationSigma = 0.02
    
    $SimpleFraction = 0.25
    $MediumFraction = 0.50
    $HardFraction = 0.25
}
elseif ($Profile -eq "bootstrap") {
    # Main supervised bootstrap dataset.
    $DatasetName = "case118_bootstrap_v1"

    # Raw GridFM data stays here:
    $OutputRoot = "$GeneratedRoot/$DatasetName"

    # Stable transitions and manifest copy go here:
    $TransitionExportRoot = "$TransitionsRoot/$DatasetName"

    $TargetTotal = 12000
    $ChunkSize = 2000
    $MaxChunks = 80
    $SeedStart = 20000

    $GenerationPerturbationType = "cost_perturbation"
    $GenerationPerturbationSigma = 0.20

    $AdmittancePerturbationType = "random_perturbation"
    $AdmittancePerturbationSigma = 0.05
    
    $SimpleFraction = 0.20
    $MediumFraction = 0.50
    $HardFraction = 0.30
}
else {
    throw "Unknown profile: $Profile"
}

# ======================================================================================
# CLI flags
# ======================================================================================

$CliArgs = @(
    "-m", "scripts.data.build_balanced_gridfm_dataset",
    "--dataset-name", $DatasetName,
    "--network-name", $NetworkName,
    "--network-source", $NetworkSource,
    "--raw-network-dir-name", $RawNetworkDirName,
    "--output-root", $OutputRoot,
    "--target-total", "$TargetTotal",
    "--simple-fraction", "$SimpleFraction",
    "--medium-fraction", "$MediumFraction",
    "--hard-fraction", "$HardFraction",
    "--chunk-size", "$ChunkSize",
    "--max-chunks", "$MaxChunks",
    "--seed-start", "$SeedStart",
    "--num-processes", "$NumProcesses",
    "--gridfm-command-template", $GridFMCommandTemplate,
    "--sigma", "$Sigma",
    "--global-range", "$GlobalRange",
    "--max-scaling-factor", "$MaxScalingFactor",
    "--step-size", "$StepSize",
    "--start-scaling-factor", "$StartScalingFactor",
    "--topology-variants", "$TopologyVariants",
    "--topology-k", "$TopologyK",
    "--generation-perturbation-type", $GenerationPerturbationType,
    "--generation-perturbation-sigma", "$GenerationPerturbationSigma",
    "--admittance-perturbation-type", $AdmittancePerturbationType,
    "--admittance-perturbation-sigma", "$AdmittancePerturbationSigma",
    "--min-loading", "$MinLoading",
    "--max-loading", "$MaxLoading",
    "--simple-min-loading", "$SimpleMinLoading",
    "--simple-max-loading", "$SimpleMaxLoading",
    "--simple-max-hard", "$SimpleMaxHard",
    "--simple-max-overloaded", "$SimpleMaxOverloaded",
    "--medium-min-loading", "$MediumMinLoading",
    "--medium-max-loading", "$MediumMaxLoading",
    "--medium-max-hard", "$MediumMaxHard",
    "--medium-max-overloaded", "$MediumMaxOverloaded",
    "--hard-min-loading", "$HardMinLoading",
    "--hard-min-hard", "$HardMinHard",
    "--train-fraction", "$TrainFraction",
    "--split-seed", "$SplitSeed"
)

if ($AllowPartial) {
    $CliArgs += "--allow-partial"
}

if ($Resume) {
    $CliArgs += "--resume"
}

# ======================================================================================
# Summary
# ======================================================================================

Write-Host "===================================================================================================="
Write-Host "Building GridFM dataset"
Write-Host "===================================================================================================="
Write-Host "Profile:                $Profile"
Write-Host "Dataset:                $DatasetName"
Write-Host "Target total:           $TargetTotal"
Write-Host "Chunk size:             $ChunkSize"
Write-Host "Max chunks:             $MaxChunks"
Write-Host "Seed start:             $SeedStart"
Write-Host "Network:                $NetworkName"
Write-Host "Output root:            $OutputRoot"
Write-Host "Transition export root: $TransitionExportRoot"
Write-Host "Resume:                 $Resume"
Write-Host "Allow partial:          $AllowPartial"
Write-Host ""
Write-Host "Data layout:"
Write-Host "  Generated raw:         $GeneratedRoot"
Write-Host "  Transitions:           $TransitionsRoot"
Write-Host "  Scratch:               $ScratchRoot"
Write-Host "  Future datasets:       $DatasetsRoot"
Write-Host "  Future self-play:      $SelfPlayRoot"
Write-Host "  Future training:       $TrainingRoot"
Write-Host ""
Write-Host "Runtime/cache settings:"
Write-Host "  TEMP:                  $env:TEMP"
Write-Host "  TMP:                   $env:TMP"
Write-Host "  JULIA_DEPOT_PATH:      $env:JULIA_DEPOT_PATH"
Write-Host "  Julia bin:             $JuliaBin"
Write-Host ""
Write-Host "GridFM command template:"
Write-Host "  $GridFMCommandTemplate"
Write-Host ""

# ======================================================================================
# Run generation and balanced transition builder
# ======================================================================================

& python @CliArgs

# ======================================================================================
# Export stable transition files into data/gridfm_transitions
# ======================================================================================

Write-Host ""
Write-Host "===================================================================================================="
Write-Host "Exporting transition files"
Write-Host "===================================================================================================="

New-Item -ItemType Directory -Force -Path $TransitionExportRoot | Out-Null

$SourceTransitionsDir = Join-Path $OutputRoot "transitions"
$SourceManifestDir = Join-Path $OutputRoot "manifest"
$TargetManifestDir = Join-Path $TransitionExportRoot "manifest"

if (-not (Test-Path $SourceTransitionsDir)) {
    throw "Transitions directory not found: $SourceTransitionsDir"
}

Copy-Item -Path (Join-Path $SourceTransitionsDir "*.csv") -Destination $TransitionExportRoot -Force

if (Test-Path $SourceManifestDir) {
    New-Item -ItemType Directory -Force -Path $TargetManifestDir | Out-Null
    Copy-Item -Path (Join-Path $SourceManifestDir "*.csv") -Destination $TargetManifestDir -Force -ErrorAction SilentlyContinue
    Copy-Item -Path (Join-Path $SourceManifestDir "*.json") -Destination $TargetManifestDir -Force -ErrorAction SilentlyContinue
}

$RawDir = Join-Path $OutputRoot "raw"

$DatasetInfo = [ordered]@{
    profile = $Profile
    dataset_name = $DatasetName
    output_root = $OutputRoot
    raw_dir = $RawDir
    transition_export_root = $TransitionExportRoot
    transitions_balanced = (Join-Path $TransitionExportRoot "transitions_balanced.csv")
    transitions_train = (Join-Path $TransitionExportRoot "transitions_train.csv")
    transitions_val = (Join-Path $TransitionExportRoot "transitions_val.csv")
    manifest_dir = $TargetManifestDir
    target_total = $TargetTotal
    chunk_size = $ChunkSize
    max_chunks = $MaxChunks
    seed_start = $SeedStart
    network_name = $NetworkName
    topology_k = $TopologyK
}

$DatasetInfoPath = Join-Path $TransitionExportRoot "dataset_info.json"

$DatasetInfo |
    ConvertTo-Json -Depth 10 |
    Set-Content -Path $DatasetInfoPath -Encoding UTF8

Write-Host "Exported transitions to:"
Write-Host "  $TransitionExportRoot"
Write-Host ""
Write-Host "Dataset info:"
Write-Host "  $DatasetInfoPath"

# ======================================================================================
# Final summary
# ======================================================================================

Write-Host ""
Write-Host "===================================================================================================="
Write-Host "Dataset build finished"
Write-Host "===================================================================================================="
Write-Host "Raw GridFM directory:"
Write-Host "  $RawDir"
Write-Host ""
Write-Host "Transitions:"
Write-Host "  $(Join-Path $TransitionExportRoot 'transitions_balanced.csv')"
Write-Host "  $(Join-Path $TransitionExportRoot 'transitions_train.csv')"
Write-Host "  $(Join-Path $TransitionExportRoot 'transitions_val.csv')"
Write-Host ""
Write-Host "Manifest:"
Write-Host "  $TargetManifestDir"
Write-Host ""