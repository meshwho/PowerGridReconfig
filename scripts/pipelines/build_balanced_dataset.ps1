param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"

# ======================================================================================
# Runtime/cache locations
# ======================================================================================
# Keep temporary files and Julia packages on drive D: to avoid filling drive C:.

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

Write-Host "Runtime/cache settings:"
Write-Host "  TEMP:             $env:TEMP"
Write-Host "  TMP:              $env:TMP"
Write-Host "  JULIA_DEPOT_PATH: $env:JULIA_DEPOT_PATH"
Write-Host "  Julia bin:         $JuliaBin"
Write-Host ""

# ======================================================================================
# Balanced GridFM dataset settings
# ======================================================================================

$DatasetName = "case118_balanced_v1"

# Final target dataset size.
$TargetTotal = 5000

# Class distribution.
$SimpleFraction = 0.2
$MediumFraction = 0.50
$HardFraction = 0.3

# Network settings.
$NetworkName = "case118_ieee"
$NetworkSource = "pglib"

# Обычно GridFM создает папку с именем network.name.
# Если для другой сети GridFM создает другое имя папки, поменяй только это.
$RawNetworkDirName = $NetworkName

# GridFM generation.
$ChunkSize = 2000
$MaxChunks = 60
$SeedStart = 20000
$NumProcesses = 6

# Output.
$OutputRoot = "data/datasets/$DatasetName"

# IMPORTANT:
# Put here the exact GridFM command that you normally use to run a YAML config.
# The token {config} will be replaced by the generated YAML path.
#
# Examples, depending on your local GridFM installation:
# $GridFMCommandTemplate = 'python -m gridfm_datakit --config "{config}"'
# $GridFMCommandTemplate = 'gridfm-datakit generate --config "{config}"'
# $GridFMCommandTemplate = 'gridfm generate "{config}"'
#
# If your previous command was different, change only this line.
$GridFMCommandTemplate = 'python -m gridfm_datakit.cli generate "{config}"'
# GridFM physical / scenario settings.
$Sigma = 0.2
$GlobalRange = 0.5
$MaxScalingFactor = 2.0
$StepSize = 0.1
$StartScalingFactor = 1.0
$TopologyVariants = 10
$TopologyK = 1

# Selection thresholds.
$MinLoading = 105
$MaxLoading = 260

# Class thresholds.
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

# Train/validation split.
$TrainFraction = 0.80
$SplitSeed = 42

# ======================================================================================
# Run
# ======================================================================================

Write-Host "===================================================================================================="
Write-Host "Building balanced GridFM dataset"
Write-Host "===================================================================================================="
Write-Host "Dataset:      $DatasetName"
Write-Host "Target total: $TargetTotal"
Write-Host "Output root:  $OutputRoot"
Write-Host "Resume:       $Resume"
Write-Host ""

$ResumeFlag = ""
if ($Resume) {
    $ResumeFlag = "--resume"
}

python -m scripts.data.build_balanced_gridfm_dataset `
    --dataset-name $DatasetName `
    --network-name $NetworkName `
    --network-source $NetworkSource `
    --raw-network-dir-name $RawNetworkDirName `
    --output-root $OutputRoot `
    --target-total $TargetTotal `
    --simple-fraction $SimpleFraction `
    --medium-fraction $MediumFraction `
    --hard-fraction $HardFraction `
    --chunk-size $ChunkSize `
    --max-chunks $MaxChunks `
    --seed-start $SeedStart `
    --num-processes $NumProcesses `
    --gridfm-command-template $GridFMCommandTemplate `
    --sigma $Sigma `
    --global-range $GlobalRange `
    --max-scaling-factor $MaxScalingFactor `
    --step-size $StepSize `
    --start-scaling-factor $StartScalingFactor `
    --topology-variants $TopologyVariants `
    --topology-k $TopologyK `
    --min-loading $MinLoading `
    --max-loading $MaxLoading `
    --simple-min-loading $SimpleMinLoading `
    --simple-max-loading $SimpleMaxLoading `
    --simple-max-hard $SimpleMaxHard `
    --simple-max-overloaded $SimpleMaxOverloaded `
    --medium-min-loading $MediumMinLoading `
    --medium-max-loading $MediumMaxLoading `
    --medium-max-hard $MediumMaxHard `
    --medium-max-overloaded $MediumMaxOverloaded `
    --hard-min-loading $HardMinLoading `
    --hard-min-hard $HardMinHard `
    --train-fraction $TrainFraction `
    --split-seed $SplitSeed `
    $ResumeFlag

Write-Host ""
Write-Host "Done."