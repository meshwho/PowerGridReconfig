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

Set-StrictMode -Version 3.0
$ErrorActionPreference = "Stop"


# ======================================================================================
# Native process helper
# ======================================================================================

function ConvertTo-WindowsCommandLineArgument {
    param(
        [AllowNull()]
        [AllowEmptyString()]
        [string]$Argument
    )

    if ($null -eq $Argument -or $Argument.Length -eq 0) {
        return '""'
    }

    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    # Windows CreateProcess quoting rules:
    # - quote arguments containing spaces or quotes;
    # - double backslashes before embedded quotes;
    # - double trailing backslashes before the closing quote.
    $Escaped = [regex]::Replace(
        $Argument,
        '(\\*)"',
        '$1$1\"'
    )

    $Escaped = [regex]::Replace(
        $Escaped,
        '(\\+)$',
        '$1$1'
    )

    return '"' + $Escaped + '"'
}


function Invoke-NativeProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,

        [Parameter(Mandatory = $true)]
        [string]$StdoutLogPath,

        [Parameter(Mandatory = $true)]
        [string]$StderrLogPath
    )

    $StdoutLogDirectory = Split-Path -Parent $StdoutLogPath
    $StderrLogDirectory = Split-Path -Parent $StderrLogPath

    New-Item -ItemType Directory -Force -Path $StdoutLogDirectory | Out-Null
    New-Item -ItemType Directory -Force -Path $StderrLogDirectory | Out-Null

    Remove-Item -Force -ErrorAction SilentlyContinue $StdoutLogPath
    Remove-Item -Force -ErrorAction SilentlyContinue $StderrLogPath

    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $FilePath
    $StartInfo.WorkingDirectory = $WorkingDirectory
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true

    # PowerShell 7 / modern .NET exposes ArgumentList.
    # Windows PowerShell 5.1 needs one correctly quoted argument string.
    if ($StartInfo.PSObject.Properties.Name -contains "ArgumentList") {
        foreach ($Argument in $Arguments) {
            [void]$StartInfo.ArgumentList.Add([string]$Argument)
        }
    }
    else {
        $QuotedArguments = foreach ($Argument in $Arguments) {
            ConvertTo-WindowsCommandLineArgument -Argument ([string]$Argument)
        }

        $StartInfo.Arguments = $QuotedArguments -join " "
    }

    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $StartInfo

    Write-Host "Executable:"
    Write-Host "  $FilePath"
    Write-Host ""
    Write-Host "Working directory:"
    Write-Host "  $WorkingDirectory"
    Write-Host ""
    Write-Host "Starting process. Output will be printed when the process finishes."
    Write-Host "GridFM chunk logs are also written inside the dataset logs directory."
    Write-Host ""

    [void]$Process.Start()

    # Start both asynchronous reads before waiting. This prevents a deadlock
    # when either stdout or stderr fills its operating-system buffer.
    $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
    $StderrTask = $Process.StandardError.ReadToEndAsync()

    $Process.WaitForExit()

    $StdoutText = $StdoutTask.GetAwaiter().GetResult()
    $StderrText = $StderrTask.GetAwaiter().GetResult()
    $ExitCode = $Process.ExitCode

    [System.IO.File]::WriteAllText(
        $StdoutLogPath,
        [string]$StdoutText,
        [System.Text.UTF8Encoding]::new($false)
    )

    [System.IO.File]::WriteAllText(
        $StderrLogPath,
        [string]$StderrText,
        [System.Text.UTF8Encoding]::new($false)
    )

    if (-not [string]::IsNullOrWhiteSpace($StdoutText)) {
        Write-Host $StdoutText
    }

    if (-not [string]::IsNullOrWhiteSpace($StderrText)) {
        # Print stderr as ordinary text, not as a PowerShell error record.
        Write-Host $StderrText
    }

    Write-Host ""
    Write-Host "Process exit code: $ExitCode"
    Write-Host "stdout log: $StdoutLogPath"
    Write-Host "stderr log: $StderrLogPath"
    Write-Host ""

    if ($ExitCode -ne 0) {
        throw (
            "Dataset builder failed with exit code $ExitCode. " +
            "See logs: $StdoutLogPath and $StderrLogPath"
        )
    }
}


# ======================================================================================
# Project root
# ======================================================================================

$ProjectRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\..")
)

$PythonScriptPath = Join-Path `
    $ProjectRoot `
    "scripts\data\build_balanced_gridfm_dataset.py"

if (-not (Test-Path $PythonScriptPath)) {
    throw "Python dataset builder not found: $PythonScriptPath"
}

$PythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue

if ($null -eq $PythonCommand) {
    $PythonCommand = Get-Command python -ErrorAction Stop
}

$PythonExecutable = $PythonCommand.Source

Push-Location $ProjectRoot

try {
    # ==================================================================================
    # Runtime/cache locations
    # ==================================================================================

    New-Item -ItemType Directory -Force -Path $TempGridFM | Out-Null
    New-Item -ItemType Directory -Force -Path $JuliaDepot | Out-Null

    $env:TEMP = $TempGridFM
    $env:TMP = $TempGridFM
    $env:JULIA_DEPOT_PATH = $JuliaDepot

    if (-not [string]::IsNullOrWhiteSpace($JuliaBin)) {
        if (Test-Path $JuliaBin) {
            $env:Path = "$JuliaBin;$env:Path"
        }
        else {
            Write-Warning "Julia bin directory not found: $JuliaBin"
        }
    }

    # ==================================================================================
    # Project data layout
    # ==================================================================================

    $DataRoot = Join-Path $ProjectRoot "data"

    $GeneratedRoot = Join-Path $DataRoot "gridfm_generated"
    $TransitionsRoot = Join-Path $DataRoot "gridfm_transitions"
    $ScratchRoot = Join-Path $DataRoot "_scratch"
    $DatasetsRoot = Join-Path $DataRoot "datasets"
    $SelfPlayRoot = Join-Path $DataRoot "self_play"
    $TrainingRoot = Join-Path $DataRoot "training"

    New-Item -ItemType Directory -Force -Path $GeneratedRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $TransitionsRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $ScratchRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $DatasetsRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $SelfPlayRoot | Out-Null
    New-Item -ItemType Directory -Force -Path $TrainingRoot | Out-Null

    # ==================================================================================
    # Shared network settings
    # ==================================================================================

    $NetworkName = "case118_ieee"
    $NetworkSource = "pglib"
    $RawNetworkDirName = $NetworkName

    # ==================================================================================
    # Shared physical / scenario settings
    # ==================================================================================

    $Sigma = 0.20
    $GlobalRange = 0.50
    $MaxScalingFactor = 2.00
    $StepSize = 0.10
    $StartScalingFactor = 1.00

    $TopologyVariants = 10
    $TopologyK = 1

    $GenerationPerturbationType = "none"
    $GenerationPerturbationSigma = 0.0

    $AdmittancePerturbationType = "none"
    $AdmittancePerturbationSigma = 0.0

    # ==================================================================================
    # Shared selection thresholds
    # ==================================================================================

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

    # ==================================================================================
    # Profile-specific settings
    # ==================================================================================

    $AllowPartial = $false

    if ($Profile -eq "smoke") {
        # Smoke checks the entire generation path with the same perturbation
        # mechanisms that will be used in the larger runs.
        $DatasetName = "case118_smoke_v1"

        $OutputRoot = Join-Path $ScratchRoot $DatasetName
        $TransitionExportRoot = Join-Path $OutputRoot "transitions_export"

        $TargetTotal = 50
        $ChunkSize = 200
        $MaxChunks = 3
        $SeedStart = 10000

        $SimpleFraction = 0.25
        $MediumFraction = 0.50
        $HardFraction = 0.25

        $GenerationPerturbationType = "cost_perturbation"
        $GenerationPerturbationSigma = 0.10

        $AdmittancePerturbationType = "random_perturbation"
        $AdmittancePerturbationSigma = 0.02

        $AllowPartial = $true
    }
    elseif ($Profile -eq "eval") {
        # Fixed evaluation set. Do not use it for training.
        $DatasetName = "case118_eval_v1"

        $OutputRoot = Join-Path $GeneratedRoot $DatasetName
        $TransitionExportRoot = Join-Path $TransitionsRoot $DatasetName

        $TargetTotal = 500
        $ChunkSize = 1000
        $MaxChunks = 20
        $SeedStart = 90000

        $SimpleFraction = 0.25
        $MediumFraction = 0.50
        $HardFraction = 0.25

        $GenerationPerturbationType = "cost_perturbation"
        $GenerationPerturbationSigma = 0.10

        $AdmittancePerturbationType = "random_perturbation"
        $AdmittancePerturbationSigma = 0.02
    }
    elseif ($Profile -eq "bootstrap") {
        # Main supervised bootstrap dataset.
        $DatasetName = "case118_bootstrap_v1"

        $OutputRoot = Join-Path $GeneratedRoot $DatasetName
        $TransitionExportRoot = Join-Path $TransitionsRoot $DatasetName

        $TargetTotal = 12000
        $ChunkSize = 2000
        $MaxChunks = 80
        $SeedStart = 20000

        $SimpleFraction = 0.20
        $MediumFraction = 0.50
        $HardFraction = 0.30

        $GenerationPerturbationType = "cost_perturbation"
        $GenerationPerturbationSigma = 0.20

        $AdmittancePerturbationType = "random_perturbation"
        $AdmittancePerturbationSigma = 0.05
    }
    else {
        throw "Unknown profile: $Profile"
    }

    # ==================================================================================
    # CLI arguments
    # ==================================================================================

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

    # ==================================================================================
    # Summary
    # ==================================================================================

    Write-Host "===================================================================================================="
    Write-Host "Building GridFM dataset"
    Write-Host "===================================================================================================="
    Write-Host "Profile:                   $Profile"
    Write-Host "Dataset:                   $DatasetName"
    Write-Host "Target total:              $TargetTotal"
    Write-Host "Chunk size:                $ChunkSize"
    Write-Host "Max chunks:                $MaxChunks"
    Write-Host "Seed start:                $SeedStart"
    Write-Host "Network:                   $NetworkName"
    Write-Host "Output root:               $OutputRoot"
    Write-Host "Transition export root:    $TransitionExportRoot"
    Write-Host "Resume:                    $Resume"
    Write-Host "Allow partial:             $AllowPartial"
    Write-Host "Generation perturbation:   $GenerationPerturbationType"
    Write-Host "Generation sigma:          $GenerationPerturbationSigma"
    Write-Host "Admittance perturbation:   $AdmittancePerturbationType"
    Write-Host "Admittance sigma:          $AdmittancePerturbationSigma"
    Write-Host ""
    Write-Host "Runtime/cache settings:"
    Write-Host "  TEMP:                    $env:TEMP"
    Write-Host "  TMP:                     $env:TMP"
    Write-Host "  JULIA_DEPOT_PATH:        $env:JULIA_DEPOT_PATH"
    Write-Host "  Julia bin:               $JuliaBin"
    Write-Host "  Python:                  $PythonExecutable"
    Write-Host ""
    Write-Host "GridFM command template:"
    Write-Host "  $GridFMCommandTemplate"
    Write-Host ""

    # ==================================================================================
    # Run generation and balanced transition builder
    # ==================================================================================

    $PipelineLogDirectory = Join-Path $OutputRoot "pipeline_logs"
    $PipelineStdoutLog = Join-Path $PipelineLogDirectory "python_stdout.log"
    $PipelineStderrLog = Join-Path $PipelineLogDirectory "python_stderr.log"

    Invoke-NativeProcess `
        -FilePath $PythonExecutable `
        -Arguments $CliArgs `
        -WorkingDirectory $ProjectRoot `
        -StdoutLogPath $PipelineStdoutLog `
        -StderrLogPath $PipelineStderrLog

    # ==================================================================================
    # Export stable transition files
    # ==================================================================================

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

    $TransitionFiles = Get-ChildItem `
        -Path $SourceTransitionsDir `
        -Filter "*.csv" `
        -File `
        -ErrorAction Stop

    if ($TransitionFiles.Count -eq 0) {
        throw "No transition CSV files found in: $SourceTransitionsDir"
    }

    foreach ($TransitionFile in $TransitionFiles) {
        Copy-Item `
            -Path $TransitionFile.FullName `
            -Destination $TransitionExportRoot `
            -Force
    }

    if (Test-Path $SourceManifestDir) {
        New-Item -ItemType Directory -Force -Path $TargetManifestDir | Out-Null

        Get-ChildItem `
            -Path $SourceManifestDir `
            -File `
            -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Extension -in @(".csv", ".json")
            } |
            ForEach-Object {
                Copy-Item `
                    -Path $_.FullName `
                    -Destination $TargetManifestDir `
                    -Force
            }
    }

    $RawDir = Join-Path $OutputRoot "raw"

    if (-not (Test-Path $RawDir)) {
        throw "Merged raw directory not found: $RawDir"
    }

    # ==================================================================================
    # Dataset metadata
    # ==================================================================================

    $DatasetInfo = [ordered]@{
        profile = $Profile
        dataset_name = $DatasetName
        project_root = $ProjectRoot
        output_root = $OutputRoot
        raw_dir = $RawDir
        transition_export_root = $TransitionExportRoot
        transitions_balanced = (
            Join-Path $TransitionExportRoot "transitions_balanced.csv"
        )
        transitions_train = (
            Join-Path $TransitionExportRoot "transitions_train.csv"
        )
        transitions_val = (
            Join-Path $TransitionExportRoot "transitions_val.csv"
        )
        manifest_dir = $TargetManifestDir
        target_total = $TargetTotal
        chunk_size = $ChunkSize
        max_chunks = $MaxChunks
        seed_start = $SeedStart
        network_name = $NetworkName
        network_source = $NetworkSource
        topology_variants = $TopologyVariants
        topology_k = $TopologyK
        generation_perturbation = [ordered]@{
            type = $GenerationPerturbationType
            sigma = $GenerationPerturbationSigma
        }
        admittance_perturbation = [ordered]@{
            type = $AdmittancePerturbationType
            sigma = $AdmittancePerturbationSigma
        }
        python_executable = $PythonExecutable
        gridfm_command_template = $GridFMCommandTemplate
        resume = [bool]$Resume
    }

    $DatasetInfoPath = Join-Path $TransitionExportRoot "dataset_info.json"

    $DatasetInfo |
        ConvertTo-Json -Depth 10 |
        Set-Content -Path $DatasetInfoPath -Encoding UTF8

    # ==================================================================================
    # Final summary
    # ==================================================================================

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
    Write-Host "Dataset info:"
    Write-Host "  $DatasetInfoPath"
    Write-Host ""
}
finally {
    Pop-Location
}
