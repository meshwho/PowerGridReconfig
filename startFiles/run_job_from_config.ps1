param(
    [string]$ConfigPath = "D:\проекты\PowerGridReconfig\StartFiles\job_config.json"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

$config = Get-Content $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json

$WorkingDir = [string]$config.working_dir
$Executable = [string]$config.executable
$LogPath = [string]$config.log_path
$JobName = [string]$config.job_name

if ([string]::IsNullOrWhiteSpace($WorkingDir)) {
    throw "working_dir is empty in config"
}

if ([string]::IsNullOrWhiteSpace($Executable)) {
    throw "executable is empty in config"
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    throw "log_path is empty in config"
}

$LogDir = Split-Path $LogPath -Parent
$StatusPath = Join-Path $LogDir "job_status.json"
$CommandPath = Join-Path $LogDir "command.txt"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-JobStatus {
    param(
        [string]$State,
        [int]$ExitCode = -999
    )

    $status = [ordered]@{
        job_name = $JobName
        state = $State
        exit_code = $ExitCode
        time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        working_dir = $WorkingDir
        executable = $Executable
        log_path = $LogPath
    }

    $status | ConvertTo-Json -Depth 5 | Set-Content -Path $StatusPath -Encoding UTF8
}

Set-Location $WorkingDir

if ($config.env) {
    $config.env.PSObject.Properties | ForEach-Object {
        [Environment]::SetEnvironmentVariable($_.Name, [string]$_.Value, "Process")
    }
}

if ($config.prepend_path) {
    foreach ($p in $config.prepend_path) {
        if (Test-Path $p) {
            $env:Path = "$p;$env:Path"
        }
    }
}

if ($env:TEMP) {
    New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
}

if ($env:TMP) {
    New-Item -ItemType Directory -Force -Path $env:TMP | Out-Null
}

if ($env:JULIA_DEPOT_PATH) {
    New-Item -ItemType Directory -Force -Path $env:JULIA_DEPOT_PATH | Out-Null
}

$Arguments = @()

if ($config.arguments) {
    foreach ($arg in $config.arguments) {
        $Arguments += [string]$arg
    }
}

$CommandText = $Executable + " " + ($Arguments -join " ")
$CommandText | Set-Content -Path $CommandPath -Encoding UTF8

Write-JobStatus -State "running"

"====================================================================================================" | Out-File $LogPath -Encoding UTF8
"START JOB: $(Get-Date)" | Out-File $LogPath -Encoding UTF8 -Append
"Job name: $JobName" | Out-File $LogPath -Encoding UTF8 -Append
"Working dir: $WorkingDir" | Out-File $LogPath -Encoding UTF8 -Append
"Executable: $Executable" | Out-File $LogPath -Encoding UTF8 -Append
"Log path: $LogPath" | Out-File $LogPath -Encoding UTF8 -Append
"Command:" | Out-File $LogPath -Encoding UTF8 -Append
$CommandText | Out-File $LogPath -Encoding UTF8 -Append
"====================================================================================================" | Out-File $LogPath -Encoding UTF8 -Append

try {
    & $Executable @Arguments 2>&1 | ForEach-Object {
        $_.ToString() | Out-File $LogPath -Encoding UTF8 -Append
    }

    $ExitCode = $LASTEXITCODE

    "====================================================================================================" | Out-File $LogPath -Encoding UTF8 -Append
    "END JOB: $(Get-Date)" | Out-File $LogPath -Encoding UTF8 -Append
    "Exit code: $ExitCode" | Out-File $LogPath -Encoding UTF8 -Append
    "====================================================================================================" | Out-File $LogPath -Encoding UTF8 -Append

    if ($ExitCode -eq 0) {
        Write-JobStatus -State "finished" -ExitCode $ExitCode
    }
    else {
        Write-JobStatus -State "failed" -ExitCode $ExitCode
    }

    exit $ExitCode
}
catch {
    "====================================================================================================" | Out-File $LogPath -Encoding UTF8 -Append
    "JOB FAILED: $(Get-Date)" | Out-File $LogPath -Encoding UTF8 -Append
    $_.Exception.Message | Out-File $LogPath -Encoding UTF8 -Append
    $_.ScriptStackTrace | Out-File $LogPath -Encoding UTF8 -Append
    "====================================================================================================" | Out-File $LogPath -Encoding UTF8 -Append

    Write-JobStatus -State "failed" -ExitCode 999

    exit 999
}
