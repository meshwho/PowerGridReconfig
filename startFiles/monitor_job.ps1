param(
    [string]$ConfigPath = "D:\проекты\PowerGridReconfig\StartFiles\job_config.json",
    [string]$TaskName = "PowerGridJobRunner",
    [int]$IntervalSec = 5,
    [int]$Tail = 25
)

$ErrorActionPreference = "SilentlyContinue"

function Get-SystemSnapshot {
    $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    $os = Get-CimInstance Win32_OperatingSystem

    $total = [math]::Round($os.TotalVisibleMemorySize / 1024, 1)
    $free = [math]::Round($os.FreePhysicalMemory / 1024, 1)
    $used = [math]::Round($total - $free, 1)
    $usedPercent = [math]::Round(($used / $total) * 100, 1)

    [pscustomobject]@{
        CPU = [math]::Round($cpu, 1)
        RAM_Total_MB = $total
        RAM_Used_MB = $used
        RAM_Free_MB = $free
        RAM_Used_Percent = $usedPercent
    }
}

function Get-PythonProcesses {
    Get-Process python -ErrorAction SilentlyContinue |
        Select-Object Id,
            @{Name='CPU_Total_s';Expression={[math]::Round($_.CPU, 1)}},
            @{Name='RAM_MB';Expression={[math]::Round($_.WorkingSet64 / 1MB, 1)}},
            @{Name='StartTime';Expression={$_.StartTime}}
}

while ($true) {
    Clear-Host

    $config = $null
    $logPath = $null
    $outputDir = $null

    if (Test-Path $ConfigPath) {
        $config = Get-Content $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $logPath = [string]$config.log_path

        if ($config.arguments) {
            $outIndex = -1
            for ($i = 0; $i -lt $config.arguments.Count; $i++) {
                if ([string]$config.arguments[$i] -eq "--output-dir") {
                    $outIndex = $i + 1
                    break
                }
            }

            if ($outIndex -ge 0 -and $outIndex -lt $config.arguments.Count) {
                $outputDir = Join-Path ([string]$config.working_dir) ([string]$config.arguments[$outIndex])
            }
        }
    }

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $taskInfo = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
    $sys = Get-SystemSnapshot

    Write-Host "===================================================================================================="
    Write-Host "POWERGRID JOB MONITOR"
    Write-Host "===================================================================================================="
    Write-Host ("Time:        {0}" -f (Get-Date))
    Write-Host ("Task:        {0}" -f $TaskName)
    Write-Host ("Task state:  {0}" -f $task.State)
    Write-Host ("Last result: {0}" -f $taskInfo.LastTaskResult)
    Write-Host ("Job name:    {0}" -f $config.job_name)
    Write-Host ("Log:         {0}" -f $logPath)
    Write-Host ""
    Write-Host ("CPU total:   {0:N1}%" -f $sys.CPU)
    Write-Host ("RAM used:    {0:N1} MB / {1:N1} MB ({2:N1}%)" -f $sys.RAM_Used_MB, $sys.RAM_Total_MB, $sys.RAM_Used_Percent)
    Write-Host ("RAM free:    {0:N1} MB" -f $sys.RAM_Free_MB)
    Write-Host ""

    Write-Host "Python processes:"
    $py = Get-PythonProcesses
    if ($py) {
        $py | Sort-Object RAM_MB -Descending | Format-Table -AutoSize
    }
    else {
        Write-Host "No python.exe processes found."
    }

    Write-Host ""

    if ($outputDir -and (Test-Path $outputDir)) {
        $statesDir = Join-Path $outputDir "states"
        $examplesCsv = Join-Path $outputDir "examples.csv"

        if (Test-Path $statesDir) {
            $stateCount = (Get-ChildItem $statesDir -Filter "*.npz" -ErrorAction SilentlyContinue | Measure-Object).Count
            Write-Host ("State files: {0}" -f $stateCount)
        }

        if (Test-Path $examplesCsv) {
            Write-Host ("Examples CSV exists: {0}" -f $examplesCsv)
        }
    }

    Write-Host ""
    Write-Host "Last log lines:"
    Write-Host "----------------------------------------------------------------------------------------------------"

    if ($logPath -and (Test-Path $logPath)) {
        Get-Content $logPath -Tail $Tail
    }
    else {
        Write-Host "Log file not found yet."
    }

    Write-Host ""
    Write-Host "Press Ctrl+C to exit monitor. The job will continue running."
    Start-Sleep -Seconds $IntervalSec
}
