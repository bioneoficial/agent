param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

# Resolve script directory
$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir = Split-Path -Parent $ScriptPath

# Load environment variables from .env if present
$DotEnvPath = Join-Path $ScriptDir ".env"
if (Test-Path $DotEnvPath) {
    Get-Content $DotEnvPath | ForEach-Object {
        $line = $_.Trim()
        if ([string]::IsNullOrWhiteSpace($line)) { return }
        if ($line.StartsWith('#')) { return }
        $kv = $line -split '=', 2
        if ($kv.Length -eq 2) {
            $key = $kv[0].Trim()
            $val = $kv[1].Trim()
            if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
                $val = $val.Substring(1, $val.Length - 2)
            }
            Set-Item -Path Env:$key -Value $val | Out-Null
        }
    }
    Write-Host "Loaded environment from $DotEnvPath"
    if ($env:LLM_PROVIDER -or $env:LLM_MODEL) {
        Write-Host "Using provider: $($env:LLM_PROVIDER) with model: $($env:LLM_MODEL)"
    }
}

$MainScript = Join-Path $ScriptDir "main.py"

# Try to find a suitable Python executable (prefer venvs)
$pyCandidates = @(
    (Join-Path $ScriptDir "venv\Scripts\python.exe"),
    (Join-Path $ScriptDir ".venv\Scripts\python.exe"),
    'python',
    'python3',
    'py'
)

$PythonExe = $null
foreach ($cand in $pyCandidates) {
    if ($cand -in @('python','python3','py')) {
        $PythonExe = $cand
        break
    } elseif (Test-Path $cand) {
        $PythonExe = $cand
        break
    }
}
if (-not $PythonExe) { $PythonExe = 'python' }

# Execute
& $PythonExe $MainScript @Args
