$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Create the project .venv first.' }
Push-Location $projectRoot
try {
    & $pythonExe -B (Join-Path $PSScriptRoot 'server.py') --mode local
    if ($LASTEXITCODE -ne 0) { throw 'MLflow server exited unsuccessfully; inspect local logs.' }
} finally { Pop-Location }
