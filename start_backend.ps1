# Starts the SkyPredict FastAPI backend.
# Creates the .venv and installs backend dependencies first if they are missing.
# Run from the repository root: .\start_backend.ps1 [-Port 8000] [-BindHost 127.0.0.1]

param(
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = ".\.venv\Scripts\python.exe"

function Test-BackendReady {
    if (-not (Test-Path $venvPython)) { return $false }
    # A failed import is an expected signal (deps missing); don't let it abort.
    try { & $venvPython -c "import fastapi, uvicorn" 2>$null } catch { return $false }
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-BackendReady)) {
    if (-not (Test-Path ".venv")) {
        Write-Host "Creating virtual environment in .venv ..."
        python -m venv .venv
    }
    Write-Host "Installing backend dependencies ..."
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r backend\requirements.txt
}

Write-Host "Starting backend on http://${BindHost}:${Port}  (docs: /docs, stop: Ctrl+C)"
& $venvPython -m uvicorn backend.main:app --host $BindHost --port $Port --reload
