# Sets up the project virtual environment and installs dependencies.
# Run from the repository root: .\setup_venv.ps1

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..."
    python -m venv .venv
} else {
    Write-Host ".venv already exists, skipping creation."
}

& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "Installing requirements.txt ..."
pip install -r requirements.txt

if (Test-Path "backend\requirements.txt") {
    Write-Host "Installing backend\requirements.txt ..."
    pip install -r backend\requirements.txt
}

Write-Host "Done. Virtual environment is active in this shell."
Write-Host "To reactivate later: .\.venv\Scripts\Activate.ps1"
