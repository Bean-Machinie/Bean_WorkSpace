$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host ".venv not found."
    Write-Host "Run: py -m venv .venv"
    Write-Host "     .\\.venv\\Scripts\\Activate.ps1"
    Write-Host "     pip install -e .[dev]"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

& $pythonExe -m space_dynamics_workbench.app.main
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Application exited with an error."
    Read-Host "Press Enter to exit"
}
