@echo off
setlocal

set "REPO_ROOT=%~dp0"
set "PYTHON_EXE=%REPO_ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
  echo .venv not found.
  echo Run: py -m venv .venv
  echo      .\.venv\Scripts\Activate.ps1
  echo      pip install -e .[dev]
  echo.
  pause
  exit /b 1
)

"%PYTHON_EXE%" -m space_dynamics_workbench.app.main
if errorlevel 1 (
  echo.
  echo Application exited with an error.
  pause
)
endlocal
