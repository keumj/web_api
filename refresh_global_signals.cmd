@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
set "MODE=%~1"
if "%MODE%"=="" set "MODE=direct"
if /i "%MODE%"=="init" (
  "%PYTHON_EXE%" -B -m global_signals.init_db
) else (
  "%PYTHON_EXE%" -B -m global_signals.refresh --mode "%MODE%"
)
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
