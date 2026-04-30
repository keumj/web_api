@echo off
setlocal
set PYTHONUTF8=1

if "%KEUMJM_ACCESS_MODE%"=="" set KEUMJM_ACCESS_MODE=lan
if "%KEUMJM_HOST%"=="" set KEUMJM_HOST=0.0.0.0
if "%KEUMJM_PORT%"=="" set KEUMJM_PORT=8515

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

echo.
echo Keumjm Portfolio Lab LAN mode
echo Local:   http://localhost:%KEUMJM_PORT%
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1' } | ForEach-Object { 'LAN:     http://' + $_.IPAddress + ':%KEUMJM_PORT%' }"
echo Mode:    %KEUMJM_ACCESS_MODE%
echo.

"%PYTHON_EXE%" scripts\run_uvicorn.py
exit /b %ERRORLEVEL%
