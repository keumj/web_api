@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if "%KEUMJM_ACCESS_MODE%"=="" set "KEUMJM_ACCESS_MODE=lan"
if "%KEUMJM_HOST%"=="" set "KEUMJM_HOST=0.0.0.0"
if "%KEUMJM_PORT%"=="" set "KEUMJM_PORT=8515"

if "%KEUMJM_ENSURE_SERVER_DRY_RUN%"=="1" (
  echo Dry run: would ensure local FastAPI server on port %KEUMJM_PORT%.
  endlocal & exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$port = [int]$env:KEUMJM_PORT; " ^
  "$root = (Get-Location).Path; " ^
  "$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; " ^
  "if ($listener) { Write-Host ('FastAPI server already listening on port ' + $port); exit 0 }; " ^
  "$script = Join-Path $root 'run_lan_service.cmd'; " ^
  "if (-not (Test-Path $script)) { Write-Error ('Missing server script: ' + $script); exit 1 }; " ^
  "Start-Process -FilePath $script -WorkingDirectory $root -WindowStyle Hidden; " ^
  "Start-Sleep -Seconds 3; " ^
  "$started = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1; " ^
  "if ($started) { Write-Host ('Started FastAPI server on port ' + $port); exit 0 }; " ^
  "Write-Host ('FastAPI server start requested; port ' + $port + ' is not listening yet.'); exit 0"

endlocal & exit /b %ERRORLEVEL%
