@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if "%KEUMJM_PORT%"=="" set "KEUMJM_PORT=8515"

:menu
cls
echo ========================================
echo  Keumjm LAN Server Manager
echo ========================================
echo.
echo  Port: %KEUMJM_PORT%
echo.
echo  1. Start LAN version
echo  2. Stop LAN version
echo  0. Exit
echo.
set /p "CHOICE=Select: "

if "%CHOICE%"=="1" goto start_lan
if "%CHOICE%"=="2" goto stop_lan
if "%CHOICE%"=="0" goto done

echo.
echo Invalid selection.
pause
goto menu

:start_lan
echo.
echo Starting LAN version on port %KEUMJM_PORT%...
call ensure_lan_server.cmd
if %ERRORLEVEL% neq 0 (
  echo Failed to start LAN version.
  pause
  goto menu
)
echo.
echo LAN version is running.
echo Local: http://localhost:%KEUMJM_PORT%
pause
goto menu

:stop_lan
echo.
echo Stopping process listening on port %KEUMJM_PORT%...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$port = [int]$env:KEUMJM_PORT; " ^
  "$listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; " ^
  "if (-not $listeners) { Write-Host ('No process is listening on port ' + $port + '.'); exit 0 }; " ^
  "$pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique; " ^
  "foreach ($processId in $pids) { " ^
  "  $process = Get-Process -Id $processId -ErrorAction SilentlyContinue; " ^
  "  if ($process) { " ^
  "    Write-Host ('Stopping PID ' + $processId + ' (' + $process.ProcessName + ')'); " ^
  "    Stop-Process -Id $processId -Force; " ^
  "  } " ^
  "}; " ^
  "Start-Sleep -Seconds 1; " ^
  "$remaining = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue; " ^
  "if ($remaining) { Write-Error ('Port ' + $port + ' is still listening.'); exit 1 }; " ^
  "Write-Host ('LAN version stopped on port ' + $port + '.'); exit 0"
if %ERRORLEVEL% neq 0 (
  echo Failed to stop LAN version.
  pause
  goto menu
)
pause
goto menu

:done
endlocal & exit /b 0
