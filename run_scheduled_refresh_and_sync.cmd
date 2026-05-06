@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "outputs" mkdir "outputs"

set "LOG_FILE=%CD%\outputs\refresh_local_data_scheduler.log"
set "KEUMJM_AUTO_SYNC_SHARED_DB=0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

if "%FRED_API_KEY%"=="" (
  for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v FRED_API_KEY 2^>nul') do set "FRED_API_KEY=%%B"
)

echo.>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"
echo [%date% %time%] Scheduled refresh started>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"

call ensure_lan_server.cmd >>"%LOG_FILE%" 2>&1

"%PYTHON_EXE%" scripts\record_refresh_state.py started --source scheduled >>"%LOG_FILE%" 2>&1

call refresh_local_data.cmd 5 >>"%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

"%PYTHON_EXE%" scripts\record_refresh_state.py finished --source scheduled --exit-code %EXIT_CODE% >>"%LOG_FILE%" 2>&1

if "%EXIT_CODE%"=="0" (
  echo.>>"%LOG_FILE%"
  echo ------------------------------------------------------------>>"%LOG_FILE%"
  echo SQLite auto sync is disabled. Review the start page and push manually if needed.>>"%LOG_FILE%"
  echo ------------------------------------------------------------>>"%LOG_FILE%"
)

echo [%date% %time%] Scheduled refresh finished with exit_code=%EXIT_CODE%>>"%LOG_FILE%"
echo.>>"%LOG_FILE%"

endlocal & exit /b %EXIT_CODE%
