@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "outputs" mkdir "outputs"

set "LOG_FILE=%CD%\outputs\refresh_local_data_scheduler.log"
set "KEUMJM_AUTO_SYNC_SHARED_DB=0"

if "%FRED_API_KEY%"=="" (
  for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v FRED_API_KEY 2^>nul') do set "FRED_API_KEY=%%B"
)

echo.>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"
echo [%date% %time%] Scheduled refresh started>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"

call refresh_local_data.cmd 5 >>"%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if "%EXIT_CODE%"=="0" (
  echo.>>"%LOG_FILE%"
  echo ------------------------------------------------------------>>"%LOG_FILE%"
  echo SQLite auto sync is disabled. Review the start page and push manually if needed.>>"%LOG_FILE%"
  echo ------------------------------------------------------------>>"%LOG_FILE%"
)

echo [%date% %time%] Scheduled refresh finished with exit_code=%EXIT_CODE%>>"%LOG_FILE%"
echo.>>"%LOG_FILE%"

endlocal & exit /b %EXIT_CODE%
