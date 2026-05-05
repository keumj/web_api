@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "outputs" mkdir "outputs"

set "LOG_FILE=%CD%\outputs\refresh_local_data_scheduler.log"
set "KEUMJM_AUTO_SYNC_SHARED_DB=1"

echo.>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"
echo [%date% %time%] Scheduled refresh started>>"%LOG_FILE%"
echo ============================================================>>"%LOG_FILE%"

call refresh_local_data.cmd 4 >>"%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

echo [%date% %time%] Scheduled refresh finished with exit_code=%EXIT_CODE%>>"%LOG_FILE%"
echo.>>"%LOG_FILE%"

endlocal & exit /b %EXIT_CODE%
