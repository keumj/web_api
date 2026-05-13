@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "EXIT_CODE=0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

if "%FRED_API_KEY%"=="" (
  for /f "tokens=2,*" %%A in ('reg query HKCU\Environment /v FRED_API_KEY 2^>nul') do set "FRED_API_KEY=%%B"
)

echo.
echo ============================================================
echo  Keumj Turso data refresh / upload
echo ============================================================
echo  This updates Turso calculation DBs.
echo  Local SQLite files are kept local and are not committed.
echo.
echo  Python: %PYTHON_EXE%
echo.
echo  [1] Direct refresh: provider -^> Turso ^(S^&P 500 prices/fundamentals/news + macro^)
echo  [2] Direct refresh: provider -^> Turso ^(S^&P 500 prices/fundamentals/news only^)
echo  [3] Direct refresh: provider -^> Turso ^(macro only^)
echo  [4] Fallback upload: local SQLite -^> Turso ^(S^&P 500 prices/fundamentals/news + macro^)
echo  [5] Fallback upload: local SQLite -^> Turso ^(S^&P 500 prices/fundamentals/news only^)
echo  [6] Fallback upload: local SQLite -^> Turso ^(macro only^)
echo  [0] Exit
echo.

if not "%~1"=="" (
  set "CHOICE=%~1"
) else (
  set /p "CHOICE=Choose an option: "
)

if "%CHOICE%"=="0" goto :done
if "%CHOICE%"=="1" goto :direct_all
if "%CHOICE%"=="2" goto :direct_sp500
if "%CHOICE%"=="3" goto :direct_macro
if "%CHOICE%"=="4" goto :upload_all
if "%CHOICE%"=="5" goto :upload_sp500
if "%CHOICE%"=="6" goto :upload_macro
if /i "%CHOICE%"=="direct" goto :direct_all
if /i "%CHOICE%"=="direct-all" goto :direct_all
if /i "%CHOICE%"=="direct-sp500" goto :direct_sp500
if /i "%CHOICE%"=="direct-macro" goto :direct_macro
if /i "%CHOICE%"=="upload" goto :upload_all
if /i "%CHOICE%"=="upload-local" goto :upload_all
if /i "%CHOICE%"=="upload-sp500" goto :upload_sp500
if /i "%CHOICE%"=="upload-macro" goto :upload_macro

echo Invalid option.
goto :done

:direct_all
call :run_refresh --mode direct --target all
goto :status

:direct_sp500
call :run_refresh --mode direct --target sp500
goto :status

:direct_macro
call :run_refresh --mode direct --target macro
goto :status

:upload_all
call :run_refresh --mode upload-local --target all
goto :status

:upload_sp500
call :run_refresh --mode upload-local --target sp500
goto :status

:upload_macro
call :run_refresh --mode upload-local --target macro
goto :status

:run_refresh
echo.
echo ------------------------------------------------------------
echo Running: scripts\refresh_turso_daily.py %*
echo ------------------------------------------------------------
"%PYTHON_EXE%" -u scripts\refresh_turso_daily.py %*
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%

:status
echo.
echo ============================================================
if "%EXIT_CODE%"=="0" (
  echo  Turso refresh finished: SUCCESS
) else (
  echo  Turso refresh finished: FAILED ^(exit_code=%EXIT_CODE%^)
)
echo ============================================================
echo.
echo Current local Git data status:
git status --short data
echo.
echo Tip:
echo   Render Cron default command:
echo     python scripts/refresh_turso_daily.py
echo   Local fallback upload:
echo     refresh_turso_data.cmd upload-local
echo.

:done
endlocal & exit /b %EXIT_CODE%
