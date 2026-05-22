@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "EXIT_CODE=0"
set "INTERACTIVE=1"
if not "%~1"=="" set "INTERACTIVE=0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

:menu
set "EXIT_CODE=0"
set "CHOICE="

echo.
echo ============================================================
echo  Keumj Supabase incremental upload
echo ============================================================
echo  Upload changed/recent rows from local SQLite to Supabase
echo  PostgreSQL. Run refresh_local_data.cmd first if local data
echo  is stale.
echo.
echo  Python: %PYTHON_EXE%
echo.
echo  [1] Prices / market caps
echo  [2] Fundamentals
echo  [3] News
echo  [4] Macro
echo  [5] Run all
echo  [0] Exit
echo.

if not "%~1"=="" (
  set "CHOICE=%~1"
) else (
  set /p "CHOICE=Choose an option: "
)

if "%CHOICE%"=="0" goto :done
if "%CHOICE%"=="1" goto :prices
if "%CHOICE%"=="2" goto :fundamentals
if "%CHOICE%"=="3" goto :news
if "%CHOICE%"=="4" goto :macro
if "%CHOICE%"=="5" goto :all
if /i "%CHOICE%"=="prices" goto :prices
if /i "%CHOICE%"=="stock" goto :prices
if /i "%CHOICE%"=="fundamentals" goto :fundamentals
if /i "%CHOICE%"=="quarterly" goto :fundamentals
if /i "%CHOICE%"=="news" goto :news
if /i "%CHOICE%"=="macro" goto :macro
if /i "%CHOICE%"=="all" goto :all

echo Invalid option.
if "%INTERACTIVE%"=="1" goto :menu
goto :done

:prices
call :run_update prices
goto :status

:fundamentals
call :run_update fundamentals
goto :status

:news
call :run_update news
goto :status

:macro
call :run_update macro
goto :status

:all
call :run_update all
goto :status

:run_update
echo.
echo ------------------------------------------------------------
echo Supabase PostgreSQL incremental upload: %~1
echo ------------------------------------------------------------
"%PYTHON_EXE%" -u scripts\refresh_supabase_daily.py --mode upload-local --target %~1
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%

:status
echo.
echo ============================================================
if "%EXIT_CODE%"=="0" (
  echo  Supabase incremental upload finished: SUCCESS
) else (
  echo  Supabase incremental upload finished: FAILED ^(exit_code=%EXIT_CODE%^)
)
echo ============================================================
echo.
echo Local data file status:
git status --short data
echo.
if "%INTERACTIVE%"=="1" (
  pause
  goto :menu
)

:done
endlocal & exit /b %EXIT_CODE%
