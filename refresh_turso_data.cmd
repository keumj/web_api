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
echo  Keumj manual update - Part 2
echo ============================================================
echo  Upload incremental data from local SQLite to Turso.
echo  Run refresh_local_data.cmd first when local data is stale.
echo.
echo  Python: %PYTHON_EXE%
echo.
echo  [1] Prices / market caps: local SQLite -^> Turso
echo  [2] Fundamentals: local SQLite -^> Turso
echo  [3] News: local SQLite -^> Turso
echo  [4] Macro: local SQLite -^> Turso
echo  [5] Upload all local data to Turso
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
call :run_upload prices
goto :status

:fundamentals
call :run_upload fundamentals
goto :status

:news
call :run_upload news
goto :status

:macro
call :run_upload macro
goto :status

:all
call :run_upload all
goto :status

:run_upload
echo.
echo ------------------------------------------------------------
echo Uploading local incremental data to Turso: %~1
echo ------------------------------------------------------------
"%PYTHON_EXE%" -u scripts\refresh_turso_daily.py --mode upload-local --target %~1
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%

:status
echo.
echo ============================================================
if "%EXIT_CODE%"=="0" (
  echo  Turso upload finished: SUCCESS
) else (
  echo  Turso upload finished: FAILED ^(exit_code=%EXIT_CODE%^)
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
