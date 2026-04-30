@echo off
setlocal
set PYTHONUTF8=1

if "%KEUMJM_ACCESS_MODE%"=="" set KEUMJM_ACCESS_MODE=lan
if "%KEUMJM_HOST%"=="" set KEUMJM_HOST=0.0.0.0
if "%KEUMJM_PORT%"=="" set KEUMJM_PORT=8515
if "%KEUMJM_AUTH_COOKIE_SECURE%"=="" set KEUMJM_AUTH_COOKIE_SECURE=1
if "%KEUMJM_SSL_CERTFILE%"=="" set KEUMJM_SSL_CERTFILE=certs\keumjm-lan.crt
if "%KEUMJM_SSL_KEYFILE%"=="" set KEUMJM_SSL_KEYFILE=certs\keumjm-lan.key

set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

if not exist "%KEUMJM_SSL_CERTFILE%" (
  "%PYTHON_EXE%" scripts\create_https_cert.py --certfile "%KEUMJM_SSL_CERTFILE%" --keyfile "%KEUMJM_SSL_KEYFILE%"
  if errorlevel 1 exit /b %ERRORLEVEL%
)
if not exist "%KEUMJM_SSL_KEYFILE%" (
  "%PYTHON_EXE%" scripts\create_https_cert.py --certfile "%KEUMJM_SSL_CERTFILE%" --keyfile "%KEUMJM_SSL_KEYFILE%"
  if errorlevel 1 exit /b %ERRORLEVEL%
)

echo.
echo Keumjm Portfolio Lab HTTPS LAN mode
echo Local:   https://localhost:%KEUMJM_PORT%
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '169.254.*' -and $_.IPAddress -ne '127.0.0.1' } | ForEach-Object { 'LAN:     https://' + $_.IPAddress + ':%KEUMJM_PORT%' }"
echo Mode:    %KEUMJM_ACCESS_MODE%
echo.

"%PYTHON_EXE%" scripts\run_uvicorn.py
exit /b %ERRORLEVEL%
