$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { $pythonExe = 'python' }
$logDirectory = Join-Path $projectRoot 'outputs\global_signals'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory ('refresh-' + (Get-Date -Format 'yyyy-MM-dd') + '.log')
# Local scheduled refresh preserves the SQLite recovery source before uploading.
$ErrorActionPreference = 'Continue' # Native provider stderr must not skip fail-closed publication.
& $pythonExe -B -m global_signals.refresh --mode local >> $logPath 2>&1
$collectionExit = $LASTEXITCODE
# Publish failed/missing inputs too, to replace yesterday's research colors with gray.
& $pythonExe -B -m global_signals.refresh --mode upload-local >> $logPath 2>&1
$uploadExit = $LASTEXITCODE
if ($collectionExit -ne 0) { exit $collectionExit }
exit $uploadExit
