$ErrorActionPreference = "Stop"

function Import-DotEnvFile {
  param(
    [string]$Path = ".env"
  )

  if (-not (Test-Path $Path)) {
    return
  }

  foreach ($line in Get-Content $Path) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
      continue
    }
    $separatorIndex = $trimmed.IndexOf("=")
    if ($separatorIndex -lt 1) {
      continue
    }
    $name = $trimmed.Substring(0, $separatorIndex).Trim()
    $value = $trimmed.Substring($separatorIndex + 1).Trim()
    if (
      ($value.StartsWith('"') -and $value.EndsWith('"')) -or
      ($value.StartsWith("'") -and $value.EndsWith("'"))
    ) {
      $value = $value.Substring(1, $value.Length - 2)
    }
    $existing = Get-Item -Path "Env:$name" -ErrorAction SilentlyContinue
    if (-not [string]::IsNullOrWhiteSpace($name) -and $null -eq $existing) {
      Set-Item -Path "Env:$name" -Value $value
    }
  }
}

Import-DotEnvFile

if (-not $env:PYTHONUTF8) { $env:PYTHONUTF8 = "1" }
if (-not $env:KEUMJM_ACCESS_MODE) { $env:KEUMJM_ACCESS_MODE = "lan" }
if (-not $env:KEUMJM_HOST) { $env:KEUMJM_HOST = "0.0.0.0" }
if (-not $env:KEUMJM_PORT) { $env:KEUMJM_PORT = "8515" }
if (-not $env:KEUMJM_AUTH_ENABLED) { $env:KEUMJM_AUTH_ENABLED = "0" }
if (-not $env:ENABLE_MACRO) { $env:ENABLE_MACRO = "0" }

$python = "python"
if (Test-Path ".venv\Scripts\python.exe") {
  $python = ".venv\Scripts\python.exe"
}

$addresses = Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne "127.0.0.1" } |
  Select-Object -ExpandProperty IPAddress

Write-Host ""
Write-Host "Keumjm Portfolio Lab LAN mode"
Write-Host "Local:   http://localhost:$env:KEUMJM_PORT"
foreach ($address in $addresses) {
  Write-Host "LAN:     http://${address}:$env:KEUMJM_PORT"
}
Write-Host "Mode:    $env:KEUMJM_ACCESS_MODE"
Write-Host "Macro:   $env:ENABLE_MACRO"
Write-Host "Auth:    $env:KEUMJM_AUTH_ENABLED"
Write-Host ""

& $python scripts/run_uvicorn.py
