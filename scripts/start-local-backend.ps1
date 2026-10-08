param(
    [int]$Port = 8000,
    [string]$HostName = "127.0.0.1",
    [switch]$Mock
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Backend = Join-Path $Root "backend"

function Find-Python {
    try {
        & py -3.12 -V *> $null
        if ($LASTEXITCODE -eq 0) { return @{ Command = "py"; Args = @("-3.12") } }
    } catch {}

    try {
        & python -V *> $null
        if ($LASTEXITCODE -eq 0) { return @{ Command = "python"; Args = @() } }
    } catch {}

    throw "Python was not found. Install Python 3.12 or make py/python available in PATH."
}

if ($Mock) {
    $env:MODEL_PROVIDER = "mock"
    $env:VLLM_ENABLED = "false"
    $env:VLLM_BASE_URLS = ""
    Write-Warning "Explicit mock mode: real model inference is unavailable."
}
$env:BACKEND_URL = "http://${HostName}:$Port"
if (-not $env:SECURITY_MIDDLEWARE_ENABLED) { $env:SECURITY_MIDDLEWARE_ENABLED = "true" }

$Python = Find-Python
$PythonArgs = @($Python.Args) + @("run.py", "--host", $HostName, "--port", "$Port", "--reload", "--workers", "1")

Write-Host "Starting backend: $($env:BACKEND_URL)" -ForegroundColor Cyan
Write-Host "Provider settings come from the environment and backend/.env or database configuration."
Push-Location $Backend
try {
    & $Python.Command @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Backend process exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
