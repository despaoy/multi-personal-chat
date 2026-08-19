param(
    [switch]$Frontend
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
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

function Invoke-Step {
    param(
        [string]$Name,
        [string]$WorkDir,
        [string]$Command,
        [string[]]$Arguments = @()
    )

    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    Push-Location $WorkDir
    try {
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$Name failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

$Python = Find-Python
$PyArgs = @($Python.Args)
$PytestTmp = Join-Path ([IO.Path]::GetTempPath()) ("multi-personal-chat-pytest-{0}" -f $PID)

$CompileTargets = @(
    "app/main.py",
    "api/generate.py",
    "api/loras.py",
    "api/stats.py",
    "api/integrations.py",
    "bot/bot.py",
    "db/database.py",
    "db/pg_database.py",
    "inference/vllm_client.py",
    "inference/model_manager.py",
    "infra/circuit_breaker.py",
    "infra/deployment.py",
    "scripts/local_smoke.py"
)

Invoke-Step "Python syntax check" $Backend $Python.Command ($PyArgs + @("-m", "py_compile") + $CompileTargets)
# Alembic 不提供 __main__ 入口，`python -m alembic` 会报
# "No module named alembic.__main__"。改为从所选 Python 的 sysconfig
# 定位 alembic 可执行文件，确保 Python 和 Alembic 使用同一环境。
$AlembicCandidate = & $Python.Command @($PyArgs + @("-c", "import sysconfig,os; print(os.path.join(sysconfig.get_path('scripts'),'alembic'+('.exe' if os.name=='nt' else '')))"))
if (-not [string]::IsNullOrWhiteSpace($AlembicCandidate)) {
    $AlembicCandidate = $AlembicCandidate.Trim()
}
if ([string]::IsNullOrWhiteSpace($AlembicCandidate) -or -not (Test-Path $AlembicCandidate)) {
    # 回退到 backend venv 或 PATH 上的 alembic
    $VenvAlembic = Join-Path $Backend ".venv/Scripts/alembic.exe"
    if (Test-Path $VenvAlembic) {
        $AlembicExe = $VenvAlembic
    } else {
        $AlembicExe = "alembic"
    }
} else {
    $AlembicExe = $AlembicCandidate
}
if (Get-Command $AlembicExe -ErrorAction SilentlyContinue) {
Invoke-Step "Alembic migration graph check" $Backend $AlembicExe @("heads")
} else {
    Write-Host ""
    Write-Host "Alembic not found; skipped migration graph check." -ForegroundColor DarkGray
}
Invoke-Step "Backend core tests" $Backend $Python.Command ($PyArgs + @(
    "-m", "pytest", "tests", "-q", "--basetemp", $PytestTmp
))
Invoke-Step "API smoke test and mock AstrBot event" $Backend $Python.Command ($PyArgs + @("-m", "scripts.local_smoke"))
Invoke-Step "Git whitespace check" $Root "git" @("diff", "--check")

if ($Frontend) {
    Invoke-Step "Frontend TypeScript check" $Root "pnpm" @("ts-check")
    Invoke-Step "Frontend lint" $Root "pnpm" @("lint")
    Invoke-Step "Frontend production build" $Root "pnpm" @("build")
    Invoke-Step "Frontend production dependency audit" $Root "pnpm" @("audit", "--prod", "--registry=https://registry.npmjs.org")
} else {
    Write-Host ""
    Write-Host "Skipped frontend TypeScript check. Run: pnpm verify:local:frontend" -ForegroundColor DarkGray
}

Write-Host ""

$TempPaths = @(
    @{ Path = (Join-Path $Backend ".test_tmp"); Root = $Backend },
    @{ Path = $PytestTmp; Root = [IO.Path]::GetTempPath() }
)
foreach ($TempItem in $TempPaths) {
    $ResolvedTmp = Resolve-Path -LiteralPath $TempItem.Path -ErrorAction SilentlyContinue
    if (-not $ResolvedTmp) { continue }

    $ResolvedPath = $ResolvedTmp.Path
    $AllowedRoot = [IO.Path]::GetFullPath($TempItem.Root).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $AllowedPrefix = $AllowedRoot + [IO.Path]::DirectorySeparatorChar
    if (-not $ResolvedPath.StartsWith($AllowedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove test temp outside allowed root: $ResolvedPath"
    }
    Remove-Item -LiteralPath $ResolvedPath -Recurse -Force
}

Write-Host "Local baseline verification completed." -ForegroundColor Green
