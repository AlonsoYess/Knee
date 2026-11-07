param([int]$Port = 8000, [switch]$NoBrowser, [switch]$Recompilar)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'El puerto debe estar entre 1024 y 65535.' }
$backendDirectory = Join-Path $PSScriptRoot 'backend'
$frontendDirectory = Join-Path $PSScriptRoot 'frontend'
$pythonExecutable = Join-Path $backendDirectory '.venv\Scripts\python.exe'
$prepareScript = Join-Path $backendDirectory 'scripts\prepare_local_frontend.py'
$runtimeDirectory = Join-Path $backendDirectory 'runtime'
$localAddress = "http://127.0.0.1:$Port"
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    try {
        $manifest = Invoke-RestMethod -Uri "$localAddress/manifest.webmanifest" -TimeoutSec 3
        $version = Invoke-RestMethod -Uri "$localAddress/pwa-version.json" -TimeoutSec 3
        if ($manifest.short_name -ne 'KneeOAI' -or -not $version.buildId) { throw 'Otro servicio.' }
    } catch { throw "El puerto $Port ya esta ocupado. No se detuvo ningun proceso. Revisa el servidor antes de volver a iniciar KneeOAI." }
    if ($Recompilar) { throw 'KneeOAI esta abierto. Conserva tus cambios y detiene el servidor antes de recompilar; no se recargaron tus ventanas.' }
    Write-Output "KneeOAI ya esta disponible en $localAddress/acceso."
    if (-not $NoBrowser) { Start-Process "$localAddress/acceso" }
    exit 0
}
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) { throw 'Falta el entorno de la aplicacion. Consulta app/README.md para instalarlo una sola vez.' }
New-Item -ItemType Directory -Force -Path $runtimeDirectory | Out-Null
$env:PYTHONUTF8 = '1'
$sourceJson = & $pythonExecutable $prepareScript fingerprint --frontend $frontendDirectory
if ($LASTEXITCODE -ne 0) { throw 'No se pudieron verificar las fuentes de la interfaz.' }
$sourceDigest = $sourceJson | ConvertFrom-Json
$statePath = Join-Path $runtimeDirectory 'frontend-build.json'
$buildState = $null
if (Test-Path -LiteralPath $statePath) {
    try { $buildState = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } catch { $buildState = $null }
}
$needsBuild = $Recompilar -or -not $buildState -or $buildState.sourceSha256 -ne $sourceDigest -or -not (Test-Path -LiteralPath (Join-Path $frontendDirectory 'dist\sw.js'))
if ($needsBuild) {
    if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Hace falta Node.js para preparar esta version. Consulta la instalacion inicial en la guia.' }
    Push-Location $frontendDirectory
    try {
        if (-not (Test-Path -LiteralPath 'node_modules\vite\bin\vite.js')) {
            Write-Output 'Instalando las dependencias de interfaz. Este paso inicial necesita Internet.'
            & npm.cmd ci --ignore-scripts --no-audit --no-fund
            if ($LASTEXITCODE -ne 0) { throw 'No se pudieron instalar las dependencias de interfaz.' }
        }
        Write-Output 'Preparando KneeOAI para uso local...'
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw 'No se completo la compilacion; no se inicio el servidor.' }
    } finally { Pop-Location }
}
$snapshotJson = & $pythonExecutable $prepareScript snapshot --dist (Join-Path $frontendDirectory 'dist') --root (Join-Path $runtimeDirectory 'frontend_versions')
if ($LASTEXITCODE -ne 0) { throw 'No se pudo verificar la distribucion local.' }
$snapshot = $snapshotJson | ConvertFrom-Json
@{sourceSha256=$sourceDigest;snapshotDigest=$snapshot.digest} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
& (Join-Path $PSScriptRoot 'IniciarServidor.ps1') -Port $Port -FrontendDist $snapshot.snapshotPath
if ($LASTEXITCODE -ne 0) { throw 'No se pudo iniciar el servidor de la aplicacion.' }
$available = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $manifest = Invoke-RestMethod -Uri "$localAddress/manifest.webmanifest" -TimeoutSec 2
        if ($manifest.short_name -eq 'KneeOAI') { $available = $true; break }
    } catch { Start-Sleep -Milliseconds 250 }
}
if (-not $available) { throw 'El servidor no confirmo disponibilidad. Revisa backend/runtime/logs; no se cambio tu cuenta.' }
Write-Output "KneeOAI esta disponible en $localAddress/acceso. Mantiene los mismos accesos locales."
if (-not $NoBrowser) { Start-Process "$localAddress/acceso" }
