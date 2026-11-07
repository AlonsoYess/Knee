param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'El puerto debe estar entre 1024 y 65535.' }
$backendDirectory = Join-Path $PSScriptRoot 'backend'
$recordPath = Join-Path $backendDirectory "runtime\local-server-$Port.json"
if (-not (Test-Path -LiteralPath $recordPath -PathType Leaf)) {
    Write-Output 'No hay un servidor registrado por este iniciador. No se detuvo ningun proceso.'
    exit 0
}
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
if ($record.port -ne $Port -or $record.backend -ne $backendDirectory) { throw 'El registro no corresponde a esta aplicacion.' }
$serverProcess = Get-Process -Id $record.pid -ErrorAction SilentlyContinue
if (-not $serverProcess) { Write-Output 'El servidor registrado ya esta detenido.'; exit 0 }
$processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $($serverProcess.Id)"
$expectedPython = Join-Path $backendDirectory '.venv\Scripts\python.exe'
# PowerShell 7 can decode ISO JSON strings as DateTime; Windows PowerShell 5
# leaves them as strings. Compare the same exact UTC ticks in both versions.
if ($record.startUtc -is [DateTime]) { $recordedStartTicks = $record.startUtc.ToUniversalTime().Ticks }
else { $recordedStartTicks = [DateTimeOffset]::ParseExact([string]$record.startUtc, 'o', [Globalization.CultureInfo]::InvariantCulture).UtcDateTime.Ticks }
if ($serverProcess.StartTime.ToUniversalTime().Ticks -ne $recordedStartTicks -or
    $processInfo.ExecutablePath -ne $expectedPython -or
    $processInfo.CommandLine -notmatch ('-m\s+uvicorn\s+knee_app\.api:create_app\s+--factory\s+--host\s+127\.0\.0\.1\s+--port\s+' + $Port + '(\s|$)')) {
    throw 'El proceso no coincide con el servidor registrado. No se detuvo nada.'
}
& taskkill.exe /PID "$($serverProcess.Id)" /T /F | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'No se pudo confirmar la detencion del servidor registrado.' }
Write-Output 'Servidor local de KneeOAI detenido. Las evaluaciones guardadas se conservan. Cierra sus ventanas si terminaste.'
