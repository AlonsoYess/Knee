param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'El puerto debe estar entre 1024 y 65535.' }
$backendDirectory = Join-Path $PSScriptRoot 'backend'
$pythonExecutable = Join-Path $backendDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Falta instalar el entorno del backend. Consulta app/README.md.' }
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { Write-Output "El puerto $Port ya esta en uso. No se inicio otro servidor."; exit 0 }
$processorConfig = Join-Path $backendDirectory 'runtime\frozen_processor.json'
if (Test-Path -LiteralPath $processorConfig) {
    $configuration = Get-Content -LiteralPath $processorConfig -Raw | ConvertFrom-Json
    $names = @($configuration.PSObject.Properties.Name)
    if ($names.Count -ne 2 -or 'python' -notin $names -or 'checkout' -notin $names) {
        throw 'La configuracion privada del preparador debe contener solamente python y checkout.'
    }
    $processorPython = [string]$configuration.python
    $processorCheckout = [string]$configuration.checkout
    if (-not [System.IO.Path]::IsPathRooted($processorPython) -or -not (Test-Path -LiteralPath $processorPython -PathType Leaf) -or
        -not [System.IO.Path]::IsPathRooted($processorCheckout) -or -not (Test-Path -LiteralPath $processorCheckout -PathType Container)) {
        throw 'No se encontraron las ubicaciones privadas del preparador aprobado.'
    }
    $env:KNEEOAI_FROZEN_PYTHON = $processorPython
    $env:KNEEOAI_FROZEN_CHECKOUT = $processorCheckout
}
$logDirectory = Join-Path $backendDirectory 'runtime\logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$serverProcess = Start-Process -FilePath $pythonExecutable -ArgumentList @('-m','uvicorn','knee_app.api:create_app','--factory','--host','127.0.0.1','--port',"$Port") -WorkingDirectory $backendDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'servidor.out.log') -RedirectStandardError (Join-Path $logDirectory 'servidor.err.log') -PassThru
Write-Output "Servidor local iniciado en http://127.0.0.1:$Port (proceso $($serverProcess.Id))."
