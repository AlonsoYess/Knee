param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'El puerto debe estar entre 1024 y 65535.' }
$backendDirectory = Join-Path $PSScriptRoot 'backend'
$pythonExecutable = Join-Path $backendDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExecutable)) { throw 'Falta instalar el entorno del backend. Consulta app/README.md.' }
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) { Write-Output "El puerto $Port ya esta en uso. No se inicio otro servidor."; exit 0 }
$logDirectory = Join-Path $backendDirectory 'runtime\logs'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$serverProcess = Start-Process -FilePath $pythonExecutable -ArgumentList @('-m','uvicorn','knee_app.api:create_app','--factory','--host','127.0.0.1','--port',"$Port") -WorkingDirectory $backendDirectory -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDirectory 'servidor.out.log') -RedirectStandardError (Join-Path $logDirectory 'servidor.err.log') -PassThru
Write-Output "Servidor local iniciado en http://127.0.0.1:$Port (proceso $($serverProcess.Id))."
