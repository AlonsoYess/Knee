@echo off
powershell.exe -NoProfile -File "%~dp0IniciarKneeOAI.ps1" %*
if errorlevel 1 pause
