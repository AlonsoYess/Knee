@echo off
powershell.exe -NoProfile -File "%~dp0DetenerKneeOAI.ps1" %*
if errorlevel 1 pause
