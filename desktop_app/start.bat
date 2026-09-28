@echo off
setlocal
cd /d "%~dp0.."

if not exist ".training-env\python.exe" (
  echo Ambiente .training-env nao encontrado.
  echo Consulte desktop_app\README.md para criar o ambiente.
  pause
  exit /b 1
)

".training-env\python.exe" "desktop_app\main.py"
if errorlevel 1 pause
