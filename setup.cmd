@echo off
setlocal
cd /d "%~dp0"
set "torch_index=https://download.pytorch.org/whl/cu128"
if /i "%~1"=="cpu" set "torch_index=https://download.pytorch.org/whl/cpu"
if not "%~1"=="" if /i not "%~1"=="cpu" (
    echo Usage: setup.cmd [cpu]
    exit /b 2
)
set "project_python=.venv\Scripts\python.exe"
if not exist "%project_python%" (
    python -m venv .venv || exit /b 1
)
"%project_python%" -m pip install torch==2.10.0 --index-url "%torch_index%" || exit /b 1
"%project_python%" -m pip install -r requirements.txt || exit /b 1
"%project_python%" -m pip check || exit /b 1
"%project_python%" mtg.py doctor
exit /b %errorlevel%
