@echo off
chcp 65001 >nul 2>&1
title Exam Grader

set VENV_DIR=.venv
set REQUIREMENTS=requirements.txt

echo ============================================
echo         Exam Grader — Launcher
echo ============================================
echo.

:: ── Detect Python ─────────────────────────────
:: Prefer portable Python (python\ folder) if available
if exist "python\python.exe" (
    set PYTHON=python\python.exe
    set MODE=portable
    echo [OK] Using portable Python
    for /f "tokens=*" %%i in ('"python\python.exe" --version 2^>^&1') do echo     %%i
    goto :launch
)

:: Fall back to system Python
set PYTHON=python
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found on this computer.
    echo.
    echo Option 1: Install Python 3.10 or newer:
    echo   https://www.python.org/downloads/
    echo   Check "Add Python to PATH" during installation.
    echo.
    echo Option 2: Run build_portable.bat to create a
    echo   self-contained portable Python setup.
    echo.
    pause
    exit /b 1
)
set MODE=system
for /f "tokens=*" %%i in ('%PYTHON% --version 2^>^&1') do echo [OK] Found %%i

:: ── Create virtual environment if needed ──────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo [SETUP] Creating virtual environment...
    %PYTHON% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.

    echo.
    echo [SETUP] Installing dependencies...
    echo         This may take several minutes on the first run.
    echo.
    call "%VENV_DIR%\Scripts\activate.bat"
    pip install --upgrade pip >nul 2>&1
    pip install -r %REQUIREMENTS%
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    pip install -e . >nul 2>&1
    echo.
    echo [OK] All dependencies installed.
) else (
    call "%VENV_DIR%\Scripts\activate.bat"
)

:: ── Launch ────────────────────────────────────
:launch
echo.
echo ============================================
echo   Starting Exam Grader...
echo   The app will open in your browser.
echo.
echo   If it doesn't open automatically, go to:
echo     http://localhost:8501
echo.
echo   To stop the app, close this window
echo   or press Ctrl+C.
echo ============================================
echo.

if "%MODE%"=="portable" (
    "%PYTHON%" -m streamlit run app.py --server.headless false
) else (
    streamlit run app.py --server.headless false
)
