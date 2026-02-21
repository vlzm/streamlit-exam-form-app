@echo off
chcp 65001 >nul 2>&1
title Building Portable Exam Grader

set PY_MAJOR=3
set PY_MINOR=12
set PY_PATCH=9
set PY_VER=%PY_MAJOR%.%PY_MINOR%.%PY_PATCH%
set PY_VER_SHORT=%PY_MAJOR%%PY_MINOR%
set PYTHON_ZIP=python-%PY_VER%-embed-amd64.zip
set PYTHON_URL=https://www.python.org/ftp/python/%PY_VER%/%PYTHON_ZIP%
set PYTHON_DIR=python
set PTH_FILE=%PYTHON_DIR%\python%PY_VER_SHORT%._pth
set GET_PIP_URL=https://bootstrap.pypa.io/get-pip.py

echo ============================================
echo   Building Portable Exam Grader
echo   Python %PY_VER% (64-bit)
echo ============================================
echo.

if exist "%PYTHON_DIR%\python.exe" (
    echo [INFO] Portable Python already exists in %PYTHON_DIR%\
    echo        Delete the "%PYTHON_DIR%" folder to rebuild.
    echo.
    pause
    exit /b 0
)

:: ── Step 1: Download Python embeddable ────────
echo [1/5] Downloading Python %PY_VER% embeddable package...
curl -L -o "%PYTHON_ZIP%" "%PYTHON_URL%"
if errorlevel 1 (
    echo [ERROR] Failed to download Python. Check your internet connection.
    pause
    exit /b 1
)

:: ── Step 2: Extract ───────────────────────────
echo [2/5] Extracting Python...
mkdir "%PYTHON_DIR%" 2>nul
tar -xf "%PYTHON_ZIP%" -C "%PYTHON_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to extract Python archive.
    pause
    exit /b 1
)
del "%PYTHON_ZIP%"
echo [OK] Python extracted to %PYTHON_DIR%\

:: ── Step 3: Enable pip and site-packages ──────
echo [3/5] Configuring Python for pip...

(
    echo python%PY_VER_SHORT%.zip
    echo .
    echo import site
) > "%PTH_FILE%"

curl -sL -o "%PYTHON_DIR%\get-pip.py" "%GET_PIP_URL%"
"%PYTHON_DIR%\python.exe" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to install pip.
    pause
    exit /b 1
)
del "%PYTHON_DIR%\get-pip.py"
echo [OK] pip installed.

:: ── Step 4: Install dependencies ──────────────
echo [4/5] Installing dependencies (this will take several minutes)...
echo.
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

:: ── Step 5: Install the project package ───────
echo.
echo [5/5] Installing exam_grader package...
"%PYTHON_DIR%\python.exe" -m pip install --no-warn-script-location --no-deps -e .
if errorlevel 1 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build complete!
echo.
echo   Total size of python\ folder:
for /f "tokens=3" %%a in ('dir "%PYTHON_DIR%" /s /-c ^| findstr "File(s)"') do echo     %%a bytes
echo.
echo   To run the app, double-click start.bat
echo ============================================
echo.
pause
