@echo off
setlocal
cd /d "%~dp0"
title Agri-SOC Offline Leaf Assistant

echo ================================================
echo       Agri-SOC Offline Leaf Assistant
echo ================================================
echo.

REM Use an isolated Python 3.11 environment so existing system packages
REM cannot cause NumPy/Pandas/OpenCV binary conflicts.
where py >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python Launcher was not found.
    echo Install Python 3.11 from https://www.python.org/downloads/
    echo During installation, enable "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating an isolated Python 3.11 environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo ERROR: Python 3.11 was not found. Install Python 3.11 and try again.
        pause
        exit /b 1
    )
)

echo Installing/checking compatible packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Could not update pip.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo Starting Agri-SOC...
".venv\Scripts\python.exe" field_assistant.py
if errorlevel 1 (
    echo.
    echo The application stopped with an error. Please share the error text.
    pause
    exit /b 1
)

endlocal
