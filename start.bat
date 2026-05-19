@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Executive Team Bridge v2.0.3
color 0A
chcp 437 >nul

cls
echo.
echo  ==========================================
echo       AI EXECUTIVE TEAM - STARTING UP
echo  ==========================================
echo.

REM ── Check config.yaml exists ──────────────
if not exist "config.yaml" (
    echo  [ERROR] No config.yaml found.
    echo  Run: python setup.py
    echo.
    pause
    exit /b 1
)

REM ── Check Python ──────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Install from https://python.org
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] !PYVER! found
echo.

REM ── Show current preset ───────────────────
echo  Current preset:
python -c "from core.config_loader import cfg; print('  ' + cfg.preset_name.upper() + ' - ' + cfg.preset_description)"
echo.

REM ── Preset selector ───────────────────────
echo  ==========================================
echo       SELECT MODE (10 seconds to choose)
echo  ==========================================
echo.
echo  [1] FASTEST    - Tiny GPU models only
echo                   30s-1min  GPT-3.5 level
echo.
echo  [2] FAST       - All small GPU models
echo                   2-4 min   GPT-4o Mini
echo.
echo  [3] BALANCED   - Best daily driver
echo                   4-7 min   GPT-4o + Claude + o1
echo.
echo  [4] SMART      - Max reasoning
echo                   8-15 min  Dual o1 Preview
echo.
echo  [5] NUCLEAR    - Everything maxed
echo                   20-40min  Frontier AI level
echo.
echo  [6] GEMMA 12B  - OG single model
echo                   2-4 min   GPT-4o Mini
echo.
echo  [ENTER] Keep current preset and start
echo.

del temp_selection.txt >nul 2>&1
python preset_selector.py

set PRESET_CHOICE=KEEP
for /f "tokens=2 delims=:" %%a in ('findstr "SELECTED" temp_selection.txt') do set PRESET_CHOICE=%%a
del temp_selection.txt >nul 2>&1
set PRESET_CHOICE=!PRESET_CHOICE: =!

echo.
if "!PRESET_CHOICE!"=="KEEP" (
    echo  Keeping current preset.
) else (
    echo  Switching to: !PRESET_CHOICE!
    python -c "from core.config_loader import cfg; cfg.switch_preset('!PRESET_CHOICE!')"
)
echo.

REM ── Show final active config ──────────────
echo  ==========================================
echo    Active Configuration:
echo  ==========================================
python -c "from core.config_loader import cfg; cfg.summary()"
echo.

REM ── Install packages ──────────────────────
echo  Checking packages...
python -c "import flask, flask_cors, dotenv, requests, yaml, rich, openai" >nul 2>&1
if errorlevel 1 (
    echo  Installing missing packages...
    pip install flask flask-cors python-dotenv requests pyyaml rich openai waitress -q
    echo  [OK] Packages installed.
) else (
    echo  [OK] Packages ready.
)
echo.

REM ── Check LM Studio ───────────────────────
echo  Checking LM Studio...
curl -s http://localhost:1234/v1/models >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] LM Studio is not running on port 1234.
    echo.
    echo  Please:
    echo    1. Open LM Studio
    echo    2. Go to Local Server tab
    echo    3. Click Start Server
    echo    4. Then run this script again
    echo.
    pause
    exit /b 1
)
echo  [OK] LM Studio is running.
echo.

REM ── Load primary model ────────────────────
echo  ==========================================
echo    Loading primary model...
echo  ==========================================
echo.
python -c "from core.config_loader import cfg; import subprocess, sys; subprocess.run([sys.executable, 'load_model.py', cfg.ceo_model])"
echo.

REM ── Kill anything on port 5555 ────────────
echo  ==========================================
echo    Clearing port 5555...
echo  ==========================================
echo.

set PORT=5555
set FOUND_PID=

for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    if not defined FOUND_PID set FOUND_PID=%%a
)

if not defined FOUND_PID (
    echo  [OK] Port %PORT% is already free
    echo.
    goto :start_bridge
)

echo  [!!] Found process PID !FOUND_PID! on port %PORT% -- killing it...
taskkill /PID !FOUND_PID! /F >nul 2>&1
taskkill /f /im python.exe /fi "WINDOWTITLE eq AI Executive Team Bridge" >nul 2>&1

echo.
echo  Waiting for port to clear...
echo.

set CLEARED=0
set ATTEMPTS=0

:port_wait_loop
    set /a ATTEMPTS+=1

    set "BAR="
    set /a FILLED=!ATTEMPTS!
    if !FILLED! gtr 10 set FILLED=10
    set /a EMPTY=10 - !FILLED!
    for /l %%f in (1,1,!FILLED!) do set "BAR=!BAR!#"
    for /l %%e in (1,1,!EMPTY!)  do set "BAR=!BAR!."

    netstat -aon 2>nul | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
    if errorlevel 1 (
        set CLEARED=1
        echo  [##########] Port %PORT% is clear!
        echo.
        goto :port_done
    )

    echo  [!BAR!] Attempt !ATTEMPTS!/10 -- port still held...
    timeout /t 1 /nobreak >nul

    if !ATTEMPTS! lss 10 goto :port_wait_loop

:port_done
if !CLEARED! == 0 (
    echo  [WARNING] Port %PORT% did not fully clear after 10s
    echo            Starting anyway -- old process may still be exiting
    echo.
)

REM ── Start bridge ──────────────────────────
:start_bridge
echo  ==========================================
echo    Starting bridge on http://localhost:5555
echo  ==========================================
echo.
echo  Health:    http://localhost:5555/health
echo  Board:     http://localhost:5555/v1/board
echo  OpenWebUI: http://localhost:3000
echo.
echo  ------------------------------------------
echo   HOW TO STOP THE BRIDGE:
echo     Press Ctrl+C  then Y  then Enter
echo.
echo   HOW TO ABORT CURRENT GENERATION:
echo     Press X in this window anytime
echo.
echo   HOW TO SWITCH PRESET WHILE RUNNING:
echo     Press SPACE in this window anytime
echo.
echo   TOKEN STATS appear here after each reply
echo  ------------------------------------------
echo.
echo  ==========================================
echo    BRIDGE IS LIVE -- READY FOR REQUESTS
echo  ==========================================
echo.
echo  Send a message from any of these clients:
echo.
echo    [OpenWebUI]
echo      URL:   http://localhost:3000
echo      Step:  Pick "Executive Swarm" in model dropdown
echo             Type your message and press Send
echo.
echo    [VS Code - Continue extension]
echo      Step:  Press Ctrl+L to open chat
echo             Pick any model from the dropdown
echo             Type your question and press Enter
echo.
echo    [AnythingLLM]
echo      URL:   http://localhost:5555/v1
echo      Model: executive-swarm
echo.
echo    [cURL test]
echo      CMD:   curl http://localhost:5555/health
echo.
echo  Waiting for first request...
echo.
echo  ------------------------------------------
echo.

python swarm_bridge_server.py

REM ── After bridge exits ────────────────────
echo.
echo  ==========================================
echo    Bridge stopped.
echo  ==========================================
echo.
pause