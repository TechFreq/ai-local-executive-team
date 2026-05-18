@echo off
cd /d "%~dp0"
title AI Executive Team — Restart Bridge
color 0C

echo.
echo  ==========================================
echo    RESTARTING BRIDGE
echo  ==========================================
echo.

REM ── Kill anything on port 5555 ──
echo  Checking port 5555...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5555" ^| findstr "LISTENING"') do (
    echo  Killing process %%a on port 5555...
    taskkill /f /pid %%a >nul 2>&1
)

REM ── Kill any stray Python bridge processes ──
echo  Killing any stray Python processes...
taskkill /f /im python.exe /fi "WINDOWTITLE eq AI Executive Team Bridge" >nul 2>&1

REM ── Wait a moment ──
timeout /t 2 /nobreak >nul

REM ── Verify port is clear ──
netstat -ano | findstr ":5555" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  WARNING: Port 5555 still in use!
    echo  Try running as Administrator.
    echo.
    pause
    exit
)

echo  Port 5555 is clear.
echo.

REM ── Start fresh ──
echo  ==========================================
echo    Starting fresh bridge...
echo  ==========================================
echo.
python swarm_bridge_server.py
pause