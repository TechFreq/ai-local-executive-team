@echo off
cd /d "%~dp0"
title AI Executive Team — Status Check
color 0B

echo.
echo  ==========================================
echo    AI EXECUTIVE TEAM — STATUS CHECK
echo  ==========================================
echo.

REM ── Check LM Studio ──
echo  Checking LM Studio (port 1234)...
curl -s http://localhost:1234/v1/models >nul 2>&1
if errorlevel 1 (
    echo  [X] LM Studio NOT running on port 1234
) else (
    echo  [OK] LM Studio is running on port 1234
)

echo.

REM ── Check Bridge ──
echo  Checking Bridge (port 5555)...
curl -s http://localhost:5555/health >nul 2>&1
if errorlevel 1 (
    echo  [X] Bridge NOT running on port 5555
) else (
    echo  [OK] Bridge is running on port 5555
)

echo.

REM ── Show what's on port 5555 ──
echo  Processes on port 5555:
netstat -ano | findstr ":5555" | findstr "LISTENING"
if errorlevel 1 (
    echo  None
)

echo.

REM ── Show what's on port 1234 ──
echo  Processes on port 1234:
netstat -ano | findstr ":1234" | findstr "LISTENING"
if errorlevel 1 (
    echo  None
)

echo.
echo  ==========================================
pause