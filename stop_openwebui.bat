@echo off
title Stop OpenWebUI
color 0C
cd /d "%~dp0"

echo.
echo  ==========================================
echo       STOP OPENWEBUI
echo  ==========================================
echo.

docker ps --filter "name=open-webui" --format "{{.Names}}" 2>nul | findstr "open-webui" >nul 2>&1
if errorlevel 1 (
    echo  [INFO] OpenWebUI container is not running.
    echo.
    pause
    exit /b 0
)

echo  Stopping OpenWebUI container...
docker stop open-webui >nul 2>&1

if errorlevel 1 (
    echo  [ERROR] Could not stop container.
    echo          Is Docker running?
    echo.
    pause
    exit /b 1
)

echo  [OK] OpenWebUI stopped.
echo.
pause
