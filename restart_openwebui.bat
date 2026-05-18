@echo off
title Restart OpenWebUI
color 0A
cd /d "%~dp0"

echo.
echo  ==========================================
echo       RESTART OPENWEBUI
echo  ==========================================
echo.

REM ── Check Docker is running ───────────────
docker info >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Docker is not running.
    echo          Start Docker Desktop first, then run this again.
    echo.
    pause
    exit /b 1
)

REM ── Stop existing container if running ────
docker ps --filter "name=open-webui" --format "{{.Names}}" 2>nul | findstr "open-webui" >nul 2>&1
if not errorlevel 1 (
    echo  Stopping existing OpenWebUI container...
    docker stop open-webui >nul 2>&1
    docker rm open-webui >nul 2>&1
    echo  [OK] Stopped.
    echo.
)

REM ── Start fresh ───────────────────────────
echo  Starting OpenWebUI...
echo  (This may take a moment on first launch)
echo.

docker run -d ^
  --name open-webui ^
  --restart always ^
  -p 3000:8080 ^
  -v open-webui:/app/backend/data ^
  --add-host=host.docker.internal:host-gateway ^
  ghcr.io/open-webui/open-webui:main >nul 2>&1

if errorlevel 1 (
    echo  [ERROR] Failed to start OpenWebUI.
    echo          Check: docker logs open-webui
    echo.
    pause
    exit /b 1
)

echo  [OK] OpenWebUI started.
echo.
echo  ==========================================
echo    OpenWebUI is running at:
echo    http://localhost:3000
echo.
echo    Add the bridge as Ollama connection:
echo    Settings > Connections > Ollama
echo    URL: http://host.docker.internal:5555
echo  ==========================================
echo.
pause
