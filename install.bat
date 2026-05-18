@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title AI Executive Team - First Time Setup
color 0A
chcp 437 >nul

cls
echo.
echo  ==========================================
echo       AI EXECUTIVE TEAM - FIRST TIME SETUP
echo  ==========================================
echo.
echo  This will:
echo    1. Check Python version
echo    2. Install required packages
echo    3. Create config.yaml if missing
echo    4. Verify your model overrides file
echo    5. Test the config loader
echo.
echo  Run once. After this, just use start.bat
echo.
pause

REM ── Check Python ──────────────────────────
echo.
echo  [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Python not found in PATH.
    echo.
    echo  Install Python 3.10+ from https://python.org
    echo  Make sure to check "Add to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] !PYVER!

REM ── Check Python version is 3.10+ ─────────
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set FULLVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!FULLVER!") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if !MAJOR! LSS 3 (
    echo  [ERROR] Need Python 3.10 or higher. Found: !FULLVER!
    pause
    exit /b 1
)
if !MAJOR! EQU 3 if !MINOR! LSS 10 (
    echo  [WARNING] Python 3.10+ recommended. Found: !FULLVER!
    echo            Continuing anyway — may work with 3.8+
)

REM ── Install packages ──────────────────────
echo.
echo  [2/5] Installing packages...
echo.
pip install flask flask-cors python-dotenv requests pyyaml rich openai waitress keyboard
if errorlevel 1 (
    echo.
    echo  [WARNING] Some packages may have failed to install.
    echo            Check the output above.
    echo.
) else (
    echo.
    echo  [OK] All packages installed.
)

REM ── Create config.yaml if missing ─────────
echo.
echo  [3/5] Checking config.yaml...
if exist "config.yaml" (
    echo  [OK] config.yaml already exists.
) else (
    echo  Creating config.yaml from default...
    python setup.py
    if errorlevel 1 (
        echo  [ERROR] setup.py failed.
        echo          Run: python setup.py manually.
        pause
        exit /b 1
    )
    echo  [OK] config.yaml created.
)

REM ── Check my_models.yaml ──────────────────
echo.
echo  [4/5] Checking model override file...
if exist "config\my_models.yaml" (
    echo  [OK] config\my_models.yaml exists.
    echo       Edit it to swap models across all presets.
) else (
    echo  [WARNING] config\my_models.yaml not found.
    echo            This is normal — it will be created on first run.
)

REM ── Test config loader ────────────────────
echo.
echo  [5/5] Testing config loader...
echo.
python -c "from core.config_loader import cfg; cfg.summary()" 2>&1
if errorlevel 1 (
    echo.
    echo  [ERROR] Config loader failed.
    echo          Check config.yaml and preset files.
    pause
    exit /b 1
)

REM ── Done ──────────────────────────────────
echo.
echo  ==========================================
echo    Setup complete!
echo  ==========================================
echo.
echo  Next steps:
echo.
echo    1. Open LM Studio
echo       - Go to Local Server tab
echo       - Click Start Server (port 1234)
echo.
echo    2. Download at least one model in LM Studio
echo       - Recommended: google/gemma-3-12b (8.2GB)
echo         for the gemma12b preset (most reliable)
echo.
echo    3. Run start.bat to launch the bridge
echo.
echo    4. Optional: Install OpenWebUI (Docker required)
echo       Then run restart_openwebui.bat
echo       Add bridge as Ollama: http://localhost:5555
echo.
echo    5. Optional: Edit config\my_models.yaml
echo       to swap models without touching preset files
echo.
echo  Full guide: README.md
echo.
pause
