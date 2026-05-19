@echo off
title AI Executive Team - Mode Switcher
color 0A
chcp 437 >nul

echo.
echo  ==========================================
echo       AI EXECUTIVE TEAM - MODE SWITCHER
echo  ==========================================
echo.

REM Show current preset
echo  Current active preset:
python -c "from core.config_loader import cfg; print('  ' + cfg.preset_name.upper() + ' - ' + cfg.preset_description)"
echo.

echo  Pick your mode:
echo.
echo  [1] FASTEST    - Tiny GPU-only models
echo                   VRAM: ~4-6GB  Time: 30s-1min
echo                   Like: GPT-3.5 — quick questions
echo.
echo  [2] FAST       - All agents in GPU VRAM
echo                   VRAM: ~10GB  Time: 2-4 min
echo                   Like: GPT-4o Mini
echo.
echo  [3] BALANCED   - Sweet spot daily driver
echo                   VRAM: mixed  Time: 4-7 min
echo                   Like: GPT-4o + Claude + o1
echo.
echo  [4] SMART      - Best quality RAM offload
echo                   VRAM: mixed  Time: 8-15 min
echo                   Like: Dual o1 Preview reasoning
echo.
echo  [5] NUCLEAR    - Everything maxed, heavy offload
echo                   VRAM: mixed  Time: 20-40 min
echo                   Like: Full frontier AI board
echo.
echo  [6] GEMMA 12B  - OG mode 1 model all roles
echo                   VRAM: 8.2GB  Time: 3-6 min
echo                   Most stable — reliable fallback
echo.
echo  [7] CUSTOM     - Edit config.yaml directly
echo.
echo  [8] List all presets
echo.
echo  [9] Sync models  - scan LM Studio for new/removed models
echo.
set /p choice="  Enter 1-9: "

if "%choice%"=="1" set PRESET=fastest
if "%choice%"=="2" set PRESET=fast
if "%choice%"=="3" set PRESET=balanced
if "%choice%"=="4" set PRESET=smart
if "%choice%"=="5" set PRESET=nuclear
if "%choice%"=="6" set PRESET=gemma12b
if "%choice%"=="7" goto custom
if "%choice%"=="8" goto list_presets
if "%choice%"=="9" goto sync_models
if "%PRESET%"=="" goto invalid

goto switch_preset

:custom
echo.
echo  Opening config.yaml in notepad...
echo  Change the active_preset line to switch modes.
echo  Or edit the custom board section directly.
echo.
notepad config.yaml
goto end

:list_presets
echo.
python -c "from core.config_loader import cfg; cfg.list_presets()"
echo.
pause
goto end

:switch_preset
echo.
echo  ==========================================
echo    Switching to: %PRESET%
echo  ==========================================
echo.
python -c "from core.config_loader import cfg; cfg.switch_preset('%PRESET%'); cfg.summary()"

echo.
echo  ==========================================
echo    Restarting bridge server...
echo  ==========================================
echo.
taskkill /f /im python.exe /fi "WINDOWTITLE eq AI Executive Team Bridge" >nul 2>&1
start "AI Executive Team Bridge" python swarm_bridge_server.py

echo.
echo  ==========================================
echo    Done. Now running in %PRESET% mode.
echo.
echo    Bridge:  http://localhost:5555
echo    Health:  http://localhost:5555/health
echo    OpenWeb: http://localhost:3000
echo  ==========================================
echo.
pause
goto end

:sync_models
echo.
echo  ==========================================
echo    Syncing models with LM Studio...
echo  ==========================================
echo.
echo  Make sure LM Studio is open and its server is running.
echo.
cd /d "%~dp0.."
python sync_models.py --apply
echo.
echo  ==========================================
echo    Done. Check config/my_models.yaml to
echo    activate any newly found model in a role.
echo  ==========================================
echo.
pause
goto end

:invalid
echo.
echo    Invalid choice. Run again and pick 1-9.
echo.
pause

:end