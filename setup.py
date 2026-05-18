# setup.py
# Run once: python setup.py
# Sets up folders packages and checks everything

import os
import sys
import subprocess
import requests
from pathlib import Path


def banner():
    print("""
╔══════════════════════════════════════════════════════╗
║         AI EXECUTIVE TEAM — LOCAL SETUP              ║
║                                                      ║
║  CEO  → Gemma 4 31B        (GPT-4o)                 ║
║  CTO  → Qwen3 Coder 30B    (Claude Sonnet Coding)   ║
║  CFO  → DeepSeek R1 32B    (OpenAI o1 Preview)      ║
║  CPO  → Gemma 4 26B A4B    (Claude Sonnet)          ║
║  COO  → Qwen2.5 Coder 14B  (GitHub Copilot Pro)     ║
║  VIS  → Qwen2.5 VL 7B      (GPT-4o Vision)          ║
║                                                      ║
║  Config: config.yaml                                 ║
║  Presets: presets/                                   ║
╚══════════════════════════════════════════════════════╝
""")


def create_folders():
    folders = [
        "agents",
        "routers",
        "core",
        "config",
        "logs",
        "presets",
    ]
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
        if folder in ["agents", "routers", "core"]:
            init = Path(folder) / "__init__.py"
            if not init.exists():
                init.touch()
    print("✓ Folders ready")


def check_config():
    print("\nChecking config files...")

    if Path("config.yaml").exists():
        print("  ✓ config.yaml found")
    else:
        print("  ✗ config.yaml MISSING")
        print("    Create config.yaml in your root folder")

    presets = [
        "presets/balanced.yaml",
        "presets/fast.yaml",
        "presets/smart.yaml",
        "presets/nuclear.yaml",
        "presets/gemma12b.yaml",
    ]
    for preset in presets:
        if Path(preset).exists():
            print(f"  ✓ {preset}")
        else:
            print(f"  ✗ MISSING: {preset}")


def install_packages():
    print("\nInstalling packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-r", "requirements.txt", "-q"
    ])
    print("✓ Packages installed")


def check_lm_studio():
    try:
        r = requests.get(
            "http://localhost:1234/v1/models", timeout=3
        )
        if r.status_code == 200:
            models = r.json().get("data", [])
            print(f"\n✓ LM Studio running")
            if models:
                print("  Loaded models:")
                for m in models:
                    print(f"    - {m['id']}")
            else:
                print("  ⚠ No model loaded yet")
                print(
                    "  → Load a model in LM Studio "
                    "then click Start Server"
                )
            return True
    except Exception:
        pass
    print("\n✗ LM Studio not detected on port 1234")
    print("  → Open LM Studio → Local Server → Start Server")
    return False


def check_config_loader():
    print("\nChecking config loader...")
    try:
        from core.config_loader import cfg
        print(f"  ✓ Config loaded: preset = {cfg.preset_name}")
        print(f"  ✓ CEO model: {cfg.ceo_model}")
        return True
    except Exception as e:
        print(f"  ✗ Config loader error: {e}")
        return False


def print_next_steps(lm_up):
    print("""
╔══════════════════════════════════════════════════════╗
║                    NEXT STEPS                        ║
╚══════════════════════════════════════════════════════╝
""")
    if not lm_up:
        print("  1. Open LM Studio")
        print("  2. Go to Local Server tab")
        print("  3. Load any model")
        print("  4. Click Start Server")
        print("  5. Double click start.bat\n")
    else:
        print("  1. Double click start.bat")
        print("     Select a preset or wait 10s for current\n")
        print("  2. Or switch preset anytime:")
        print("     tools/swap.bat\n")
        print("  3. Test the pipeline:")
        print("     python debug_stream.py\n")
        print("  4. Configure LM Studio settings:")
        print("     python load_model.py\n")
        print("  5. Check config:")
        print("     python core/config_loader.py summary\n")
        print("  6. Check health:")
        print("     http://localhost:5555/health\n")
        print("  7. Open VS Code → Continue extension → Ctrl+L\n")
        print("  8. Open OpenWebUI:")
        print("     http://localhost:3000\n")
        print("  9. Analyze an image:")
        print(
            "     python agents/vision.py "
            "screenshot.png 'What do you see?'\n"
        )


def main():
    banner()
    create_folders()
    install_packages()
    check_config()
    lm_up  = check_lm_studio()
    cfg_ok = check_config_loader()
    print_next_steps(lm_up)
    print("✓ Setup complete\n")


if __name__ == "__main__":
    main()
