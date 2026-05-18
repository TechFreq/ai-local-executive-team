# core/config_loader.py
# ══════════════════════════════════════════════════════
# CENTRAL CONFIG LOADER
# Reads config.yaml from root directory
# Loads the active preset automatically
# Every script imports this one file
#
# Usage:
#   from core.config_loader import cfg
#
#   cfg.ceo_model        → active CEO model
#   cfg.bridge_port      → server port
#   cfg.preset_name      → which preset is active
#
# CLI:
#   python core/config_loader.py summary
#   python core/config_loader.py presets
#   python core/config_loader.py switch balanced
#   python core/config_loader.py test
# ══════════════════════════════════════════════════════

import os
import sys
import yaml
from pathlib import Path
from rich.console import Console

console = Console()

ROOT_DIR        = Path(__file__).parent.parent
CONFIG_FILE     = ROOT_DIR / "config.yaml"
PRESET_DIR      = ROOT_DIR / "presets"
MY_MODELS_FILE  = ROOT_DIR / "config" / "my_models.yaml"


class Config:
    """
    Loads config.yaml and the active preset.
    Exposes everything as simple dot notation.
    """

    def __init__(self):
        self._raw        = {}
        self._preset     = {}
        self.preset_name = "unknown"
        self._load()

    def _load(self):
        if not CONFIG_FILE.exists():
            console.print(
                f"[red]✗ config.yaml not found at {CONFIG_FILE}[/red]"
            )
            console.print(
                "[yellow]  Run: python setup.py[/yellow]"
            )
            self._use_defaults()
            self._cache_values()
            return

        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                self._raw = yaml.safe_load(f) or {}
        except Exception as e:
            console.print(f"[red]✗ Could not read config.yaml: {e}[/red]")
            self._use_defaults()
            self._cache_values()
            return

        active           = self._raw.get("active_preset", "balanced")
        self.preset_name = active

        if active == "custom":
            self._preset = self._raw.get("custom", {})
            console.print(
                "[cyan]Config:[/cyan] Using custom board settings"
            )
        else:
            preset_file = PRESET_DIR / f"{active}.yaml"
            if not preset_file.exists():
                console.print(
                    f"[yellow]⚠ Preset '{active}' not found[/yellow]"
                )
                self._use_defaults()
                self._cache_values()
                return

            try:
                with open(preset_file, "r", encoding="utf-8") as f:
                    self._preset = yaml.safe_load(f) or {}
                console.print(
                    f"[cyan]Config:[/cyan] Preset "
                    f"[bold]{active}[/bold] — "
                    f"{self._preset.get('description', '')}"
                )
            except Exception as e:
                console.print(
                    f"[red]✗ Could not load preset {active}: {e}[/red]"
                )
                self._use_defaults()

        self._cache_values()

    def _cache_values(self):
        """Flatten all config values into plain instance attributes.
        Eliminates repeated double-dict lookups on every property access."""
        # Board models
        self.ceo_model        = self._get_board("ceo",        "google/gemma-4-31b")
        self.cto_model        = self._get_board("cto",        "qwen/qwen3-coder-30b")
        self.cfo_model        = self._get_board("cfo",        "deepseek/deepseek-r1-distill-qwen-32b")
        self.cfo_backup_model = self._get_board("cfo_backup", "qwen/qwq-32b")
        self.cpo_model        = self._get_board("cpo",        "google/gemma-4-26b-a4b")
        self.coo_model        = self._get_board("coo",        "qwen/qwen2.5-coder-14b-instruct")
        # Utility models
        self.autocomplete_model = self._get_utility("autocomplete", "google/gemma-4-e2b")
        self.embed_model        = self._get_utility("embed",        "nomic-ai/nomic-embed-text-v1.5")
        self.vision_model       = self._get_utility("vision",       "qwen/qwen2.5-vl-7b-instruct")
        # Fallback models
        self.fallback_model          = self._get_fallback("primary",        "microsoft/phi-4-reasoning-plus")
        self.fallback_general        = self._get_fallback("general",        "qwen/qwen3.5-9b")
        self.fallback_fast_reasoning = self._get_fallback("fast_reasoning", "deepseek/deepseek-r1-0528-qwen3-8b")
        self.fallback_last_resort    = self._get_fallback("last_resort",    "google/gemma-3-12b")
        # Server settings
        _srv = self._raw.get("server", {})
        self.bridge_port    = _srv.get("bridge_port",    5555)
        self.bridge_host    = _srv.get("bridge_host",    "localhost")
        self.lm_studio_url  = _srv.get("lm_studio_url",  "http://localhost:1234/v1")
        self.ollama_url     = _srv.get("ollama_url",     "http://localhost:11434/v1")
        self.backend        = _srv.get("backend",        "lmstudio")
        self.use_cloud      = _srv.get("use_cloud",      False)
        # Timeout settings
        _to = self._raw.get("timeouts", {})
        self.first_token_secs  = _to.get("first_token_secs",  120)
        self.hard_timeout_secs = _to.get("hard_timeout_secs", 600)
        self.keepalive_interval = _to.get("keepalive_interval", 5)
        self.load_timeout_secs = _to.get("load_timeout_secs", 480)
        # Generation settings
        _gen = self._raw.get("generation", {})
        self.ceo_temperature     = _gen.get("ceo_temperature",     0.7)
        self.cto_temperature     = _gen.get("cto_temperature",     0.4)
        self.cfo_temperature     = _gen.get("cfo_temperature",     0.3)
        self.cpo_temperature     = _gen.get("cpo_temperature",     0.8)
        self.coo_temperature     = _gen.get("coo_temperature",     0.5)
        self.default_max_tokens  = _gen.get("default_max_tokens",  2000)
        # Logging settings
        _log = self._raw.get("logging", {})
        self.track_performance = _log.get("track_performance", True)
        self.performance_file  = _log.get("performance_file", "model_performance.json")
        # Preset display info
        self.preset_display      = self._preset.get("display", {})
        self.preset_description  = self._preset.get("description", "")
        # Apply user overrides from config/my_models.yaml (runs last — always wins)
        self._apply_my_models()

    def _use_defaults(self):
        self.preset_name = "defaults"
        self._preset = {
            "board": {
                "ceo":        "google/gemma-4-31b",
                "cto":        "qwen/qwen3-coder-30b",
                "cfo":        "deepseek/deepseek-r1-distill-qwen-32b",
                "cfo_backup": "qwen/qwq-32b",
                "cpo":        "google/gemma-4-26b-a4b",
                "coo":        "qwen/qwen2.5-coder-14b-instruct",
            },
            "utility": {
                "autocomplete": "google/gemma-4-e2b",
                "embed":        "nomic-ai/nomic-embed-text-v1.5",
                "vision":       "qwen/qwen2.5-vl-7b-instruct",
            },
            "fallback": {
                "primary":        "microsoft/phi-4-reasoning-plus",
                "general":        "qwen/qwen3.5-9b",
                "fast_reasoning": "deepseek/deepseek-r1-0528-qwen3-8b",
                "last_resort":    "google/gemma-3-12b",
            }
        }

    def _apply_my_models(self):
        """Apply model overrides from config/my_models.yaml.
        Any uncommented model line there wins over the active preset.
        Called last in _cache_values() so it always takes priority."""
        if not MY_MODELS_FILE.exists():
            return
        try:
            with open(MY_MODELS_FILE, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f) or {}
        except Exception as e:
            console.print(f"[yellow]⚠ Could not read my_models.yaml: {e}[/yellow]")
            return

        board = overrides.get("board") or {}
        if board.get("ceo"):    self.ceo_model    = board["ceo"]
        if board.get("cto"):    self.cto_model    = board["cto"]
        if board.get("cfo"):    self.cfo_model    = board["cfo"]
        if board.get("cfo_backup"): self.cfo_backup_model = board["cfo_backup"]
        if board.get("cpo"):    self.cpo_model    = board["cpo"]
        if board.get("coo"):    self.coo_model    = board["coo"]

        util = overrides.get("utility") or {}
        if util.get("autocomplete"): self.autocomplete_model = util["autocomplete"]
        if util.get("vision"):       self.vision_model       = util["vision"]
        if util.get("embed"):        self.embed_model        = util["embed"]

        fb = overrides.get("fallback") or {}
        if fb.get("primary"):        self.fallback_model          = fb["primary"]
        if fb.get("general"):        self.fallback_general        = fb["general"]
        if fb.get("fast_reasoning"): self.fallback_fast_reasoning = fb["fast_reasoning"]
        if fb.get("last_resort"):    self.fallback_last_resort    = fb["last_resort"]

        # Count active overrides to report (skip if none)
        active = sum(
            1 for d in (board, util, fb)
            for v in (d or {}).values() if v
        )
        if active:
            console.print(
                f"[cyan]Config:[/cyan] "
                f"[bold]{active}[/bold] model override(s) applied from "
                f"[dim]config/my_models.yaml[/dim]"
            )

    # ── Internal helpers ──────────────────────────────

    def _get_board(self, key: str, default: str) -> str:
        return self._preset.get("board", {}).get(key, default)

    def _get_utility(self, key: str, default: str) -> str:
        return self._preset.get("utility", {}).get(key, default)

    def _get_fallback(self, key: str, default: str) -> str:
        return self._preset.get("fallback", {}).get(key, default)

    # ── Public methods ────────────────────────────────

    def summary(self):
        display = self.preset_display
        console.print()
        console.print(f"[bold cyan]{'═' * 56}[/bold cyan]")
        console.print(
            f"[bold cyan]  ACTIVE PRESET :[/bold cyan] "
            f"[bold]{self.preset_name.upper()}[/bold]"
        )
        console.print(
            f"[bold cyan]  Description   :[/bold cyan] "
            f"{self.preset_description}"
        )
        if display:
            console.print(
                f"[bold cyan]  VRAM          :[/bold cyan] "
                f"{display.get('vram', 'N/A')}"
            )
            console.print(
                f"[bold cyan]  Meeting Time  :[/bold cyan] "
                f"{display.get('meeting_time', 'N/A')}"
            )
            console.print(
                f"[bold cyan]  Quality       :[/bold cyan] "
                f"{display.get('quality', 'N/A')}"
            )
        console.print(f"[bold cyan]{'─' * 56}[/bold cyan]")
        console.print(f"[bold cyan]  BOARD OF DIRECTORS[/bold cyan]")
        console.print(f"  CEO  → {self.ceo_model}")
        console.print(f"  CTO  → {self.cto_model}")
        console.print(f"  CFO  → {self.cfo_model}")
        console.print(f"  CPO  → {self.cpo_model}")
        console.print(f"  COO  → {self.coo_model}")
        console.print(f"[bold cyan]{'─' * 56}[/bold cyan]")
        console.print(f"[bold cyan]  UTILITY[/bold cyan]")
        console.print(f"  Vision → {self.vision_model}")
        console.print(f"  Auto   → {self.autocomplete_model}")
        console.print(f"  Embed  → {self.embed_model}")
        console.print(f"[bold cyan]{'─' * 56}[/bold cyan]")
        console.print(f"[bold cyan]  FALLBACKS[/bold cyan]")
        console.print(f"  Primary      → {self.fallback_model}")
        console.print(f"  General      → {self.fallback_general}")
        console.print(f"  Fast Reason  → {self.fallback_fast_reasoning}")
        console.print(f"  Last Resort  → {self.fallback_last_resort}")
        console.print(f"[bold cyan]{'─' * 56}[/bold cyan]")
        console.print(f"[bold cyan]  SERVER[/bold cyan]")
        console.print(
            f"  Bridge    → "
            f"http://{self.bridge_host}:{self.bridge_port}"
        )
        console.print(f"  LM Studio → {self.lm_studio_url}")
        console.print(f"  Backend   → {self.backend}")
        console.print(f"[bold cyan]{'═' * 56}[/bold cyan]")
        console.print()

    def switch_preset(self, preset_name: str) -> bool:
        preset_file = PRESET_DIR / f"{preset_name}.yaml"
        if not preset_file.exists():
            console.print(
                f"[red]✗ Preset '{preset_name}' not found[/red]"
            )
            available = [p.stem for p in PRESET_DIR.glob("*.yaml")]
            console.print(
                f"[yellow]  Available: {available}[/yellow]"
            )
            return False

        try:
            self._raw["active_preset"] = preset_name
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(
                    self._raw, f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
            self._load()
            console.print(
                f"[green]✅ Switched to: "
                f"[bold]{preset_name}[/bold][/green]"
            )
            return True
        except Exception as e:
            console.print(
                f"[red]✗ Could not switch preset: {e}[/red]"
            )
            return False

    def list_presets(self):
        console.print(
            "\n[bold cyan]Available Presets:[/bold cyan]\n"
        )
        for preset_file in sorted(PRESET_DIR.glob("*.yaml")):
            try:
                with open(preset_file, "r") as f:
                    data = yaml.safe_load(f)
                active = (
                    " ← ACTIVE"
                    if preset_file.stem == self.preset_name
                    else ""
                )
                display = data.get("display", {})
                console.print(
                    f"  [bold]{preset_file.stem:<12}[/bold]"
                    f"[cyan]{active}[/cyan]\n"
                    f"    {data.get('description', '')}\n"
                    f"    Time: {display.get('meeting_time', 'N/A')}  "
                    f"Quality: {display.get('quality', 'N/A')}\n"
                )
            except Exception:
                console.print(
                    f"  {preset_file.stem} (could not read)"
                )


# ── Singleton instance ─────────────────────────────────
cfg = Config()


# ── CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "summary":
            cfg.summary()

        elif command == "presets":
            cfg.list_presets()

        elif command == "switch" and len(sys.argv) > 2:
            cfg.switch_preset(sys.argv[2])
            cfg.summary()

        elif command == "test":
            console.print(
                "\n[bold]Testing config values:[/bold]\n"
            )
            console.print(f"  Preset:        {cfg.preset_name}")
            console.print(f"  CEO model:     {cfg.ceo_model}")
            console.print(f"  CTO model:     {cfg.cto_model}")
            console.print(f"  CFO model:     {cfg.cfo_model}")
            console.print(f"  CPO model:     {cfg.cpo_model}")
            console.print(f"  COO model:     {cfg.coo_model}")
            console.print(f"  Vision model:  {cfg.vision_model}")
            console.print(f"  Bridge port:   {cfg.bridge_port}")
            console.print(f"  Timeout:       {cfg.first_token_secs}s")
            console.print(f"  Track perf:    {cfg.track_performance}")
        else:
            console.print(
                "Usage: python core/config_loader.py "
                "[summary|presets|switch <name>|test]"
            )
    else:
        cfg.summary()