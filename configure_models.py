# configure_models.py
# Writes optimal LM Studio settings for all your models
# Reads hardware config from config.yaml via cfg
# Run: python configure_models.py --dry-run
# Run: python configure_models.py
# Run: python configure_models.py --verify

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from core.config_loader import cfg

console = Console()

YOUR_VRAM_MB    = 12287
VRAM_RESERVE_MB = 1024

CONFIG_DIR = (
    Path.home() /
    ".lmstudio" /
    ".internal" /
    "user-concrete-model-default-config"
)

KV_BYTES_PER_TOKEN_Q8 = {
    "3b":131072,"4b":131072,"7b":262144,"8b":262144,
    "9b":262144,"12b":393216,"14b":393216,"20b":524288,
    "22b":524288,"26b":655360,"27b":655360,"30b":786432,
    "31b":786432,"32b":786432,"35b":917504,"70b":1835008,
}

MODEL_WEIGHT_VRAM_MB = {
    "gemma-4-e2b":                   4400,
    "gemma-3-4b":                    3100,
    "gemma-3n-e4b":                  4200,
    "gemma-4-e4b":                   6300,
    "deepseek-r1-0528-qwen3-8b":     5000,
    "qwen2.5-vl-7b-instruct":        7500,
    "qwen3.5-9b":                    6500,
    "phi-4-reasoning-plus":          9100,
    "gemma-3-12b":                   8200,
    "qwen2.5-coder-14b-instruct":    9000,
    "gemma-4-26b-a4b":               18000,
    "gemma-4-31b":                   19900,
    "qwen3-coder-30b":               18600,
    "deepseek-r1-distill-qwen-32b":  19900,
    "qwq-32b":                       19900,
    "qwen3.6-27b":                   17500,
    "nemotron-3-nano-omni":          26100,
    "text-embedding-nomic-embed-text-v1.5":  8000,
}

KV_QUANT_MULTIPLIER = {
    "q8_0": 1.00,
    "q6_0": 0.75,
    "q5_0": 0.63,
    "q4_0": 0.50,
}

YOUR_MODELS = [
    {"publisher": "google",    "name": "gemma-4-e2b"},
    {"publisher": "google",    "name": "gemma-4-e4b"},
    {"publisher": "deepseek",  "name": "deepseek-r1-0528-qwen3-8b"},
    {"publisher": "qwen",      "name": "qwen2.5-vl-7b-instruct"},
    {"publisher": "qwen",      "name": "qwen3.5-9b"},
    {"publisher": "microsoft", "name": "phi-4-reasoning-plus"},
    {"publisher": "google",    "name": "gemma-3-12b"},
    {"publisher": "qwen",      "name": "qwen2.5-coder-14b-instruct"},
    {"publisher": "google",    "name": "gemma-4-26b-a4b"},
    {"publisher": "google",    "name": "gemma-4-31b"},
    {"publisher": "qwen",      "name": "qwen3-coder-30b"},
    {"publisher": "deepseek",  "name": "deepseek-r1-distill-qwen-32b"},
    {"publisher": "qwen",      "name": "qwq-32b"},
    {"publisher": "qwen",      "name": "qwen3.6-27b"},
    {"publisher": "nvidia",    "name": "nemotron-3-nano-omni"},
    {"publisher": "unknown",     "name": "text-embedding-nomic-embed-text-v1.5"},  # 0.0 GB
]


def get_size_key(model_name: str) -> str:
    sizes = [
        "70b","35b","32b","31b","30b","27b","26b",
        "22b","20b","14b","12b","9b","8b","7b","4b","3b"
    ]
    for s in sizes:
        if s in model_name.lower():
            return s
    return "12b"


def get_weight_vram(model_name: str) -> int:
    name_lower = model_name.lower()
    for key, vram in MODEL_WEIGHT_VRAM_MB.items():
        if key in name_lower:
            return vram
    size     = get_size_key(model_name)
    defaults = {
        "3b":3000,"4b":4000,"7b":5000,"8b":5500,
        "9b":6500,"12b":8000,"14b":9000,"20b":12000,
        "22b":14000,"26b":17000,"27b":17000,"30b":18000,
        "31b":19000,"32b":20000,"35b":22000,"70b":43000,
    }
    return defaults.get(size, 8000)


def get_kv_quant(model_name: str) -> tuple:
    weight_mb    = get_weight_vram(model_name)
    available_mb = YOUR_VRAM_MB - VRAM_RESERVE_MB - weight_mb
    vram_pct     = weight_mb / (YOUR_VRAM_MB - VRAM_RESERVE_MB)

    if available_mb <= 0:
        return ("q4_0", "q4_0")
    elif vram_pct < 0.40:
        return ("q8_0", "q8_0")
    elif vram_pct < 0.65:
        return ("q8_0", "q4_0")
    else:
        return ("q6_0", "q4_0")


def get_optimal_context(model_name: str, v_quant: str) -> int:
    weight_mb       = get_weight_vram(model_name)
    size_key        = get_size_key(model_name)
    available_mb    = YOUR_VRAM_MB - VRAM_RESERVE_MB - weight_mb

    if available_mb <= 0:
        return 2048

    kv_bytes_q8     = KV_BYTES_PER_TOKEN_Q8.get(size_key, 393216)
    multiplier      = KV_QUANT_MULTIPLIER.get(v_quant, 1.0)
    kv_bytes_eff    = kv_bytes_q8 * multiplier
    available_bytes = available_mb * 1024 * 1024
    max_tokens      = int(available_bytes / kv_bytes_eff)

    power_of_2 = [
        512,1024,2048,4096,8192,16384,32768,65536,131072
    ]
    optimal = 512
    for s in power_of_2:
        if s <= max_tokens:
            optimal = s
        else:
            break

    hard_caps = {
        "3b":32768,"4b":32768,"7b":16384,"8b":16384,
        "9b":16384,"12b":8192,"14b":8192,"20b":4096,
        "22b":4096,"26b":4096,"27b":4096,"30b":4096,
        "31b":4096,"32b":4096,"35b":2048,"70b":2048,
    }
    return min(optimal, hard_caps.get(size_key, 8192))


def build_config(
    model_name:      str,
    context:         int,
    k_quant:         str,
    v_quant:         str,
    existing_config: dict = None
) -> dict:
    config = existing_config.copy() if existing_config else {
        "preset":    "",
        "operation": {"fields": []},
        "load":      {"fields": []}
    }

    if "load" not in config:
        config["load"] = {"fields": []}
    if "fields" not in config["load"]:
        config["load"]["fields"] = []

    new_load_fields = [
        {
            "key":   "llm.load.contextLength",
            "value": context
        },
        {
            "key":   "llm.load.llama.kcachequantizationtype",
            "value": {"checked": True, "value": k_quant}
        },
        {
            "key":   "llm.load.llama.vcachequantizationtype",
            "value": {"checked": True, "value": v_quant}
        },
        {
            "key":   "llm.load.gpuOffload.ratio",
            "value": 1.0
        },
        {
            "key":   "llm.load.llama.flashAttention",
            "value": True
        },
    ]

    existing_keys = {
        field["key"]: i
        for i, field in enumerate(config["load"]["fields"])
    }

    for new_field in new_load_fields:
        key = new_field["key"]
        if key in existing_keys:
            config["load"]["fields"][existing_keys[key]] = new_field
        else:
            config["load"]["fields"].append(new_field)

    return config


def write_model_config(
    publisher:   str,
    model_name:  str,
    context:     int,
    k_quant:     str,
    v_quant:     str,
    dry_run:     bool = False
) -> bool:
    config_path     = CONFIG_DIR / publisher / f"{model_name}.json"
    existing_config = None

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
        except Exception:
            pass

    new_config = build_config(
        model_name, context, k_quant, v_quant, existing_config
    )

    if dry_run:
        console.print(
            f"[dim]  DRY RUN: {config_path}[/dim]"
        )
        return True

    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        backup = config_path.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(config_path, backup)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=2)
        return True
    except Exception as e:
        console.print(f"[red]  Write failed: {e}[/red]")
        return False


def configure_all_models(dry_run: bool = False):
    if dry_run:
        console.print(
            "\n[yellow]DRY RUN — no files written[/yellow]\n"
        )
    else:
        console.print(
            "\n[bold red]Close LM Studio before running![/bold red]\n"
        )
        confirm = input(
            "Is LM Studio closed? (yes/no): "
        ).strip().lower()
        if confirm not in ["yes", "y"]:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    table = Table(
        title=f"Writing LM Studio Configs (Preset: {cfg.preset_name})",
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Model",   width=36)
    table.add_column("Context", width=9,  justify="right")
    table.add_column("K Cache", width=8,  justify="center")
    table.add_column("V Cache", width=8,  justify="center")
    table.add_column("Mode",    width=12)
    table.add_column("Status",  width=10)

    success_count = 0
    fail_count    = 0

    for model in YOUR_MODELS:
        publisher  = model["publisher"]
        name       = model["name"]
        weight_mb  = get_weight_vram(name)
        k_q, v_q   = get_kv_quant(name)
        context    = get_optimal_context(name, v_q)
        fits_gpu   = (weight_mb <= YOUR_VRAM_MB - VRAM_RESERVE_MB)
        mode       = "GPU" if fits_gpu else "RAM offload"

        success = write_model_config(
            publisher, name, context, k_q, v_q, dry_run
        )

        status       = (
            "[green]OK[/green]" if success
            else "[red]FAIL[/red]"
        )
        mode_display = (
            "[green]GPU[/green]" if fits_gpu
            else "[yellow]RAM[/yellow]"
        )

        table.add_row(
            name, f"{context:,}", k_q, v_q,
            mode_display, status
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    console.print()
    console.print(table)
    console.print()

    if not dry_run:
        console.print(
            f"[green]Done! {success_count} models configured.[/green]"
        )
        if fail_count > 0:
            console.print(
                f"[red]{fail_count} failed.[/red]"
            )
        console.print(
            "\n[bold]Open LM Studio.[/bold]\n"
            "All models now have perfect settings pre-loaded.\n"
        )
    else:
        console.print(
            "[dim]Dry run complete. "
            "Run without --dry-run to apply.[/dim]"
        )


def verify_configs():
    console.print(
        "\n[bold cyan]Verifying configs...[/bold cyan]\n"
    )
    all_good = True

    for model in YOUR_MODELS:
        publisher   = model["publisher"]
        name        = model["name"]
        config_path = CONFIG_DIR / publisher / f"{name}.json"

        if not config_path.exists():
            console.print(f"[red]  MISSING: {name}[/red]")
            all_good = False
            continue

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            load_fields = {
                field["key"]: field["value"]
                for field in config.get("load", {}).get("fields", [])
            }

            ctx = load_fields.get(
                "llm.load.contextLength", "NOT SET"
            )
            k_entry = load_fields.get(
                "llm.load.llama.kcachequantizationtype", {}
            )
            v_entry = load_fields.get(
                "llm.load.llama.vcachequantizationtype", {}
            )
            k_q = (
                k_entry.get("value", "NOT SET")
                if isinstance(k_entry, dict) else "NOT SET"
            )
            v_q = (
                v_entry.get("value", "NOT SET")
                if isinstance(v_entry, dict) else "NOT SET"
            )

            console.print(
                f"  [green]OK[/green]  {name:<38} "
                f"ctx={ctx}  K={k_q}  V={v_q}"
            )

        except Exception as e:
            console.print(f"  [red]ERR[/red] {name}: {e}")
            all_good = False

    if all_good:
        console.print(
            "\n[bold green]All configs verified.[/bold green]"
        )
    else:
        console.print(
            "\n[bold yellow]Some configs have issues.[/bold yellow]"
        )


if __name__ == "__main__":
    import sys

    if "--verify" in sys.argv:
        verify_configs()
    elif "--dry-run" in sys.argv:
        configure_all_models(dry_run=True)
    else:
        configure_all_models(dry_run=False)