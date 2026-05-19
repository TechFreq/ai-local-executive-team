# load_model.py
# ══════════════════════════════════════════════════════════════════
# LM Studio Smart Loader — Full Working Edition
#
# CONFIRMED FROM LOGS:
#   ✅ POST /api/v1/models/load  → {"model": "..."} only
#   ✅ GET  /api/v1/models       → lists downloaded models
#   ✅ GET  /v1/models           → lists loaded models
#   ✅ POST /api/v1/models/unload → requires instance_id
#   ✅ Config path: .lmstudio\.internal\model-data.json
#   ✗  GET  /api/v1/system       → 404
#   ✗  GET  /api/v1/models/status → 404
#   ✗  Config in load payload    → "Unrecognized key: config"
#
# YOUR HARDWARE (confirmed from logs):
#   GPU:  NVIDIA RTX 3060 — 12287 MiB VRAM
#   VRAM free at load: ~11253 MiB
#
# KEY BEHAVIOURS CONFIRMED:
#   • Unload requires instance_id not model name
#   • Warmup can take 6-22s AFTER model appears in /v1/models
#   • Gemma 4 is multimodal — loads vision encoder too during warmup
#   • model-data.json is the real config file LM Studio reads
#   • Config file write + unload strategy works for Gemma 4
#   • Phi-4: 37/41 layers on GPU, 8.43GB, loads in ~3s + warmup
#   • Gemma 4 31B: 27/61 layers on GPU, 17.39GB, n_ctx=2048 worked
# ══════════════════════════════════════════════════════════════════

import requests
import threading
import time
import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    BarColumn,
)
from core.config_loader import cfg

load_dotenv()
console = Console()

# ── Base URL ──────────────────────────────────────────────────────
_raw           = cfg.lm_studio_url
LM_STUDIO_BASE = _raw.rstrip("/").removesuffix("/v1")
LOAD_TIMEOUT   = cfg.load_timeout_secs

# ── Confirmed model ID map ────────────────────────────────────────
# Models downloaded from non-lmstudio-community HF orgs (Donnyed,
# unsloth, openfree, etc.) lose their publisher prefix in LM Studio's
# API — it returns just the model name, not publisher/name.
# These are confirmed mappings: our registry ID → LM Studio API ID.
# Checked first in resolve_model_id before any dynamic lookup.
CONFIRMED_ID_MAP: dict[str, str] = {
    "qwen/qwen2.5-coder-14b-instruct":      "qwen2.5-coder-14b-instruct",
    "qwen/qwq-32b":                          "qwq-32b",
    "deepseek/deepseek-r1-distill-qwen-32b": "deepseek-r1-distill-qwen-32b",
    "qwen/qwen2.5-vl-7b-instruct":           "qwen2.5-vl-7b-instruct",
    "nomic-ai/nomic-embed-text-v1.5":        "text-embedding-nomic-embed-text-v1.5",
}

# ── instance_id cache — populated from load responses ─────────────
# LM Studio requires instance_id to unload but doesn't expose it in
# GET /api/v1/models.  We capture it from the POST /api/v1/models/load
# response body and store it here so unload_model can use it.
_instance_id_cache: dict[str, str] = {}

# ── Hardware (confirmed from logs) ───────────────────────────────
GPU_VRAM_GB      = 12.0
GPU_VRAM_FREE_GB = 10.5   # ~11253 MiB free at load time
GPU_NAME         = "NVIDIA RTX 3060 12GB"

# ── Warmup buffer ────────────────────────────────────────────────
# Phi-4 warmup took up to 22s after model appeared in /v1/models
# Gemma 4 warmup ~5s but also loads vision encoder
# Keep POST connection alive — it returns 200 when warmup done
POST_DETECT_WARMUP_SECS = 25

# ── Logging ───────────────────────────────────────────────────────
LOG_DIR  = Path("logs") / "model_loads"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"load_{datetime.now().strftime('%Y-%m-%d')}.log"

# ── Learned settings ─────────────────────────────────────────────
LEARNED_SETTINGS_FILE = Path("learned_settings.json")

# ── LM Studio paths ───────────────────────────────────────────────
LM_STUDIO_ROOT = Path.home() / ".lmstudio"

# Confirmed real config file from logs
CONFIRMED_GLOBAL_CONFIG = (
    LM_STUDIO_ROOT / ".internal" / "model-data.json"
)

# All candidate config roots (broad sweep on first run)
CANDIDATE_CONFIG_ROOTS = [
    LM_STUDIO_ROOT / ".internal" / "user-concrete-model-default-config",
    LM_STUDIO_ROOT / ".internal" / "model-configs",
    LM_STUDIO_ROOT / "configs" / "models",
    LM_STUDIO_ROOT / "model-configs",
    LM_STUDIO_ROOT / "settings" / "models",
    LM_STUDIO_ROOT / "user-data" / "model-configs",
    Path.home() / "AppData" / "Roaming" / "LM Studio" / "model-configs",
    Path.home() / "AppData" / "Local"  / "LM Studio" / "model-configs",
    Path.home() / "AppData" / "Roaming" / "LM Studio" / "configs",
    Path.home() / "AppData" / "Local"  / "LM Studio" / "configs",
]


# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════

def _log(level: str, message: str, data: dict = None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_pad = level.upper().ljust(7)
    line      = f"[{timestamp}] [{level_pad}] {message}"
    if data:
        line += f"  |  {json.dumps(data)}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def log_info(msg: str, data: dict = None):    _log("INFO",    msg, data)
def log_warn(msg: str, data: dict = None):    _log("WARN",    msg, data)
def log_error(msg: str, data: dict = None):   _log("ERROR",   msg, data)
def log_success(msg: str, data: dict = None): _log("SUCCESS", msg, data)
def log_debug(msg: str, data: dict = None):   _log("DEBUG",   msg, data)


def log_separator(label: str = ""):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 70
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{ts}] {sep}\n")
            if label:
                f.write(f"[{ts}] {label}\n")
                f.write(f"[{ts}] {sep}\n")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# CONFIG FILE DISCOVERY
# Scans ~/.lmstudio for json files, watches which ones change
# during a load to find where LM Studio actually reads from
# ══════════════════════════════════════════════════════════════════

def scan_all_lmstudio_json_files() -> list[Path]:
    """All json files under ~/.lmstudio except model weights."""
    if not LM_STUDIO_ROOT.exists():
        return []
    results = []
    try:
        for p in LM_STUDIO_ROOT.rglob("*.json"):
            if "models" in p.parts:
                continue
            if p.name in ("learned_settings.json",):
                continue
            results.append(p)
    except Exception as e:
        log_warn(f"Error scanning .lmstudio: {e}")
    return results


def get_all_json_mtimes() -> dict[str, float]:
    """Snapshot of mtime for all json files — used to detect changes."""
    mtimes = {}
    for p in scan_all_lmstudio_json_files():
        try:
            mtimes[str(p)] = p.stat().st_mtime
        except Exception:
            pass
    return mtimes


def find_changed_json_files(
    before_mtimes: dict[str, float]
) -> list[Path]:
    """
    Compares current mtimes to snapshot.
    Returns files LM Studio modified during the load.
    This tells us exactly where it reads config from.
    """
    changed = []
    current_files = scan_all_lmstudio_json_files()
    current_paths = {str(p) for p in current_files}

    for p in current_files:
        path_str = str(p)
        try:
            current_mtime = p.stat().st_mtime
            prev_mtime    = before_mtimes.get(path_str, 0)
            if current_mtime > prev_mtime + 0.5:
                changed.append(p)
                log_info(f"Config changed during load: {p}")
        except Exception:
            pass

    # New files created during load
    for path_str in current_paths:
        if path_str not in before_mtimes:
            changed.append(Path(path_str))
            log_info(f"New config file created: {path_str}")

    return changed


def find_config_files_for_model(lm_id: str) -> list[Path]:
    """Finds existing config files matching this model's name."""
    parts      = lm_id.split("/", 1)
    model_name = parts[1] if len(parts) == 2 else parts[0]
    model_name_clean = model_name.replace("/", "_").replace("\\", "_")

    found = []
    for p in scan_all_lmstudio_json_files():
        name_lower = p.stem.lower()
        if (
            model_name_clean.lower() in name_lower
            or name_lower in model_name_clean.lower()
        ):
            found.append(p)
            log_info(f"Found existing config for {lm_id}: {p}")
    return found


def read_json_safe(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_json_safe(path: Path, data: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log_warn(f"Write failed {path}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# CONFIG WRITER
# ══════════════════════════════════════════════════════════════════

def build_settings_payload(
    context: int,
    k_cache: str,
    v_cache: str,
    flash_attn: bool,
) -> dict:
    """
    Builds config JSON in multiple formats LM Studio has used.
    Tries all known key names for maximum compatibility.
    """
    fields = [
        {"key": "llm.load.contextLength",             "value": context},
        {"key": "contextLength",                       "value": context},
        {
            "key":   "llm.load.llama.kcachequantizationtype",
            "value": {"checked": True, "value": k_cache},
        },
        {
            "key":   "llm.load.llama.kvCacheQuantizationType",
            "value": k_cache,
        },
        {"key": "kvCacheQuantizationType",             "value": k_cache},
        {
            "key":   "llm.load.llama.vcachequantizationtype",
            "value": {"checked": True, "value": v_cache},
        },
        {
            "key":   "llm.load.llama.vCacheQuantizationType",
            "value": v_cache,
        },
        {"key": "vCacheQuantizationType",              "value": v_cache},
        {"key": "llm.load.llama.flashAttention",       "value": flash_attn},
        {"key": "flashAttention",                      "value": flash_attn},
        {"key": "llm.load.gpuOffload.ratio",           "value": 1.0},
        {"key": "gpuOffloadRatio",                     "value": 1.0},
    ]

    return {
        "load": {"fields": fields},
        # Flat keys for newer LM Studio versions
        "contextLength":           context,
        "kvCacheQuantizationType": k_cache,
        "vCacheQuantizationType":  v_cache,
        "flashAttention":          flash_attn,
        "gpuOffloadRatio":         1.0,
    }


def _apply_settings_to_entry(
    entry: dict,
    context: int,
    k_cache: str,
    v_cache: str,
    flash_attn: bool,
) -> dict:
    """Applies settings to a model config entry dict."""
    settings = {
        "contextLength":           context,
        "kvCacheQuantizationType": k_cache,
        "vCacheQuantizationType":  v_cache,
        "flashAttention":          flash_attn,
    }
    entry.update(settings)
    if "load" in entry:
        entry["load"].update(settings)
    if "config" in entry:
        entry["config"].update(settings)
    return entry


def _merge_into_global_config(
    existing: dict,
    lm_id: str,
    context: int,
    k_cache: str,
    v_cache: str,
    flash_attn: bool,
) -> dict:
    """
    Merges settings into model-data.json without clobbering other models.
    Logs the full structure so we can learn the format over time.
    """
    log_info(
        f"Merging into model-data.json for {lm_id}",
        {"top_level_keys": list(existing.keys())[:15]}
    )

    # Log full structure for analysis
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"\n[MODEL-DATA.JSON STRUCTURE — {lm_id}]\n"
                f"{json.dumps(existing, indent=2)[:5000]}\n"
                f"[END MODEL-DATA.JSON]\n\n"
            )
    except Exception:
        pass

    lm_lower = lm_id.lower()
    lm_name  = lm_id.split("/")[-1].lower()

    # Structure 1: {"models": {"google/gemma-4-31b": {...}}}
    if "models" in existing and isinstance(existing["models"], dict):
        for key in list(existing["models"].keys()):
            if key.lower() == lm_lower or key.lower() == lm_name:
                existing["models"][key] = _apply_settings_to_entry(
                    existing["models"][key],
                    context, k_cache, v_cache, flash_attn
                )
                log_info(f"Updated models dict entry: {key}")
                return existing

    # Structure 2: {"entries": [...]}
    if "entries" in existing and isinstance(existing["entries"], list):
        for i, entry in enumerate(existing["entries"]):
            entry_id = (
                entry.get("id") or entry.get("model") or ""
            ).lower()
            if entry_id == lm_lower or lm_name in entry_id:
                existing["entries"][i] = _apply_settings_to_entry(
                    entry, context, k_cache, v_cache, flash_attn
                )
                log_info(f"Updated entries list entry: {entry_id}")
                return existing

    # Structure unknown — safe fallback key
    existing.setdefault("_ai_team_overrides", {})
    existing["_ai_team_overrides"][lm_id] = {
        "contextLength":           context,
        "kvCacheQuantizationType": k_cache,
        "vCacheQuantizationType":  v_cache,
        "flashAttention":          flash_attn,
        "written_at":              datetime.now().isoformat(),
    }
    log_warn(
        "model-data.json structure unrecognized — "
        "wrote to _ai_team_overrides. Check log for structure."
    )
    return existing


def get_confirmed_config_paths(model_id: str) -> list[Path]:
    """Returns config paths confirmed working for this model."""
    data  = load_learned_settings()
    model = data.get(model_id, {})
    paths = model.get("confirmed_config_paths", [])
    return [Path(p) for p in paths if Path(p).exists()]


def save_confirmed_config_paths(model_id: str, paths: list[Path]):
    """Saves the config paths LM Studio actually used."""
    data = load_learned_settings()
    data.setdefault(model_id, {})
    data[model_id]["confirmed_config_paths"] = [str(p) for p in paths]
    try:
        with open(LEARNED_SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        log_info(
            f"Saved confirmed config paths: {model_id}",
            {"paths": [str(p) for p in paths]}
        )
    except Exception as e:
        log_warn(f"Cannot save config paths: {e}")


def write_config_everywhere(
    lm_id: str,
    context: int,
    k_cache: str,
    v_cache: str,
    flash_attn: bool,
    extra_paths: list[Path] = None,
) -> tuple[int, list[Path]]:
    """
    Writes config to the right locations.

    Logic:
        - If we have confirmed paths from a previous load → use only those
        - Otherwise → broad sweep of all candidate locations
        - Always include model-data.json if it exists (confirmed from logs)
    """
    parts           = lm_id.split("/", 1)
    publisher       = parts[0] if len(parts) == 2 else "unknown"
    model_name      = parts[1] if len(parts) == 2 else parts[0]
    model_name_safe = model_name.replace("/", "_")

    payload    = build_settings_payload(context, k_cache, v_cache, flash_attn)
    confirmed  = get_confirmed_config_paths(lm_id)

    if confirmed:
        # We know what works — write only there
        paths_to_write = set(confirmed)
        if CONFIRMED_GLOBAL_CONFIG.exists():
            paths_to_write.add(CONFIRMED_GLOBAL_CONFIG)
        log_info(
            f"Writing to {len(paths_to_write)} confirmed path(s)"
        )
    else:
        # First time — broad sweep
        paths_to_write = set()

        if CONFIRMED_GLOBAL_CONFIG.exists():
            paths_to_write.add(CONFIRMED_GLOBAL_CONFIG)

        for root in CANDIDATE_CONFIG_ROOTS:
            paths_to_write.add(root / publisher / f"{model_name_safe}.json")
            paths_to_write.add(root / f"{model_name_safe}.json")
            paths_to_write.add(
                root / f"{lm_id.replace('/', '_')}.json"
            )

        for p in find_config_files_for_model(lm_id):
            paths_to_write.add(p)

    if extra_paths:
        for p in extra_paths:
            paths_to_write.add(p)

    written     = []
    write_count = 0

    for config_path in paths_to_write:
        existing = (
            read_json_safe(config_path)
            if config_path.exists()
            else {}
        )

        if config_path.name == "model-data.json":
            merged = _merge_into_global_config(
                existing, lm_id, context, k_cache, v_cache, flash_attn
            )
        else:
            existing.setdefault("load", {})
            existing["load"].setdefault("fields", [])

            existing_keys = {
                f["key"]: i
                for i, f in enumerate(existing["load"]["fields"])
            }
            our_fields = payload["load"]["fields"]
            our_keys   = {f["key"] for f in our_fields}
            kept       = [
                f for f in existing["load"]["fields"]
                if f["key"] not in our_keys
            ]
            existing["load"]["fields"] = kept + our_fields
            existing.update(
                {k: v for k, v in payload.items() if k != "load"}
            )
            merged = existing

        if write_json_safe(config_path, merged):
            write_count += 1
            written.append(config_path)

    log_info(f"Config written to {write_count} location(s)", {
        "model":      lm_id,
        "context":    context,
        "k_cache":    k_cache,
        "v_cache":    v_cache,
        "flash_attn": flash_attn,
    })

    return write_count, written


# ══════════════════════════════════════════════════════════════════
# UNLOAD
# Requires instance_id — fetch it from /api/v1/models first
# ══════════════════════════════════════════════════════════════════

def get_model_instance_id(lm_id: str) -> str | None:
    """
    Returns the instance_id required for the unload endpoint.
    Checks the load-response cache first, then falls back to GET /api/v1/models.
    """
    # Fast path: we cached it when the model was loaded
    cached = _instance_id_cache.get(lm_id)
    if cached:
        return cached

    try:
        r = requests.get(
            f"{LM_STUDIO_BASE}/api/v1/models", timeout=5
        )
        if r.status_code == 200:
            raw   = r.json()
            items = raw if isinstance(raw, list) else raw.get("data", [])

            lm_lower = lm_id.lower()
            lm_name  = lm_id.split("/")[-1].lower()

            for item in items:
                item_id   = (item.get("id") or "").lower()
                item_name = item_id.split("/")[-1]

                if item_id == lm_lower or item_name == lm_name:
                    # Log the full item to learn all available fields
                    log_info(
                        f"Model entry from /api/v1/models",
                        {"item": item}
                    )

                    instance_id = (
                        item.get("instance_id")
                        or item.get("instanceId")
                        or item.get("instance")
                    )

                    if instance_id:
                        log_info(
                            f"Found instance_id for {lm_id}: {instance_id}"
                        )
                        return instance_id

                    log_warn(
                        f"Model found but no instance_id field. "
                        f"Available keys: {list(item.keys())}"
                    )
                    return None

    except Exception as e:
        log_warn(f"Cannot get instance_id: {e}")

    return None


def unload_model(lm_id: str) -> bool:
    """
    Unloads a model via POST /api/v1/models/unload with instance_id.
    For the base instance the instance_id is just the model ID itself;
    for duplicate instances it is the :N-suffixed ID from /v1/models.
    """
    url = f"{LM_STUDIO_BASE}/api/v1/models/unload"

    # Cache hit → use what we stored from the load response
    instance_id = get_model_instance_id(lm_id)

    if not instance_id:
        # No cache: look in /v1/models for the base instance (no :N suffix)
        loaded = get_loaded_model_ids()
        lm_name = lm_id.split("/")[-1].lower()
        for mid in loaded:
            mid_name = mid.split("/")[-1].lower()
            mid_base = mid_name.split(":")[0]
            is_dup   = ":" in mid and mid.split(":")[-1].isdigit()
            if not is_dup and (mid.lower() == lm_id.lower() or mid_base == lm_name):
                instance_id = mid
                log_info(f"Resolved instance_id from /v1/models: {instance_id}")
                break

    if not instance_id:
        # Last resort: the base instance_id equals the model ID itself
        instance_id = lm_id
        log_warn(f"No instance_id resolved — using model ID as instance_id: {lm_id}")

    payload = {"instance_id": instance_id}
    log_info(f"Unloading {lm_id} with instance_id: {instance_id}")

    try:
        r = requests.post(url, json=payload, timeout=30)

        if r.status_code in (200, 201, 202, 204):
            log_success(f"Unloaded: {lm_id}")
            console.print(f"[dim]  → Unloaded {lm_id}[/dim]")
            _instance_id_cache.pop(lm_id, None)
            return True

        if r.status_code == 404:
            log_info(f"Model not loaded (404): {lm_id}")
            _instance_id_cache.pop(lm_id, None)
            return True

        log_warn(
            f"Unload returned {r.status_code}: {r.text[:300]}",
            {"payload": payload}
        )

    except Exception as e:
        log_warn(f"Unload error (continuing): {e}")

    # Fallback: TTL trick — send a 1-token request with model_ttl_seconds=0.
    # LM Studio will unload the model after the request completes.
    # Works even when we don't have instance_id.
    log_warn(f"Trying TTL=0 fallback to eject {lm_id}")
    try:
        requests.post(
            f"{LM_STUDIO_BASE}/v1/chat/completions",
            json={
                "model":             lm_id,
                "messages":          [{"role": "user", "content": "x"}],
                "max_tokens":        1,
                "temperature":       0,
                "model_ttl_seconds": 0,
            },
            timeout=30,
        )
        _instance_id_cache.pop(lm_id, None)
        console.print(f"[dim]  → Ejected {lm_id} via TTL trick[/dim]")
        time.sleep(2)
        return True
    except Exception as e:
        log_warn(f"TTL fallback also failed: {e}")
        return False


def unload_any_loaded_model():
    """Unloads all currently loaded models."""
    loaded = get_loaded_model_ids()
    if not loaded:
        log_info("No models currently loaded")
        return
    for mid in loaded:
        log_info(f"Unloading: {mid}")
        console.print(f"[dim]  → Unloading {mid}...[/dim]")
        unload_model(mid)
        time.sleep(1)


# ══════════════════════════════════════════════════════════════════
# MODEL ID RESOLUTION
# LM Studio uses inconsistent ID formats
# ══════════════════════════════════════════════════════════════════

def get_lm_studio_model_ids() -> list[str]:
    """
    Returns all model IDs LM Studio knows about.
    Queries both endpoints and merges — they return different ID formats
    depending on where a model was downloaded from.
    /api/v1/models = all downloaded (may have publisher prefix or not)
    /v1/models     = currently loaded (reliable, OpenAI-compatible)
    """
    ids: set[str] = set()

    try:
        r = requests.get(f"{LM_STUDIO_BASE}/api/v1/models", timeout=5)
        if r.status_code == 200:
            raw   = r.json()
            items = raw if isinstance(raw, list) else raw.get("data", [])
            for item in items:
                mid = item.get("id") or item.get("model_id") or ""
                if mid:
                    ids.add(mid)
    except Exception as e:
        log_warn(f"Cannot list /api/v1/models: {e}")

    try:
        r = requests.get(f"{LM_STUDIO_BASE}/v1/models", timeout=5)
        if r.status_code == 200:
            for item in r.json().get("data", []):
                mid = item.get("id", "")
                # Skip :N suffixed duplicate instances
                if mid and ":" not in mid.split("/")[-1]:
                    ids.add(mid)
    except Exception as e:
        log_warn(f"Cannot list /v1/models: {e}")

    result = list(ids)
    log_info(f"Available: {len(result)} models", {"ids": result})
    return result


def get_loaded_model_ids() -> list[str]:
    """GET /v1/models — currently loaded models only."""
    try:
        r = requests.get(
            f"{LM_STUDIO_BASE}/v1/models", timeout=5
        )
        if r.status_code == 200:
            return [
                m.get("id", "")
                for m in r.json().get("data", [])
                if m.get("id")
            ]
    except Exception:
        pass
    return []

def get_api_v0_state() -> dict[str, str]:
    """
    GET /api/v0/models — confirmed from diagnostic to return:
      state: "loaded" or "not-loaded" for each model
      
    Returns dict of {lm_id: state}
    """
    try:
        resp = requests.get(
            f"{LM_STUDIO_BASE}/api/v0/models",
            timeout=5,
        )
        if resp.status_code == 200:
            result = {}
            for m in resp.json().get("data", []):
                mid   = m.get("id", "")
                state = m.get("state", "not-loaded")
                if mid:
                    result[mid] = state
            return result
    except Exception as e:
        log_warn(f"Cannot reach /api/v0/models: {e}")
    return {}


def is_model_already_loaded_correctly(lm_id: str) -> bool:
    """
    Returns True if the model's BASE instance (no :N suffix) is present in
    GET /v1/models, which is the only reliable "currently loaded" signal.
    /api/v0/models state field is NOT reliable — it shows "not-loaded" even
    when a model is actively loaded.

    Duplicate :N instances are logged but do NOT count as "correctly loaded".
    """
    loaded_ids = get_loaded_model_ids()   # /v1/models — shows real state
    lm_name    = lm_id.split("/")[-1].lower()

    found_base      = False
    found_duplicate = False

    for mid in loaded_ids:
        mid_name = mid.split("/")[-1].lower()
        mid_base = mid_name.split(":")[0]
        is_dup   = ":" in mid and mid.split(":")[-1].isdigit()

        if mid.lower() == lm_id.lower() or mid_base == lm_name:
            if is_dup:
                found_duplicate = True
                log_warn(f"Duplicate instance detected in /v1/models: {mid}")
                console.print(
                    f"[yellow]  ⚠ Duplicate instance: {mid} — will clean up[/yellow]"
                )
            else:
                found_base = True
                log_info(f"Model confirmed loaded via /v1/models: {mid}")
                console.print(
                    f"[dim]  ✓ Base instance confirmed loaded: {mid}[/dim]"
                )

    if found_duplicate and not found_base:
        log_warn("Only duplicate :N instance found — will eject and reload cleanly")
        return False

    return found_base


def cleanup_duplicate_instances(lm_id: str):
    """
    Ejects all duplicate :N instances of a model visible in /v1/models.
    The :N suffix (e.g. google/gemma-4-e2b:2) IS the instance_id accepted by
    POST /api/v1/models/unload — no TTL trick needed.
    Note: /api/v0/models does NOT show :N suffixed instances — /v1/models does.
    """
    lm_name = lm_id.split("/")[-1].lower()

    # /v1/models is the correct endpoint — it lists every loaded instance
    # including the :2, :3... duplicates that /api/v0/models hides.
    loaded_ids = get_loaded_model_ids()

    for mid in loaded_ids:
        mid_name = mid.split("/")[-1].lower()
        # Strip the :N suffix before comparing names
        mid_base_name = mid_name.split(":")[0]
        is_dup = ":" in mid and mid.split(":")[-1].isdigit()

        if is_dup and mid_base_name == lm_name:
            console.print(f"[yellow]  Ejecting duplicate: {mid}[/yellow]")
            log_warn(f"Ejecting duplicate instance: {mid}")
            try:
                # The :N suffix IS the instance_id — use the proper unload API
                r = requests.post(
                    f"{LM_STUDIO_BASE}/api/v1/models/unload",
                    json={"instance_id": mid},
                    timeout=30,
                )
                if r.status_code in (200, 201, 202, 204):
                    console.print(f"[dim]  ✓ Ejected {mid}[/dim]")
                elif r.status_code == 404:
                    console.print(f"[dim]  ✓ Already gone {mid}[/dim]")
                else:
                    log_warn(f"Unload {mid} returned {r.status_code}: {r.text[:200]}")
                time.sleep(1)
            except Exception as e:
                log_warn(f"Could not eject {mid}: {e}")
                
def resolve_model_id(
    our_id: str,
    available_ids: list[str],
) -> str:
    """
    Maps our registry ID to what LM Studio actually calls the model.
    Falls back to our_id if no match found.

    Known mappings from logs:
        qwen/qwq-32b                  → qwq-32b
        qwen/qwen2.5-coder-14b        → qwen2.5-coder-14b-instruct
        google/gemma-4-31b            → google/gemma-4-31b  (exact)
        microsoft/phi-4-reasoning-plus → microsoft/phi-4-reasoning-plus (exact)
    """
    # Check confirmed map first — handles models from non-community HF orgs
    if our_id in CONFIRMED_ID_MAP:
        log_info(f"Resolved (confirmed map): {our_id} → {CONFIRMED_ID_MAP[our_id]}")
        return CONFIRMED_ID_MAP[our_id]

    if not available_ids:
        return our_id

    if our_id in available_ids:
        return our_id

    our_name  = our_id.split("/")[-1].lower()
    our_lower = our_id.lower()

    # Name part exact match
    for avail in available_ids:
        if avail.split("/")[-1].lower() == our_name:
            log_info(f"Resolved (name): {our_id} → {avail}")
            return avail

    # Substring match
    for avail in available_ids:
        avail_lower = avail.lower()
        if our_lower in avail_lower or avail_lower in our_lower:
            log_info(f"Resolved (substr): {our_id} → {avail}")
            return avail

    # Word overlap (handles nomic/embedding name differences)
    our_words = set(our_name.replace("-", " ").split())
    for avail in available_ids:
        avail_words = set(
            avail.split("/")[-1].lower().replace("-", " ").split()
        )
        if len(our_words & avail_words) >= 2:
            log_info(f"Resolved (words): {our_id} → {avail}")
            return avail

    # Dynamic fallback: strip publisher prefix rather than returning a 404-prone ID.
    # Models downloaded from non-lmstudio-community HF orgs appear in LM Studio
    # as bare name only (no publisher prefix).
    if "/" in our_id:
        name_only = our_id.split("/", 1)[1]
        log_warn(f"No ID match for {our_id} — falling back to name-only: {name_only}")
        return name_only

    log_warn(f"No ID match for {our_id} — using as-is")
    return our_id


def is_model_in_loaded_list(lm_id: str) -> bool:
    """Checks /v1/models for this model."""
    loaded   = get_loaded_model_ids()
    lm_lower = lm_id.lower()
    lm_name  = lm_id.split("/")[-1].lower()

    for mid in loaded:
        if mid.lower() == lm_lower:
            return True
        if mid.split("/")[-1].lower() == lm_name:
            return True
    return False


# ══════════════════════════════════════════════════════════════════
# CONTEXT PICKER
# Calibrated for RTX 3060 12GB from confirmed log data
# ══════════════════════════════════════════════════════════════════

def pick_context_for_model(size_gb: float, gpu_fit: bool) -> int:
    """
    RTX 3060 12GB:
        Phi-4 (8.43GB): 16384 ctx → 3200MB KV → ~11.6GB total ✅
        Gemma 4 (17.39GB): 2048 ctx → 202+1225MB KV → loaded ✅

    KV cache size scales roughly linearly with context.
    """
    COMPUTE_BUFFER_GB = 1.5

    if gpu_fit:
        available_for_kv = GPU_VRAM_FREE_GB - size_gb - COMPUTE_BUFFER_GB
    else:
        # Hybrid: ~55% of model on GPU
        gpu_portion      = size_gb * 0.55
        available_for_kv = GPU_VRAM_FREE_GB - gpu_portion - COMPUTE_BUFFER_GB

    kv_per_16k = size_gb * 0.16
    kv_per_8k  = kv_per_16k / 2
    kv_per_4k  = kv_per_16k / 4

    if available_for_kv >= kv_per_16k:
        return 16384
    elif available_for_kv >= kv_per_8k:
        return 8192
    elif available_for_kv >= kv_per_4k:
        return 4096
    else:
        return 2048


# ══════════════════════════════════════════════════════════════════
# LEARNED SETTINGS
# ══════════════════════════════════════════════════════════════════

def load_learned_settings() -> dict:
    if LEARNED_SETTINGS_FILE.exists():
        try:
            with open(LEARNED_SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            log_warn(f"Cannot read learned_settings: {e}")
    return {}


def save_learned_setting(
    model_id: str,
    context: int,
    k_cache: str,
    v_cache: str,
    load_time_secs: int,
    flash_attn: bool,
    config_paths: list[Path] = None,
):
    data     = load_learned_settings()
    previous = data.get(model_id, {})

    entry = {
        "context":        context,
        "k_cache":        k_cache,
        "v_cache":        v_cache,
        "flash_attn":     flash_attn,
        "load_time_secs": load_time_secs,
        "success_count":  previous.get("success_count", 0) + 1,
        "last_success":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu":            GPU_NAME,
        "vram_gb":        GPU_VRAM_GB,
    }

    if config_paths:
        entry["confirmed_config_paths"] = [str(p) for p in config_paths]
    elif previous.get("confirmed_config_paths"):
        # Keep existing confirmed paths
        entry["confirmed_config_paths"] = previous["confirmed_config_paths"]

    data[model_id] = entry

    try:
        with open(LEARNED_SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        log_success(f"Learned settings saved: {model_id}", entry)
        console.print(
            "[dim]  ✨ Settings saved to learned_settings.json[/dim]"
        )
    except Exception as e:
        log_warn(f"Cannot save learned settings: {e}")


def get_learned_setting(model_id: str) -> dict:
    return load_learned_settings().get(model_id, {})


# ══════════════════════════════════════════════════════════════════
# ATTEMPT LADDER
# Builds the ordered list of settings to try.
# Adapts based on whether config path is confirmed or not.
# ══════════════════════════════════════════════════════════════════

def build_attempt_ladder(model_id: str, learned: dict) -> list:
    info    = get_model_info(model_id)
    gpu_fit = info.get("gpu_fit", True)
    size_gb = info.get("size_gb", 8)
    name    = model_id.lower()

    # Check if config path confirmed (means our writes are being read)
    confirmed_paths = get_confirmed_config_paths(model_id)
    config_works    = len(confirmed_paths) > 0

    start_ctx   = pick_context_for_model(size_gb, gpu_fit)
    half_ctx    = max(1024, start_ctx // 2)
    quarter_ctx = max(1024, start_ctx // 4)

    is_gemma4 = "gemma-4" in name or "gemma4" in name

    if is_gemma4:
        if config_works:
            # Config confirmed working — try progressively higher context
            # Start at 4096, fall back to 2048 (known working), try 8192
            base_ladder = [
                {
                    "label":        "Gemma4 (4096 K=q8 V=f16 FA=off) [config confirmed]",
                    "context":      4096,
                    "k_cache":      "q8_0",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                    "note": (
                        "Config path confirmed. "
                        "Trying higher context than last time."
                    ),
                },
                {
                    "label":        "Gemma4 (8192 K=q8 V=f16 FA=off) [ambitious]",
                    "context":      8192,
                    "k_cache":      "q8_0",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                    "note": (
                        "Gemma 4 supports 262k ctx. "
                        "Testing 8k on 12GB VRAM."
                    ),
                },
                {
                    "label":        "Gemma4 (2048 K=q8 V=f16 FA=off) [known working]",
                    "context":      2048,
                    "k_cache":      "q8_0",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                    "note":         "Previously confirmed working.",
                },
                {
                    "label":        "Gemma4 (1024 K=f16 V=f16 FA=off) [minimal]",
                    "context":      1024,
                    "k_cache":      "f16",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                },
            ]
        else:
            # First time — conservative, just get it loading
            base_ladder = [
                {
                    "label":        f"Gemma4 ({start_ctx} K=q8 V=f16 FA=off) [unload first]",
                    "context":      start_ctx,
                    "k_cache":      "q8_0",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                    "note": (
                        "Unloading first to force config re-read. "
                        "V=f16 required (no Flash Attn on hybrid load)."
                    ),
                },
                {
                    "label":        f"Gemma4 ({half_ctx} K=q8 V=f16 FA=off) [unload first]",
                    "context":      half_ctx,
                    "k_cache":      "q8_0",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                },
                {
                    "label":        f"Gemma4 ({quarter_ctx} K=f16 V=f16 FA=off)",
                    "context":      quarter_ctx,
                    "k_cache":      "f16",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                },
                {
                    "label":        "Gemma4 (2048 K=f16 V=f16 FA=off)",
                    "context":      2048,
                    "k_cache":      "f16",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": True,
                },
                {
                    "label":        "Gemma4 (1024 K=f16 V=f16 FA=off)",
                    "context":      1024,
                    "k_cache":      "f16",
                    "v_cache":      "f16",
                    "flash_attn":   False,
                    "unload_first": False,
                },
            ]

    elif gpu_fit and size_gb < 10:
        # Fits on GPU fully (confirmed: Phi-4 8.43GB on 12GB VRAM)
        base_ladder = [
            {
                "label":        f"GPU ({start_ctx} Q8/Q8 FA=on)",
                "context":      start_ctx,
                "k_cache":      "q8_0",
                "v_cache":      "q8_0",
                "flash_attn":   True,
                "unload_first": False,
            },
            {
                "label":        f"GPU ({start_ctx} Q8/F16 FA=off)",
                "context":      start_ctx,
                "k_cache":      "q8_0",
                "v_cache":      "f16",
                "flash_attn":   False,
                "unload_first": False,
            },
            {
                "label":        f"GPU ({half_ctx} Q8/F16 FA=off)",
                "context":      half_ctx,
                "k_cache":      "q8_0",
                "v_cache":      "f16",
                "flash_attn":   False,
                "unload_first": False,
            },
            {
                "label":        f"GPU ({quarter_ctx} F16/F16 FA=off)",
                "context":      quarter_ctx,
                "k_cache":      "f16",
                "v_cache":      "f16",
                "flash_attn":   False,
                "unload_first": False,
            },
        ]

    else:
        # Large hybrid model on 12GB VRAM
        base_ladder = [
            {
                "label":        f"Hybrid ({start_ctx} Q8/F16 FA=off) [unload first]",
                "context":      start_ctx,
                "k_cache":      "q8_0",
                "v_cache":      "f16",
                "flash_attn":   False,
                "unload_first": True,
                "note":         "Large model — GPU+RAM hybrid on 12GB",
            },
            {
                "label":        f"Hybrid ({half_ctx} Q8/F16 FA=off) [unload first]",
                "context":      half_ctx,
                "k_cache":      "q8_0",
                "v_cache":      "f16",
                "flash_attn":   False,
                "unload_first": True,
            },
            {
                "label":        f"Hybrid ({quarter_ctx} F16/F16 FA=off)",
                "context":      quarter_ctx,
                "k_cache":      "f16",
                "v_cache":      "f16",
                "flash_attn":   False,
                "unload_first": False,
            },
            {
                "label":        "Hybrid (2048 F16/F16 FA=off)",
                "context":      2048,
                "k_cache":      "f16",
                "v_cache":      "f16",
                "flash_attn":   False,
                "unload_first": False,
            },
        ]

    # Prepend learned settings (put first — we know it worked before)
    if learned:
        learned_entry = {
            "label": (
                f"Previously worked "
                f"({learned['context']} ctx "
                f"K={learned.get('k_cache','?')} "
                f"V={learned.get('v_cache','?')}) "
                f"[{learned.get('success_count',1)}x]"
            ),
            "context":      learned["context"],
            "k_cache":      learned.get("k_cache", "q8_0"),
            "v_cache":      learned.get("v_cache", "f16"),
            "flash_attn":   learned.get("flash_attn", False),
            "unload_first": True,
            "is_learned":   True,
        }
        # Remove duplicate from base ladder
        base_ladder = [
            a for a in base_ladder
            if not (
                a["context"] == learned["context"]
                and a["k_cache"] == learned.get("k_cache")
                and a["v_cache"] == learned.get("v_cache")
            )
        ]
        base_ladder = [learned_entry] + base_ladder

    for i, attempt in enumerate(base_ladder):
        attempt["attempt"] = i + 1

    return base_ladder


# ══════════════════════════════════════════════════════════════════
# LOAD REQUEST
# POST /api/v1/models/load — keep connection alive through warmup
# ══════════════════════════════════════════════════════════════════

def post_load_request(lm_id: str, result: dict):
    """
    Sends the load request and keeps the connection alive.

    CRITICAL: From Phi-4 logs, "cancelled by client disconnect"
    happens when we close this connection before warmup finishes.
    The POST returns 200 ONLY after warmup completes.
    So we must NOT timeout this request during warmup.
    """
    url     = f"{LM_STUDIO_BASE}/api/v1/models/load"
    payload = {"model": lm_id}

    log_info(f"POST {url}", {"model": lm_id})

    try:
        r = requests.post(
            url,
            json=payload,
            # Long timeout: load + warmup (up to 22s) + buffer
            timeout=LOAD_TIMEOUT + POST_DETECT_WARMUP_SECS + 30,
        )

        raw_body = r.text[:800] if r.text else ""
        log_info(f"Load response: HTTP {r.status_code}", {"body": raw_body})

        if r.status_code in (200, 201, 202):
            # Cache instance_id so unload_model can use it
            try:
                rdata = r.json() if r.text else {}
                iid = (
                    rdata.get("instance_id")
                    or rdata.get("data", {}).get("instance_id")
                )
                if iid:
                    _instance_id_cache[lm_id] = iid
                    log_info(f"Cached instance_id for {lm_id}: {iid}")
            except Exception:
                pass

            if not result.get("done"):
                result["status"]  = "ok"
                result["message"] = f"HTTP {r.status_code} (warmup complete)"
                result["done"]    = True
                log_success(f"Load POST returned {r.status_code} — warmup done")
        else:
            if not result.get("done"):
                result["status"]  = "error"
                result["message"] = f"HTTP {r.status_code}: {raw_body}"
                result["done"]    = True
                log_error(
                    f"Load POST failed: {r.status_code}",
                    {"model": lm_id, "body": raw_body}
                )

    except requests.exceptions.Timeout:
        if not result.get("done"):
            result["status"]  = "timeout"
            result["message"] = "Request timed out"
            result["done"]    = True
            log_error(f"Load POST timeout: {lm_id}")

    except Exception as e:
        if not result.get("done"):
            result["status"]  = "error"
            result["message"] = str(e)
            result["done"]    = True
            log_error(f"Load POST error: {e}")


# ══════════════════════════════════════════════════════════════════
# POLLING
# Detects model in /v1/models, then waits for POST to return 200
# ══════════════════════════════════════════════════════════════════

def poll_until_loaded(
    lm_id: str,
    result: dict,
    stop_event: threading.Event,
):
    """
    Polls /v1/models every 3s.

    Strategy:
        - When model detected: set warmup_detected + timestamp
        - Don't set done immediately (warmup still in progress)
        - POST thread will set done=True when it gets 200
        - If POST doesn't return within POST_DETECT_WARMUP_SECS,
          set done ourselves as a safety net
        - This keeps the HTTP connection alive through warmup
    """
    log_info(f"Polling for: {lm_id}")
    lm_name = lm_id.split("/")[-1].lower()

    while not stop_event.is_set():
        time.sleep(3)

        # If POST already finished (success or error), stop
        if result.get("done"):
            return

        try:
            loaded = get_loaded_model_ids()

            for mid in loaded:
                mid_lower = mid.lower()
                mid_name  = mid.split("/")[-1].lower()

                if mid_lower == lm_id.lower() or mid_name == lm_name:

                    if not result.get("warmup_detected"):
                        result["warmup_detected"]    = True
                        result["warmup_detected_at"] = time.time()
                        log_info(
                            f"Model visible: {mid}. "
                            f"Keeping connection alive for warmup..."
                        )
                        console.print(
                            f"[dim]  ✓ Model loading — "
                            f"keeping connection alive for warmup...[/dim]"
                        )

                    # Safety net: if POST still hasn't returned
                    wait_time = (
                        time.time()
                        - result.get("warmup_detected_at", time.time())
                    )
                    if wait_time > POST_DETECT_WARMUP_SECS:
                        if not result.get("done"):
                            log_warn(
                                f"POST didn't return after "
                                f"{POST_DETECT_WARMUP_SECS}s — "
                                "setting success from poll safety net"
                            )
                            result["status"]  = "ok"
                            result["message"] = (
                                f"Loaded (warmup safety net): {mid}"
                            )
                            result["done"] = True
                        return
                    break

        except Exception as e:
            log_warn(f"Poll error: {e}")


# ══════════════════════════════════════════════════════════════════
# RUN ONE ATTEMPT
# ══════════════════════════════════════════════════════════════════

def run_load_attempt(
    lm_id: str,
    attempt: dict,
    info: dict,
) -> dict:
    """
    One complete load attempt:
        1. Optionally unload (clears cached config in LM Studio)
        2. Snapshot file mtimes (to detect which config LM Studio reads)
        3. Write config to all known locations
        4. POST /api/v1/models/load (keep alive through warmup)
        5. Poll /v1/models to detect when model appears
        6. After success: find which config files changed
        7. Return result
    """
    result     = {
        "done":              False,
        "status":            None,
        "message":           "",
        "warmup_detected":   False,
        "warmup_detected_at": 0,
    }
    stop_event = threading.Event()

    # Step 1: Unload if strategy requires it
    if attempt.get("unload_first"):
        console.print(
            f"[dim]  → Unloading to force config re-read...[/dim]"
        )
        unload_model(lm_id)
        time.sleep(2)

    # Step 2: Snapshot mtimes before load
    before_mtimes = get_all_json_mtimes()

    # Step 3: Write config
    write_count, written_paths = write_config_everywhere(
        lm_id,
        attempt["context"],
        attempt["k_cache"],
        attempt["v_cache"],
        attempt["flash_attn"],
    )
    console.print(
        f"[dim]  Config: ctx={attempt['context']} "
        f"K={attempt['k_cache']} V={attempt['v_cache']} "
        f"FA={attempt['flash_attn']} "
        f"→ {write_count} location(s)[/dim]"
    )
    time.sleep(0.5)

    log_info(
        f"Attempt {attempt['attempt']}: {attempt['label']}",
        {
            "lm_id":      lm_id,
            "context":    attempt["context"],
            "k_cache":    attempt["k_cache"],
            "v_cache":    attempt["v_cache"],
            "flash_attn": attempt["flash_attn"],
        }
    )

    # Step 4+5: Load + poll concurrently
    load_thread = threading.Thread(
        target=post_load_request,
        args=(lm_id, result),
        daemon=True,
    )
    poll_thread = threading.Thread(
        target=poll_until_loaded,
        args=(lm_id, result, stop_event),
        daemon=True,
    )

    start = time.time()
    load_thread.start()
    time.sleep(0.5)
    poll_thread.start()

    # Progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading...", total=100)

        while not result["done"]:
            elapsed = int(time.time() - start)
            est     = (
                info.get("est_load_s", 60) + POST_DETECT_WARMUP_SECS
            )
            pct     = min(95, int(elapsed / est * 100))

            if result.get("warmup_detected"):
                eta_str = "warmup in progress..."
            else:
                remaining = max(0, est - elapsed)
                eta_str   = (
                    f"~{remaining}s remaining"
                    if elapsed < est
                    else "finishing..."
                )

            progress.update(
                task,
                completed=pct,
                description=(
                    f"Attempt {attempt['attempt']} — "
                    f"{attempt['label']} — "
                    f"{elapsed}s ({eta_str})"
                ),
            )
            time.sleep(0.5)

            # Hard safety timeout
            hard_limit = LOAD_TIMEOUT + POST_DETECT_WARMUP_SECS + 60
            if elapsed > hard_limit:
                if not result.get("done"):
                    result["status"]  = "timeout"
                    result["message"] = f"Hard timeout after {hard_limit}s"
                    result["done"]    = True
                    log_error(f"Hard timeout: {lm_id}")
                break

    stop_event.set()
    elapsed = int(time.time() - start)
    load_thread.join(timeout=10)
    poll_thread.join(timeout=5)
    result["elapsed"] = elapsed

    # Step 6: Discover which config files LM Studio actually touched
    if result["status"] == "ok":
        changed = find_changed_json_files(before_mtimes)
        if changed:
            log_success(
                f"Config files LM Studio used during load",
                {"files": [str(p) for p in changed]}
            )
            console.print(
                f"[dim]  🔍 Discovered {len(changed)} real config path(s)[/dim]"
            )
            result["confirmed_config_paths"] = changed
        else:
            result["confirmed_config_paths"] = []

    log_info(
        f"Attempt {attempt['attempt']}: "
        f"{result['status']} in {elapsed}s",
        {"message": result.get("message", "")}
    )
    return result


# ══════════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_error(error_message: str) -> dict:
    msg = error_message.lower()

    if (
        "v cache quantization requires flash attention" in msg
        or ("vcache" in msg and "flash" in msg)
        or "quantized v cache" in msg
    ):
        return {
            "cause":   "vcache_needs_flash_attn",
            "fix":     "config_not_read",
            "message": (
                "LM Studio using cached Q4 V cache. "
                "Trying unload+reload to force config re-read."
            ),
        }

    if any(x in msg for x in [
        "out of memory", "oom", "cuda out",
        "not enough memory", "failed to allocate",
    ]):
        return {
            "cause":   "out_of_memory",
            "fix":     "reduce_context",
            "message": "VRAM out of memory. Reducing context.",
        }

    if any(x in msg for x in [
        "not found", "no such file", "does not exist",
    ]):
        return {
            "cause":   "model_not_found",
            "fix":     "skip",
            "message": "Model not found on disk.",
        }

    if "missing required field" in msg and "instance_id" in msg:
        return {
            "cause":   "unload_api_schema",
            "fix":     "ignore",
            "message": "Unload needs instance_id — will fetch on next attempt.",
        }

    if "unrecognized key" in msg:
        return {
            "cause":   "api_schema",
            "fix":     "ignore",
            "message": "API schema mismatch — config via file only.",
        }

    return {
        "cause":   "generic",
        "fix":     "reduce_context",
        "message": f"Load failed: {error_message[:150]}",
    }


# ══════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# Sizes confirmed from logs where available
# ══════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    # ── Large hybrid: exceed 12GB VRAM ───────────────────────────
    "google/gemma-4-31b": {
        "name":         "Gemma 4 31B (CEO)",
        "role":         "CEO",
        "size_gb":      17.39,  # Confirmed: "17.39 GiB" from logs
        "gpu_fit":      False,
        "est_load_s":   30,     # Confirmed: ~3s load + warmup
        "retail_equiv": "GPT-4o",
        "tok_per_sec":  "4-6",
        "gpu_layers":   "27/61 on GPU",  # Confirmed from logs
        "known_issues": [
            "V Cache Quantization requires Flash Attention (confirmed)",
            "Unload + config rewrite strategy works (confirmed)",
            "Multimodal: loads vision encoder during warmup",
            "n_ctx=2048 confirmed working on 12GB VRAM",
        ],
    },
    "qwen/qwen3-coder-30b": {
        "name":         "Qwen3 Coder 30B (CTO)",
        "role":         "CTO",
        "size_gb":      18.6,
        "gpu_fit":      False,
        "est_load_s":   160,
        "retail_equiv": "Claude 3.5 Sonnet (Coding)",
        "tok_per_sec":  "4-7",
    },
    "deepseek/deepseek-r1-distill-qwen-32b": {
        "name":         "DeepSeek R1 Distill 32B (CFO)",
        "role":         "CFO",
        "size_gb":      19.9,
        "gpu_fit":      False,
        "est_load_s":   180,
        "retail_equiv": "OpenAI o1 Preview",
        "tok_per_sec":  "3-5",
        "special_note": "REASONING MODEL: <think> blocks are normal.",
    },
    "qwen/qwq-32b": {
        "name":         "QwQ 32B (CFO Backup)",
        "role":         "CFO_BACKUP",
        "size_gb":      19.9,
        "gpu_fit":      False,
        "est_load_s":   180,
        "retail_equiv": "OpenAI o1 Preview",
        "tok_per_sec":  "3-5",
        "special_note": "REASONING MODEL: <think> blocks are normal.",
    },
    "google/gemma-4-26b-a4b": {
        "name":         "Gemma 4 26B MoE (CPO)",
        "role":         "CPO",
        "size_gb":      18.0,
        "gpu_fit":      False,
        "est_load_s":   150,
        "retail_equiv": "Claude 3.5 Sonnet",
        "tok_per_sec":  "4-7",
        "known_issues": [
            "Same V cache Flash Attention issue as Gemma 4 31B",
        ],
    },
    # ── Medium: fit on 12GB GPU ───────────────────────────────────
    "microsoft/phi-4-reasoning-plus": {
        "name":         "Phi 4 Reasoning Plus (Backup)",
        "role":         "BACKUP",
        "size_gb":      8.43,   # Confirmed: "8.43 GiB" from logs
        "gpu_fit":      True,
        "est_load_s":   25,     # ~3s load + up to 22s warmup
        "retail_equiv": "OpenAI o1 Mini",
        "tok_per_sec":  "35-45",
        "gpu_layers":   "37/41 on GPU",  # Confirmed from logs
    },
    "qwen/qwen2.5-coder-14b-instruct": {
        "name":         "Qwen2.5 Coder 14B (COO)",
        "role":         "COO",
        "size_gb":      9.0,
        "gpu_fit":      True,
        "est_load_s":   20,
        "retail_equiv": "GitHub Copilot Pro",
        "tok_per_sec":  "35-45",
    },
    "qwen/qwen3.5-9b": {
        "name":         "Qwen3.5 9B (Backup)",
        "role":         "BACKUP",
        "size_gb":      6.5,
        "gpu_fit":      True,
        "est_load_s":   15,
        "retail_equiv": "GPT-4o Mini",
        "tok_per_sec":  "40-55",
    },
    "google/gemma-3-12b": {
        "name":         "Gemma 3 12B (Fallback)",
        "role":         "BACKUP",
        "size_gb":      8.2,
        "gpu_fit":      True,
        "est_load_s":   25,
        "retail_equiv": "GPT-4o Mini",
        "tok_per_sec":  "25-35",
    },
    # ── Small / fast ─────────────────────────────────────────────
    "google/gemma-4-e2b": {
        "name":         "Gemma 4 E2B (Autocomplete)",
        "role":         "AUTO",
        "size_gb":      4.11,
        "gpu_fit":      True,
        "est_load_s":   8,
        "retail_equiv": "Gemini 1.5 Flash",
        "tok_per_sec":  "80+",
    },
    "deepseek/deepseek-r1-0528-qwen3-8b": {
        "name":         "DeepSeek R1 0528 8B",
        "role":         "BACKUP",
        "size_gb":      5.0,
        "gpu_fit":      True,
        "est_load_s":   12,
        "retail_equiv": "GPT-4o Mini + Reasoning",
        "tok_per_sec":  "45-65",
    },
    # ── Embedding / Vision ────────────────────────────────────────
    "nomic-ai/nomic-embed-text-v1.5": {
        "name":         "Nomic Embed Text (Memory)",
        "role":         "EMBED",
        "size_gb":      0.08,
        "gpu_fit":      True,
        "est_load_s":   3,
        "retail_equiv": "OpenAI text-embedding-ada-002",
        "tok_per_sec":  "instant",
    },
    "qwen/qwen2.5-vl-7b-instruct": {
        "name":         "Qwen2.5 VL 7B (Vision)",
        "role":         "VISION",
        "size_gb":      7.0,
        "gpu_fit":      True,
        "est_load_s":   15,
        "retail_equiv": "GPT-4o Vision",
        "tok_per_sec":  "30-50",
    },
}


def get_model_info(model_id: str) -> dict:
    return MODEL_REGISTRY.get(model_id, {
        "name":         model_id,
        "role":         "UNKNOWN",
        "size_gb":      8,
        "gpu_fit":      True,
        "est_load_s":   30,
        "retail_equiv": "Unknown",
        "tok_per_sec":  "Unknown",
    })


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════

def print_model_card(model_id: str, learned: dict, lm_id: str):
    info         = get_model_info(model_id)
    special      = info.get("special_note")
    known_issues = info.get("known_issues", [])
    confirmed    = get_confirmed_config_paths(model_id)

    gpu_label = (
        "[green]✅ Full GPU (fits on 12GB)[/green]"
        if info.get("gpu_fit")
        else "[yellow]⚠️  GPU+RAM Hybrid (>12GB)[/yellow]"
    )

    console.print()
    console.print(f"[bold cyan]{'═' * 64}[/bold cyan]")
    console.print(f"[bold cyan]  MODEL      :[/bold cyan]  {info['name']}")
    console.print(f"[bold cyan]  ROLE       :[/bold cyan]  {info['role']}")
    console.print(f"[bold cyan]  SIZE       :[/bold cyan]  {info['size_gb']} GB")
    console.print(f"[bold cyan]  HARDWARE   :[/bold cyan]  {gpu_label}")
    console.print(f"[bold cyan]  GPU        :[/bold cyan]  {GPU_NAME}")
    console.print(f"[bold cyan]  LIKE       :[/bold cyan]  {info['retail_equiv']}")
    console.print(
        f"[bold cyan]  SPEED      :[/bold cyan]  "
        f"~{info.get('tok_per_sec','?')} tok/s"
    )
    console.print(f"[bold cyan]  EST. LOAD  :[/bold cyan]  ~{info['est_load_s']}s")

    if lm_id and lm_id != model_id:
        console.print(
            f"[bold cyan]  LM ID      :[/bold cyan]  "
            f"[dim]{lm_id} (resolved)[/dim]"
        )

    if info.get("gpu_layers"):
        console.print(
            f"[bold cyan]  GPU LAYERS :[/bold cyan]  "
            f"[dim]{info['gpu_layers']}[/dim]"
        )

    if known_issues:
        console.print(f"[bold cyan]{'─' * 64}[/bold cyan]")
        console.print("[bold yellow]  KNOWN ISSUES:[/bold yellow]")
        for issue in known_issues:
            console.print(f"[yellow]    • {issue}[/yellow]")

    if confirmed:
        console.print(f"[bold cyan]{'─' * 64}[/bold cyan]")
        console.print(
            f"[bold green]  CONFIG PATH CONFIRMED:[/bold green] "
            f"[green]{confirmed[0].name}[/green]"
        )

    if learned:
        console.print(f"[bold cyan]{'─' * 64}[/bold cyan]")
        console.print("[bold green]  LEARNED SETTINGS:[/bold green]")
        console.print(
            f"[green]    Context    → {learned['context']} tokens[/green]"
        )
        console.print(
            f"[green]    K Cache    → {learned.get('k_cache','?')}[/green]"
        )
        console.print(
            f"[green]    V Cache    → {learned.get('v_cache','?')}[/green]"
        )
        console.print(
            f"[green]    Flash Attn → {learned.get('flash_attn','?')}[/green]"
        )
        console.print(
            f"[green]    Used       → {learned.get('success_count',1)}x[/green]"
        )
        console.print(
            f"[green]    Last load  → {learned.get('last_success','?')}[/green]"
        )
    else:
        console.print(f"[bold cyan]{'─' * 64}[/bold cyan]")
        console.print(
            "[dim]  No learned settings yet — discovering now...[/dim]"
        )

    if special:
        console.print(f"[bold cyan]{'─' * 64}[/bold cyan]")
        console.print(f"[bold yellow]  NOTE: {special}[/bold yellow]")

    console.print(f"[bold cyan]{'═' * 64}[/bold cyan]")
    console.print()
def is_model_already_loaded_correctly(lm_id: str) -> bool:
    """
    Uses /api/v0/models which has the state field.
    Confirmed working from your diagnostic output.
    """
    try:
        resp = requests.get(
            f"{LM_STUDIO_BASE}/api/v0/models",
            timeout=5,
        )
        if resp.status_code != 200:
            return False

        models = resp.json().get("data", [])
        lm_lower = lm_id.lower()
        lm_name  = lm_id.split("/")[-1].lower()

        for m in models:
            mid   = m.get("id", "")
            state = m.get("state", "not-loaded")

            # Skip duplicate instances
            if ":" in mid and mid.split(":")[-1].isdigit():
                continue

            mid_lower = mid.lower()
            mid_name  = mid.split("/")[-1].lower()

            if (
                mid_lower == lm_lower
                or mid_name == lm_name
            ) and state == "loaded":
                return True

    except Exception as e:
        log_warn(f"Could not check /api/v0/models: {e}")

    return False

def is_lm_studio_up() -> bool:
    try:
        r = requests.get(f"{LM_STUDIO_BASE}/v1/models", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# MAIN SMART LOADER
# Fully automatic — never asks user to do anything manually
# ══════════════════════════════════════════════════════════════════

def wait_for_model(model_id: str) -> bool:
    """
    Full automatic loading sequence.
    Tries every setting combination, then falls back to another model.
    Never tells the user to load manually.
    """
    log_separator(f"LOAD: {model_id}")
    log_info(f"GPU: {GPU_NAME} | VRAM: {GPU_VRAM_GB}GB")
    log_info(f"Warmup buffer: {POST_DETECT_WARMUP_SECS}s")

    # ── LM Studio reachable? ──────────────────────────────────────
    if not is_lm_studio_up():
        console.print(
            "[red]✗ LM Studio not reachable on port 1234[/red]\n"
            "[yellow]  → Open LM Studio "
            "→ Local Server → Start Server[/yellow]"
        )
        log_error("LM Studio unreachable")
        return False

    # ── Discover available models ─────────────────────────────────
    available_ids = get_lm_studio_model_ids()

    # ── Resolve our ID to LM Studio's actual ID ───────────────────
    lm_studio_id = resolve_model_id(model_id, available_ids)

    if lm_studio_id != model_id:
        console.print(
            f"[dim]  Resolved: {model_id} "
            f"→ [bold]{lm_studio_id}[/bold][/dim]"
        )

    # ── Load learned settings ─────────────────────────────────────
    learned = get_learned_setting(model_id)
    info    = get_model_info(model_id)

    print_model_card(model_id, learned, lm_studio_id)

    # ── CHECK IF ALREADY LOADED ───────────────────────────────────
    # Uses /v1/models (reliable) — /api/v0/models state field is NOT reliable
    console.print(
        f"[dim]  Checking /v1/models for loaded instances...[/dim]"
    )

    # First clean up any duplicate instances
    cleanup_duplicate_instances(lm_studio_id)

    if is_model_already_loaded_correctly(lm_studio_id):
        console.print(
            f"\n[bold green]✓ Already loaded![/bold green]"
        )
        console.print(
            f"[green]  {info['name']} is already in memory[/green]"
        )
        console.print(
            f"[green]  Skipping load — no action needed[/green]\n"
        )
        log_info(
            f"Skipped load — already in memory: {lm_studio_id}"
        )
        console.print("[bold green]Model ready.[/bold green]\n")
        return True

    console.print(
        f"[dim]  Not loaded — proceeding with load sequence[/dim]\n"
    )

    # ── Build attempt ladder ──────────────────────────────────────
    ladder  = build_attempt_ladder(model_id, learned)
    success = False

    confirmed_count = len(get_confirmed_config_paths(model_id))
    console.print(
        f"[bold cyan]Smart Loader:[/bold cyan] "
        f"{len(ladder)} combinations — fully automatic"
    )
    if confirmed_count:
        console.print(
            f"[dim]  Config path confirmed from previous load[/dim]"
        )
    console.print(f"[dim]  Log: {LOG_FILE}[/dim]\n")

    # ── Attempt loop ──────────────────────────────────────────────
    for attempt in ladder:
        attempt_num = attempt["attempt"]
        context     = attempt["context"]
        k_cache     = attempt["k_cache"]
        v_cache     = attempt["v_cache"]
        flash_attn  = attempt["flash_attn"]
        label       = attempt["label"]
        is_learned  = attempt.get("is_learned", False)
        note        = attempt.get("note", "")

        if is_learned:
            console.print(
                f"[bold green]Attempt {attempt_num}/{len(ladder)}:[/bold green] "
                f"[green]{label}[/green]"
            )
        else:
            console.print(
                f"[bold]Attempt {attempt_num}/{len(ladder)}:[/bold] {label}"
            )

        if note:
            console.print(f"[dim]  ℹ {note}[/dim]")

        result  = run_load_attempt(lm_studio_id, attempt, info)
        elapsed = result["elapsed"]

        if result["status"] == "ok":
            confirmed_paths = result.get("confirmed_config_paths", [])

            log_success(f"LOADED: {model_id}", {
                "lm_id":        lm_studio_id,
                "context":      context,
                "k_cache":      k_cache,
                "v_cache":      v_cache,
                "flash_attn":   flash_attn,
                "elapsed_s":    elapsed,
                "attempt":      attempt_num,
                "config_paths": len(confirmed_paths),
            })

            console.print(f"\n[bold green]✅ Loaded![/bold green]")
            console.print(f"[green]  Model      : {info['name']}[/green]")
            console.print(f"[green]  LM ID      : {lm_studio_id}[/green]")
            console.print(f"[green]  Context    : {context} tokens[/green]")
            console.print(f"[green]  K Cache    : {k_cache}[/green]")
            console.print(f"[green]  V Cache    : {v_cache}[/green]")
            console.print(f"[green]  Flash Attn : {flash_attn}[/green]")
            console.print(f"[green]  Time       : {elapsed}s[/green]")
            console.print(
                f"[green]  Speed      : ~{info['tok_per_sec']} tok/s[/green]"
            )
            console.print(
                f"[green]  Like       : {info['retail_equiv']}[/green]"
            )
            if confirmed_paths:
                console.print(
                    f"[green]  Config     : {confirmed_paths[0].name}[/green]"
                )
            console.print(f"[green]  Log        : {LOG_FILE}[/green]\n")

            if not is_learned:
                console.print(
                    "[green]  ✨ Optimal settings discovered "
                    "and saved![/green]\n"
                )

            save_learned_setting(
                model_id, context, k_cache, v_cache,
                elapsed, flash_attn,
                config_paths=confirmed_paths,
            )
            _record_load_time(
                model_id, elapsed, context, k_cache, v_cache
            )
            success = True
            break

        else:
            error_analysis = analyze_error(result.get("message", ""))
            log_error(
                f"Attempt {attempt_num} failed: {error_analysis['cause']}",
                {
                    "raw":     result.get("message", ""),
                    "context": context,
                }
            )
            console.print(
                f"[yellow]  ✗ {error_analysis['message']}[/yellow]"
            )

            if error_analysis["fix"] == "skip":
                console.print(
                    "[red]  → Model not found. Moving to fallback.[/red]\n"
                )
                break

            if attempt_num < len(ladder):
                console.print(
                    "[cyan]  → Auto-adjusting and retrying...[/cyan]\n"
                )
                time.sleep(2)

    # ── Automatic fallback — no manual prompt ─────────────────────
    if not success:
        console.print(
            f"\n[yellow]⚡ Primary failed — "
            f"automatically trying fallback...[/yellow]"
        )
        log_warn(f"All attempts failed: {model_id}")

        fallback_id = cfg.fallback_model
        if fallback_id and fallback_id != model_id:
            fb_lm_id         = resolve_model_id(fallback_id, available_ids)
            fallback_info    = get_model_info(fallback_id)
            fallback_learned = get_learned_setting(fallback_id)
            fallback_ladder  = build_attempt_ladder(
                fallback_id, fallback_learned
            )

            console.print(
                f"[bold yellow]  Fallback: "
                f"{fallback_info['name']}[/bold yellow]"
            )
            log_info(f"Trying fallback: {fallback_id} ({fb_lm_id})")

            # Try ALL fallback attempts — not just the first
            for fb_attempt in fallback_ladder:
                fb_result = run_load_attempt(
                    fb_lm_id, fb_attempt, fallback_info
                )
                if fb_result["status"] == "ok":
                    log_success(f"Fallback loaded: {fallback_id}")
                    console.print(
                        f"\n[bold yellow]⚡ Running on fallback: "
                        f"{fallback_info['name']}[/bold yellow]"
                    )
                    console.print(
                        f"[yellow]  {fallback_info['retail_equiv']} "
                        f"instead of "
                        f"{info['retail_equiv']}[/yellow]\n"
                    )
                    save_learned_setting(
                        fallback_id,
                        fb_attempt["context"],
                        fb_attempt["k_cache"],
                        fb_attempt["v_cache"],
                        fb_result["elapsed"],
                        fb_attempt["flash_attn"],
                        config_paths=fb_result.get("confirmed_config_paths"),
                    )
                    return True

                log_warn(
                    f"Fallback attempt "
                    f"{fb_attempt['attempt']} failed"
                )
                time.sleep(1)

        # Last resort: use whatever is already loaded
        already_loaded = get_loaded_model_ids()
        if already_loaded:
            mid = already_loaded[0]
            console.print(
                f"\n[yellow]  Using already-loaded: {mid}[/yellow]\n"
            )
            log_info(f"Using already-loaded: {mid}")
            return True

        # Nothing worked — bridge starts anyway
        console.print(
            "\n[yellow]  No model loaded successfully.[/yellow]"
        )
        console.print(
            "[yellow]  Bridge starting — requests will retry "
            "when a model becomes available.[/yellow]\n"
        )
        log_warn("No model loaded. Bridge starting without active model.")
        return False

    return success


# ══════════════════════════════════════════════════════════════════
# PERFORMANCE RECORDING
# ══════════════════════════════════════════════════════════════════

def _record_load_time(
    model_id: str,
    seconds: int,
    context: int = 0,
    k_cache: str = "unknown",
    v_cache: str = "unknown",
):
    path = os.path.join(
        os.path.dirname(__file__), cfg.performance_file
    )
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}

    data.setdefault("load_times", {})
    info = get_model_info(model_id)

    data["load_times"][model_id] = {
        "last_load_seconds": seconds,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role":              info["role"],
        "retail_equiv":      info["retail_equiv"],
        "gpu_fit":           info.get("gpu_fit"),
        "tok_per_sec":       info["tok_per_sec"],
        "gpu":               GPU_NAME,
        "vram_gb":           GPU_VRAM_GB,
        "context_used":      context,
        "k_cache_used":      k_cache,
        "v_cache_used":      v_cache,
    }

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log_info(f"Performance recorded: {model_id}", {
            "seconds": seconds, "context": context
        })
    except Exception as e:
        log_warn(f"Cannot write performance file: {e}")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    target = (
        sys.argv[1]
        if len(sys.argv) > 1
        else cfg.ceo_model
    )

    log_separator(
        f"SESSION START — "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    log_info(f"Python: {sys.version.split()[0]}")
    log_info(f"Target: {target}")
    log_info(f"GPU: {GPU_NAME} ({GPU_VRAM_GB}GB)")
    log_info(f"Timeout: {LOAD_TIMEOUT}s")
    log_info(f"Warmup buffer: {POST_DETECT_WARMUP_SECS}s")

    success = wait_for_model(target)

    if success:
        console.print("[bold green]Model ready.[/bold green]\n")
        log_success("Session complete — model ready")
    else:
        console.print(
            "[dim]Bridge starting — model will be used "
            "when available.[/dim]\n"
        )
        log_warn("Session ended without confirmed model load")