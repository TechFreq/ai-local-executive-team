# swarm_bridge_server.py
# ══════════════════════════════════════════════════════════════════
# AI Executive Team Bridge Server v2.0.3
#
# Speaks BOTH OpenAI and Ollama API formats so it works with:
#   - OpenWebUI (tries Ollama endpoints first)
#   - Continue VS Code extension (uses OpenAI format)
#   - Any OpenAI-compatible client
#
# v2.0.3 changes:
#   - Progress bar during generation wait
#   - Live token counter while streaming
#   - Chat request headers with counter
#   - Animated "Ready" waiting indicator
#   - Abort confirmation messages
#   - O key = optimize model, S key = status
#   - Role-colored output for all executives
# ══════════════════════════════════════════════════════════════════

import os
import re
import sys
import time
import json
import hashlib
import threading
import subprocess
import msvcrt
from pathlib import Path
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from rich.console import Console

from core.config_loader import cfg
from routers.local_router import (
    call_local_stream_safe,
    KEEPALIVE_SIGNAL,
    TIMEOUT_SIGNAL,
    check_lm_studio_health,
    list_local_models,
    get_loaded_model_status,
    ensure_model_loaded,
    FIRST_TOKEN_TIMEOUT,
    signal_abort,
    clear_abort,
    get_base_url,
)
from routers.hybrid_router import detect_intent
from model_performance_log import (
    record_generation,
    record_timeout,
    record_ttft,
    get_suggested_timeout,
    get_timeout_rate,
    get_reliability_score,
    print_performance_report,
    load_log,
)

load_dotenv()
console = Console()

app  = Flask(__name__)
CORS(app)

PORT = cfg.bridge_port
HOST = cfg.bridge_host

# Global request counter
_request_counter = 0

# Learned settings file — written by load_model.py tuner
_LEARNED_SETTINGS_FILE = Path("learned_settings.json")


def _read_learned_settings() -> dict:
    """Read tuner results without importing load_model (avoids heavy init)."""
    try:
        if _LEARNED_SETTINGS_FILE.exists():
            with open(_LEARNED_SETTINGS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════════
# RESPONSE CACHE
# Caches full text responses for identical (route, message) pairs.
# Only caches short / casual / repeat queries — never full board.
# TTL: 5 minutes. Max 128 entries (LRU eviction).
# ══════════════════════════════════════════════════════════════════

_RESPONSE_CACHE_TTL     = 300          # seconds
_RESPONSE_CACHE_MAXSIZE = 128

# { cache_key: (response_text, expiry_monotonic) }
_response_cache: dict[str, tuple[str, float]] = {}
_response_cache_lock = threading.Lock()


def _cache_key(route: str, message: str) -> str:
    return f"{route}:{message.strip().lower()}"


def _get_cached(route: str, message: str) -> str | None:
    key = _cache_key(route, message)
    now = time.monotonic()
    with _response_cache_lock:
        entry = _response_cache.get(key)
        if entry and now < entry[1]:
            return entry[0]
        if entry:
            del _response_cache[key]   # expired
    return None


def _set_cached(route: str, message: str, response: str) -> None:
    key    = _cache_key(route, message)
    expiry = time.monotonic() + _RESPONSE_CACHE_TTL
    with _response_cache_lock:
        if len(_response_cache) >= _RESPONSE_CACHE_MAXSIZE:
            oldest_key = next(iter(_response_cache))
            del _response_cache[oldest_key]
        _response_cache[key] = (response, expiry)


def _is_cacheable(route: str, message: str) -> bool:
    """Only cache cheap/repeatable routes — never full board meetings."""
    if route == "full_board":
        return False
    if len(message.strip()) < 120 or is_casual_message(message):
        return True
    return False


# ══════════════════════════════════════════════════════════════════
# PER-MODEL TIMEOUTS
# ══════════════════════════════════════════════════════════════════

MODEL_TIMEOUTS: dict[str, int] = {
    "google/gemma-4-31b":                    180,
    "qwen/qwen3-coder-30b":                  180,
    "deepseek/deepseek-r1-distill-qwen-32b": 180,
    "google/gemma-4-26b-a4b":                180,
    "qwen/qwq-32b":                          180,
    "qwen/qwen2.5-coder-14b-instruct":        90,
    "microsoft/phi-4-reasoning-plus":          90,
    "deepseek/deepseek-r1-0528-qwen3-8b":     90,
    "qwen/qwen3.5-9b":                         90,
    "google/gemma-3-12b":                      60,
    "qwen/qwen2.5-vl-7b-instruct":             60,
    "google/gemma-4-e2b":                      30,
    "nomic-ai/nomic-embed-text-v1.5":          30,
}

def get_model_timeout(model_id: str) -> int:
    """
    Returns timeout for a model. Priority:
      1. Hardcoded MODEL_TIMEOUTS (explicit override)
      2. Learned timeout from performance log (avg TTFT × 4)
      3. Global FIRST_TOKEN_TIMEOUT default
    """
    if model_id in MODEL_TIMEOUTS:
        hardcoded = MODEL_TIMEOUTS[model_id]
        learned   = get_suggested_timeout(model_id, floor=hardcoded)
        if learned and learned > hardcoded:
            # Model is taking longer than expected — use learned value
            return learned
        return hardcoded

    learned = get_suggested_timeout(model_id)
    return learned if learned else FIRST_TOKEN_TIMEOUT


# ══════════════════════════════════════════════════════════════════
# CASUAL MESSAGE DETECTION
# ══════════════════════════════════════════════════════════════════

_CASUAL_PATTERN_LIST: list[str] = [
    "hey", "hi", "hello", "howdy", "hiya",
    "how are you", "how r you", "how's it going",
    "good morning", "good afternoon", "good evening", "good night",
    "what's up", "whats up", "sup",
    "thanks", "thank you", "cheers", "ty",
    "ok", "okay", "got it", "sounds good", "sure",
    "bye", "goodbye", "see you", "later",
    "lol", "haha", "nice", "cool", "great",
    "who are you", "what are you", "what can you do",
    "tell me about yourself",
    "check the", "read the", "look at", "open the",
    "what does the", "what's in the", "whats in the",
    "can you check", "can you read", "can you look",
    "show me the", "summarize the", "what is in",
    "explain the", "describe the", "review the",
    "license", "readme", "changelog", "todo", "notes",
    "fix this", "fix the", "what's wrong", "whats wrong",
    "explain this", "explain that", "what does this do",
    "add a comment", "format this", "clean this up",
]

# Pre-compiled regex — much faster than a linear for-loop scan
_CASUAL_RE = re.compile(
    "|".join(re.escape(p) for p in _CASUAL_PATTERN_LIST),
    re.IGNORECASE,
)


def is_casual_message(message: str) -> bool:
    stripped = message.strip()
    if len(stripped) < 40:
        return True
    return bool(_CASUAL_RE.search(stripped))


# ══════════════════════════════════════════════════════════════════
# OPENWEBUI SYSTEM REQUEST DETECTION
# Fires BEFORE preflight so auto-suggest requests never touch
# the model — they compete with real user requests for VRAM.
# ══════════════════════════════════════════════════════════════════

_OPENWEBUI_AUTOSUGGEST_PREFIX = "### Task:"
_OPENWEBUI_AUTOSUGGEST_MARKER = "follow-up questions"

# Covers additional OpenWebUI background tasks (title gen, etc.)
_OPENWEBUI_SYSTEM_PREFIXES = (
    "### Task:",
    "Create a concise, 3-5 word title",
    "Generate a title for",
    "You are an AI assistant that generates",
)


def _is_openwebui_system_request(message: str) -> bool:
    """
    Returns True for OpenWebUI's automatic background requests:
      - Follow-up question suggestions  ("### Task: Suggest 3-5...")
      - Chat title generation           ("Create a concise, 3-5 word title...")
      - Any other known system prefixes

    Short-circuiting these prevents them from occupying the model
    while a real user request is waiting.
    """
    stripped = message.strip()
    # Primary: follow-up suggestion block
    if (
        stripped.startswith(_OPENWEBUI_AUTOSUGGEST_PREFIX)
        and _OPENWEBUI_AUTOSUGGEST_MARKER in stripped
    ):
        return True
    # Secondary: other known OpenWebUI background task prefixes
    return any(stripped.startswith(p) for p in _OPENWEBUI_SYSTEM_PREFIXES)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def stable_digest(model_id: str) -> str:
    h = hashlib.sha256(model_id.encode()).hexdigest()
    return f"sha256:{h}"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def is_client_connected() -> bool:
    try:
        return not request.environ.get("werkzeug.request_ended", False)
    except RuntimeError:
        return True


# ══════════════════════════════════════════════════════════════════
# ROLE COLORS
# ══════════════════════════════════════════════════════════════════

ROLE_COLORS = {
    "CEO":       "bold yellow",
    "CEO Final": "bold yellow",
    "CTO":       "bold blue",
    "CFO":       "bold green",
    "CPO":       "bold magenta",
    "COO":       "bold cyan",
    "VISION":    "bold white",
    "Assistant": "bold bright_white",
}


# ══════════════════════════════════════════════════════════════════
# ANIMATED WAITING INDICATOR
# ══════════════════════════════════════════════════════════════════

def _print_waiting():
    console.print()
    console.print(
        "[dim]  ═══════════════════════════════════════════════[/dim]"
    )
    console.print(
        "[bold cyan]  ◉ Ready — waiting for next chat request[/bold cyan]"
    )
    console.print(
        "[dim]  ═══════════════════════════════════════════════[/dim]"
    )
    console.print()


# ══════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════

BRIDGE_MODEL_LIST = [
    {
        "id":             "executive-swarm",
        "name":           "🏛️ Executive Swarm — Full Board",
        "description":    "CEO + CTO + CFO + CPO + COO collaborate",
        "context_length": 8192,
        "capabilities":   ["chat"],
        "route":          "full_board",
        "size_label":     "Multi-Model",
    },
    {
        "id":             "google/gemma-4-31b",
        "name":           "👑 CEO — Gemma 4 31B (GPT-4o)",
        "description":    "Strategic decisions. Multimodal.",
        "context_length": 2048,
        "capabilities":   ["chat", "vision"],
        "route":          "ceo",
        "size_label":     "31B",
    },
    {
        "id":             "qwen/qwen3-coder-30b",
        "name":           "⚙️ CTO — Qwen3 Coder 30B (Claude Sonnet)",
        "description":    "Architecture, code, security",
        "context_length": 8192,
        "capabilities":   ["chat"],
        "route":          "cto",
        "size_label":     "30B",
    },
    {
        "id":             "deepseek/deepseek-r1-distill-qwen-32b",
        "name":           "📊 CFO — DeepSeek R1 32B (o1 Preview)",
        "description":    "Cost, risk, deep reasoning",
        "context_length": 8192,
        "capabilities":   ["chat"],
        "route":          "cfo",
        "size_label":     "32B",
    },
    {
        "id":             "google/gemma-4-26b-a4b",
        "name":           "🎯 CPO — Gemma 4 26B MoE (Claude Sonnet)",
        "description":    "Product strategy, UX, features",
        "context_length": 4096,
        "capabilities":   ["chat"],
        "route":          "cpo",
        "size_label":     "26B",
    },
    {
        "id":             "qwen/qwen2.5-coder-14b-instruct",
        "name":           "📋 COO — Qwen2.5 Coder 14B (Copilot Pro)",
        "description":    "Tasks, timelines, execution",
        "context_length": 16384,
        "capabilities":   ["chat"],
        "route":          "coo",
        "size_label":     "14B",
    },
    {
        "id":             "microsoft/phi-4-reasoning-plus",
        "name":           "🧠 Phi 4 Reasoning Plus (o1 Mini)",
        "description":    "Fast deep reasoning on 12GB VRAM",
        "context_length": 16384,
        "capabilities":   ["chat"],
        "route":          "direct",
        "size_label":     "14B",
    },
    {
        "id":             "qwen/qwq-32b",
        "name":           "⚡ QwQ 32B Reasoning (o1 Preview)",
        "description":    "Extended reasoning with think blocks",
        "context_length": 8192,
        "capabilities":   ["chat"],
        "route":          "direct",
        "size_label":     "32B",
    },
    {
        "id":             "deepseek/deepseek-r1-0528-qwen3-8b",
        "name":           "🔍 DeepSeek R1 8B Fast (GPT-4o Mini + Reasoning)",
        "description":    "Fast reasoning, full GPU on 12GB",
        "context_length": 16384,
        "capabilities":   ["chat"],
        "route":          "direct",
        "size_label":     "8B",
    },
    {
        "id":             "qwen/qwen3.5-9b",
        "name":           "🌐 Qwen3.5 9B (GPT-4o Mini)",
        "description":    "Fast general purpose",
        "context_length": 16384,
        "capabilities":   ["chat"],
        "route":          "direct",
        "size_label":     "9B",
    },
    {
        "id":             "google/gemma-3-12b",
        "name":           "🏠 Gemma 3 12B (GPT-4o Mini)",
        "description":    "Reliable fallback",
        "context_length": 8192,
        "capabilities":   ["chat"],
        "route":          "direct",
        "size_label":     "12B",
    },
    {
        "id":             "qwen/qwen2.5-vl-7b-instruct",
        "name":           "👁️ Qwen2.5 VL 7B Vision (GPT-4o Vision)",
        "description":    "Image understanding and analysis",
        "context_length": 8192,
        "capabilities":   ["chat", "vision"],
        "route":          "vision",
        "size_label":     "7B",
    },
    {
        "id":             "google/gemma-4-e2b",
        "name":           "💨 Gemma 4 E2B (Gemini Flash)",
        "description":    "Tiny and fast — 80+ tok/s autocomplete",
        "context_length": 8192,
        "capabilities":   ["chat"],
        "route":          "direct",
        "size_label":     "2B",
    },
    {
        "id":             "nomic-ai/nomic-embed-text-v1.5",
        "name":           "🔗 Nomic Embed Text (Embeddings)",
        "description":    "Vector embeddings for RAG and memory",
        "context_length": 8192,
        "capabilities":   ["embeddings"],
        "route":          "direct",
        "size_label":     "137M",
    },
]

_MODEL_MAP           = {m["id"]: m for m in BRIDGE_MODEL_LIST}
_ALL_KNOWN_MODEL_IDS = {m["id"] for m in BRIDGE_MODEL_LIST}

_EXECUTIVE_ROUTES = {
    "executive-swarm":                       "full_board",
    "google/gemma-4-31b":                    "ceo",
    "qwen/qwen3-coder-30b":                  "cto",
    "deepseek/deepseek-r1-distill-qwen-32b": "cfo",
    "google/gemma-4-26b-a4b":                "cpo",
    "qwen/qwen2.5-coder-14b-instruct":       "coo",
    "qwen/qwen2.5-vl-7b-instruct":           "vision",
}


# ══════════════════════════════════════════════════════════════════
# NAME MAPS
# ══════════════════════════════════════════════════════════════════

MODEL_NAME_MAP = {
    "executive-swarm":     "executive-swarm",
    "gemma4-31b":          "google/gemma-4-31b",
    "qwen3-coder-30b":     "qwen/qwen3-coder-30b",
    "deepseek-r1-32b":     "deepseek/deepseek-r1-distill-qwen-32b",
    "gemma4-26b-a4b":      "google/gemma-4-26b-a4b",
    "qwen2.5-coder-14b":   "qwen/qwen2.5-coder-14b-instruct",
    "phi4-reasoning-plus": "microsoft/phi-4-reasoning-plus",
    "qwq-32b":             "qwen/qwq-32b",
    "deepseek-r1-8b":      "deepseek/deepseek-r1-0528-qwen3-8b",
    "qwen3.5-9b":          "qwen/qwen3.5-9b",
    "gemma3-12b":          "google/gemma-3-12b",
    "qwen2.5-vl-7b":       "qwen/qwen2.5-vl-7b-instruct",
    "gemma4-2b":           "google/gemma-4-e2b",
    "nomic-embed-text":    "nomic-ai/nomic-embed-text-v1.5",
}

MODEL_NAME_MAP_REVERSE = {v: k for k, v in MODEL_NAME_MAP.items()}

_OLLAMA_META = {
    "executive-swarm":     {"family": "llama",      "families": ["llama"],      "size": 19000000000, "parameter_size": "32B"},
    "gemma4-31b":          {"family": "gemma",      "families": ["gemma"],      "size": 18600000000, "parameter_size": "31B"},
    "qwen3-coder-30b":     {"family": "qwen2",      "families": ["qwen2"],      "size": 18000000000, "parameter_size": "30B"},
    "deepseek-r1-32b":     {"family": "qwen2",      "families": ["qwen2"],      "size": 19200000000, "parameter_size": "32B"},
    "gemma4-26b-a4b":      {"family": "gemma",      "families": ["gemma"],      "size": 15600000000, "parameter_size": "26B"},
    "qwen2.5-coder-14b":   {"family": "qwen2",      "families": ["qwen2"],      "size": 8400000000,  "parameter_size": "14B"},
    "phi4-reasoning-plus": {"family": "phi",        "families": ["phi"],        "size": 8400000000,  "parameter_size": "14B"},
    "qwq-32b":             {"family": "qwen2",      "families": ["qwen2"],      "size": 19200000000, "parameter_size": "32B"},
    "deepseek-r1-8b":      {"family": "qwen2",      "families": ["qwen2"],      "size": 4800000000,  "parameter_size": "8B"},
    "qwen3.5-9b":          {"family": "qwen2",      "families": ["qwen2"],      "size": 5400000000,  "parameter_size": "9B"},
    "gemma3-12b":          {"family": "gemma",      "families": ["gemma"],      "size": 7200000000,  "parameter_size": "12B"},
    "qwen2.5-vl-7b":       {"family": "qwen2",      "families": ["qwen2"],      "size": 4200000000,  "parameter_size": "7B"},
    "gemma4-2b":           {"family": "gemma",      "families": ["gemma"],      "size": 1200000000,  "parameter_size": "2B"},
    "nomic-embed-text":    {"family": "nomic-bert", "families": ["nomic-bert"], "size": 274000000,   "parameter_size": "137M"},
}


# ══════════════════════════════════════════════════════════════════
# ROUTING HELPERS
# ══════════════════════════════════════════════════════════════════

def get_model_route(model_id: str) -> str:
    if not model_id:
        return "full_board"
    real_id = MODEL_NAME_MAP.get(model_id, model_id)
    if real_id in _EXECUTIVE_ROUTES:
        return _EXECUTIVE_ROUTES[real_id]
    if real_id in _MODEL_MAP:
        return _MODEL_MAP[real_id].get("route", "direct")
    low = real_id.lower()
    if "gemma-4-31b"         in low: return "ceo"
    if "qwen3-coder"         in low: return "cto"
    if "deepseek-r1-distill" in low: return "cfo"
    if "gemma-4-26b"         in low: return "cpo"
    if "qwen2.5-coder"       in low: return "coo"
    if "vl-7b"               in low: return "vision"
    if "swarm"               in low: return "full_board"
    return "direct"


def get_actual_model(model_id: str) -> str:
    real_id = MODEL_NAME_MAP.get(model_id, model_id)
    route   = get_model_route(model_id)
    return {
        "ceo":        cfg.ceo_model,
        "cto":        cfg.cto_model,
        "cfo":        cfg.cfo_model,
        "cpo":        cfg.cpo_model,
        "coo":        cfg.coo_model,
        "vision":     cfg.vision_model,
        "full_board": cfg.ceo_model,
        "direct":     real_id,
    }.get(route, real_id)


# ══════════════════════════════════════════════════════════════════
# PRE-FLIGHT SYSTEM
# ══════════════════════════════════════════════════════════════════

class PreFlightResult:
    def __init__(
        self,
        ok: bool,
        model_to_use: str | None = None,
        status: str = "",
        warning: str = "",
        error: str = "",
        unknown_model: str = "",
        suggestion: str = "",
    ):
        self.ok            = ok
        self.model_to_use  = model_to_use
        self.status        = status
        self.warning       = warning
        self.error         = error
        self.unknown_model = unknown_model
        self.suggestion    = suggestion


def run_preflight(requested_model: str, route: str) -> PreFlightResult:

    lm_up = check_lm_studio_health()
    if not lm_up:
        return PreFlightResult(
            ok=False,
            error=(
                "❌ **LM Studio is not running or its local server is off.**\n\n"
                "**To fix this:**\n"
                "1. Open **LM Studio**\n"
                "2. Click the **↔ Local Server** tab on the left\n"
                "3. Click **Start Server**\n"
                "4. Wait for the green **Running** indicator\n"
                "5. Then resend your message\n\n"
                "*The bridge cannot proceed without LM Studio running.*"
            ),
        )

    console.print("[green]  ✓ LM Studio is running[/green]")

    if route == "full_board":
        return PreFlightResult(
            ok=True,
            model_to_use=requested_model,
            status="Full board mode — model swaps handled per executive",
        )

    loaded_status = get_loaded_model_status()
    loaded_model  = loaded_status.get("primary")
    target_model  = get_actual_model(requested_model)

    console.print(f"[dim]  Target model: {target_model}[/dim]")

    if loaded_model is None:
        console.print(
            f"[yellow]  Nothing loaded → Loading {target_model}...[/yellow]"
        )
        ensure_model_loaded(target_model)
        return PreFlightResult(
            ok=True,
            model_to_use=target_model,
            status=f"Loaded {target_model} (nothing was loaded)",
        )

    console.print(f"[dim]  LM Studio has loaded: {loaded_model}[/dim]")

    unknown_warning = ""
    suggestion_text = ""

    if loaded_model not in _ALL_KNOWN_MODEL_IDS:
        console.print(
            f"[yellow]  ⚠ Unknown model loaded: {loaded_model}[/yellow]"
        )
        unknown_warning = (
            f"⚠️ **Unknown model detected**\n\n"
            f"LM Studio has **`{loaded_model}`** loaded, "
            f"but this model is not registered in the bridge.\n\n"
            f"It will not appear in the model picker."
        )
        suggestion_text = (
            f"**To add it**, open `swarm_bridge_server.py` and add "
            f"an entry to `BRIDGE_MODEL_LIST` for `{loaded_model}`."
        )

    actual = ensure_model_loaded(target_model)

    if actual == target_model and loaded_model == target_model:
        status_msg = f"✓ {target_model} already loaded and ready"
    elif actual != target_model:
        status_msg = (
            f"Using **{actual}** "
            f"(already loaded, compatible with {target_model})"
        )
    else:
        status_msg = f"Loaded {target_model} (replaced {loaded_model})"

    console.print(f"[green]  ✓ Pre-flight done → {status_msg}[/green]")

    return PreFlightResult(
        ok=True,
        model_to_use=actual,
        status=status_msg,
        warning=unknown_warning,
        unknown_model=loaded_model if unknown_warning else "",
        suggestion=suggestion_text,
    )


# ══════════════════════════════════════════════════════════════════
# SSE / OPENAI FORMAT HELPERS
# ══════════════════════════════════════════════════════════════════

def make_chunk(content: str, model: str = "executive-swarm") -> str:
    ts = int(time.time())
    return "data: " + json.dumps({
        "id":      f"exec-{ts}",
        "object":  "chat.completion.chunk",
        "created": ts,
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         {"content": content},
            "finish_reason": None,
        }]
    }) + "\n\n"


def make_role_chunk(model: str = "executive-swarm") -> str:
    ts = int(time.time())
    return "data: " + json.dumps({
        "id":      f"exec-{ts}",
        "object":  "chat.completion.chunk",
        "created": ts,
        "model":   model,
        "choices": [{
            "index":         0,
            "delta":         {"role": "assistant", "content": ""},
            "finish_reason": None,
        }]
    }) + "\n\n"


def make_done(model: str = "executive-swarm") -> str:
    ts = int(time.time())
    return (
        "data: " + json.dumps({
            "id":      f"exec-{ts}",
            "object":  "chat.completion.chunk",
            "created": ts,
            "model":   model,
            "choices": [{
                "index":         0,
                "delta":         {},
                "finish_reason": "stop",
            }]
        }) + "\n\n"
        "data: [DONE]\n\n"
    )


def make_keepalive() -> str:
    return ": keepalive\n\n"


def make_non_stream_response(content: str, model: str = "executive-swarm") -> dict:
    ts     = int(time.time())
    tokens = estimate_tokens(content)
    return {
        "id":      f"exec-{ts}",
        "object":  "chat.completion",
        "created": ts,
        "model":   model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     0,
            "completion_tokens": tokens,
            "total_tokens":      tokens,
        }
    }


def make_error_response(error_text: str) -> dict:
    ts = int(time.time())
    return {
        "id":      f"exec-{ts}",
        "object":  "chat.completion",
        "created": ts,
        "model":   "executive-swarm",
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": error_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
        }
    }


# ══════════════════════════════════════════════════════════════════
# CORE STREAMING — with progress bar + live token counter
# ══════════════════════════════════════════════════════════════════

def timed_stream(
    model: str,
    role: str,
    prompt: str,
    system_message: str,
    temperature: float = 0.7,
    max_tokens: int    = 900,
):
    start            = time.time()
    success          = True
    full             = ""
    timeout          = get_model_timeout(model)
    first_token_time = None
    token_count      = 0
    _timed_out       = False

    color = ROLE_COLORS.get(role, "bold white")

    if timeout != FIRST_TOKEN_TIMEOUT:
        console.print(
            f"[dim]  Timeout: {timeout}s "
            f"(model-specific, default={FIRST_TOKEN_TIMEOUT}s)[/dim]"
        )

    console.print(f"\n[{color}]  ▶ {role}[/{color}] [dim][{model}][/dim]")

    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spinner_idx    = 0
    last_progress  = time.time()

    try:
        for chunk in call_local_stream_safe(
            prompt, system_message, model, temperature, max_tokens,
            timeout=timeout,
        ):
            if chunk in (KEEPALIVE_SIGNAL, TIMEOUT_SIGNAL):
                if chunk == TIMEOUT_SIGNAL:
                    _timed_out = True
                if not first_token_time:
                    now = time.time()
                    if now - last_progress >= 0.3:
                        elapsed_wait = now - start
                        frame = spinner_frames[
                            spinner_idx % len(spinner_frames)
                        ]
                        spinner_idx += 1

                        pct     = min(elapsed_wait / timeout, 1.0)
                        filled  = int(pct * 20)
                        empty   = 20 - filled
                        bar     = "█" * filled + "░" * empty
                        pct_str = f"{pct * 100:.0f}%"

                        console.print(
                            f"[dim]  {frame} Waiting for first token "
                            f"[{bar}] {elapsed_wait:.0f}s / {timeout}s "
                            f"({pct_str})[/dim]"
                        )
                        last_progress = now

                yield chunk
            else:
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start
                    console.print(
                        f"[{color}]  ✦ Streaming! "
                        f"(first token in {ttft:.1f}s)[/{color}]"
                    )

                token_count += 1
                full += chunk

                if token_count % 10 == 0:
                    elapsed = time.time() - first_token_time
                    live_tps = token_count / max(elapsed, 0.1)
                    console.print(
                        f"[dim]  ⟫ {token_count} tokens "
                        f"| ~{live_tps:.1f} tok/s "
                        f"| {elapsed:.0f}s elapsed[/dim]"
                    )

                yield chunk

    except Exception as e:
        success = False
        console.print(f"\n[red]  ✗ Error in {role}: {e}[/red]")
        yield f"\n[ERROR] {role} failed: {e}"

    duration   = time.time() - start
    ttft_final = (first_token_time - start) if first_token_time else 0.0
    tokens_est = estimate_tokens(full) + 50
    tps        = tokens_est / max(duration, 0.1)

    if full:
        record_generation(model, role, tokens_est, duration, success)
        if ttft_final > 0:
            record_ttft(model, ttft_final)

        timeout_rate = get_timeout_rate(model)
        learned_to   = get_suggested_timeout(model)
        status_str   = "[green]✓ OK[/green]" if success else "[red]✗ FAIL[/red]"

        console.print(f"\n")
        console.print(
            "[dim]  ╔═══════════════════════════════════════╗[/dim]"
        )
        console.print(f"[dim]  ║  {role} — Generation Complete[/dim]")
        console.print(
            "[dim]  ╠═══════════════════════════════════════╣[/dim]"
        )
        console.print(f"[dim]  ║  Model  : {model}[/dim]")
        console.print(f"[dim]  ║  Tokens : ~{tokens_est:,}[/dim]")
        console.print(f"[dim]  ║  Speed  : ~{tps:.1f} tok/s[/dim]")
        console.print(f"[dim]  ║  Total  : {duration:.1f}s[/dim]")
        console.print(f"[dim]  ║  TTFT   : {ttft_final:.1f}s[/dim]")
        if learned_to:
            console.print(f"[dim]  ║  Learned timeout: {learned_to}s[/dim]")
        if timeout_rate > 0.1:
            console.print(
                f"[dim]  ║  Timeout rate: "
                f"[yellow]{timeout_rate * 100:.0f}%[/yellow][/dim]"
            )
        console.print(f"[dim]  ║  Status : {status_str}[/dim]")
        console.print(
            "[dim]  ╚═══════════════════════════════════════╝[/dim]"
        )
    elif _timed_out:
        record_timeout(model, role)
        console.print(
            f"\n[yellow]  ⏱ {role} timed out — recorded in performance log[/yellow]"
        )
    elif not success:
        console.print(f"\n[red]  ✗ {role} produced no output[/red]")


def stream_agent(
    model: str,
    role: str,
    prompt: str,
    system_message: str,
    temperature: float = 0.7,
    max_tokens: int    = 900,
    board_mode: bool   = False,
):
    timed_out = False

    try:
        for chunk in timed_stream(
            model, role, prompt, system_message, temperature, max_tokens
        ):
            if not is_client_connected():
                console.print(
                    f"\n[yellow]  ⚠ Client disconnected "
                    f"during {role}[/yellow]"
                )
                console.print(
                    "[bold red]  ⛔ Aborting LM Studio "
                    "generation...[/bold red]"
                )
                signal_abort()
                time.sleep(0.5)
                console.print(
                    f"[green]  ✓ Generation stopped "
                    f"for {model}[/green]"
                )
                console.print(
                    "[dim]  ════════════════════════"
                    "════════════════[/dim]"
                )
                return

            if chunk == KEEPALIVE_SIGNAL:
                yield make_keepalive()
            elif chunk == TIMEOUT_SIGNAL:
                timed_out = True
                timeout   = get_model_timeout(model)
                if board_mode:
                    yield make_chunk(
                        f"\n\n> ⏱️ **{role}** did not respond "
                        f"within {timeout}s.\n"
                        f"> This executive's input has been skipped.\n"
                        f"> The board meeting continues.\n\n"
                    )
                else:
                    yield make_chunk(
                        f"\n\n> ⚡ **{role}** timed out after "
                        f"{timeout}s — finding best available "
                        f"fallback...\n\n"
                    )
                break
            else:
                yield make_chunk(chunk)
    except GeneratorExit:
        console.print(
            f"\n[yellow]  ⚠ Stream closed by client "
            f"during {role}[/yellow]"
        )
        console.print(
            "[bold red]  ⛔ Aborting LM Studio "
            "generation...[/bold red]"
        )
        signal_abort()
        time.sleep(0.5)
        console.print(
            f"[green]  ✓ Generation stopped for {model}[/green]"
        )
        console.print(
            "[dim]  ════════════════════════════════════════[/dim]"
        )
        return

    # ── Smart fallback ─────────────────────────────────────────────
    if timed_out and not board_mode:

        loaded_status = get_loaded_model_status()
        loaded_models = loaded_status.get("loaded_models", [])

        console.print(f"[dim]  Currently loaded: {loaded_models}[/dim]")

        fallback_candidates = [
            cfg.fallback_last_resort,
            cfg.fallback_general,
            cfg.fallback_fast_reasoning,
            cfg.fallback_model,
        ]

        smart_fallback = None
        for candidate in fallback_candidates:
            if candidate in loaded_models:
                smart_fallback = candidate
                console.print(
                    f"[cyan]  ✓ Smart fallback → {candidate} "
                    f"(already loaded — no swap)[/cyan]"
                )
                break

        if smart_fallback is None:
            smart_fallback = cfg.fallback_model
            console.print(
                f"[yellow]  No loaded fallback → "
                f"loading {smart_fallback}[/yellow]"
            )

        yield make_chunk(
            f"> Using fallback: **`{smart_fallback}`**\n\n"
        )

        try:
            for chunk in timed_stream(
                smart_fallback, f"{role}[fallback]",
                prompt, system_message, temperature, max_tokens,
            ):
                if not is_client_connected():
                    signal_abort()
                    time.sleep(0.5)
                    console.print(
                        "[green]  ✓ Fallback generation "
                        "stopped[/green]"
                    )
                    return
                if chunk == KEEPALIVE_SIGNAL:
                    yield make_keepalive()
                elif chunk == TIMEOUT_SIGNAL:
                    yield make_chunk(
                        f"\n\n> ❌ **{role}** fallback "
                        f"`{smart_fallback}` also timed out.\n\n"
                    )
                    break
                else:
                    yield make_chunk(chunk)
        except GeneratorExit:
            signal_abort()
            time.sleep(0.5)
            console.print(
                "[green]  ✓ Fallback generation stopped[/green]"
            )
            return


# ══════════════════════════════════════════════════════════════════
# EXECUTIVE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

_EXECUTIVES = [
    {
        "role":      "CEO",
        "emoji":     "👑",
        "section":   "Strategic Assessment",
        "system":    "You are the CEO. Be direct and decisive. Big picture only.",
        "prompt":    "Brief strategic assessment:",
        "tokens":    400,
        "cfg_model": "ceo_model",
        "cfg_temp":  "ceo_temperature",
    },
    {
        "role":      "CTO",
        "emoji":     "⚙️",
        "section":   "Technical Analysis",
        "system":    "You are the CTO. Be specific with technologies and implementation.",
        "prompt":    "Technical analysis:",
        "tokens":    900,
        "cfg_model": "cto_model",
        "cfg_temp":  "cto_temperature",
    },
    {
        "role":      "CFO",
        "emoji":     "📊",
        "section":   "Financial & Risk",
        "system":    "You are the CFO. Think step by step. Quantify everything.",
        "prompt":    "Financial and risk analysis:",
        "tokens":    900,
        "cfg_model": "cfo_model",
        "cfg_temp":  "cfo_temperature",
    },
    {
        "role":      "CPO",
        "emoji":     "🎯",
        "section":   "Product & UX",
        "system":    "You are the CPO. Always start from the user perspective.",
        "prompt":    "Product and UX analysis:",
        "tokens":    900,
        "cfg_model": "cpo_model",
        "cfg_temp":  "cpo_temperature",
    },
    {
        "role":      "COO",
        "emoji":     "📋",
        "section":   "Execution Plan",
        "system":    "You are the COO. Think in sprints. Specific tasks with owners.",
        "prompt":    "Concrete execution plan:",
        "tokens":    900,
        "cfg_model": "coo_model",
        "cfg_temp":  "coo_temperature",
    },
]
# ══════════════════════════════════════════════════════════════════
# ROUTE GENERATORS
# ══════════════════════════════════════════════════════════════════

def gen_full_board(user_message: str, requested_model: str):
    yield make_chunk("## 🏛️ Executive Board Meeting\n\n")
    yield make_chunk(f"**Topic:** {user_message}\n\n")
    yield make_chunk(
        f"**Preset:** {cfg.preset_name.upper()} — "
        f"{cfg.preset_description}\n\n"
    )

    responses_received = 0

    for exec_def in _EXECUTIVES:
        role    = exec_def["role"]
        emoji   = exec_def["emoji"]
        section = exec_def["section"]

        yield make_chunk(f"---\n\n### {emoji} {role} — {section}\n\n")
        yield make_chunk(f"*⏳ Loading {role} model...*\n\n")

        target_model = getattr(cfg, exec_def["cfg_model"])
        temperature  = getattr(cfg, exec_def["cfg_temp"])

        try:
            actual_model = ensure_model_loaded(
                target_model, force_exact=True
            )
        except Exception as e:
            console.print(
                f"[red]  ✗ {role} model load failed: {e}[/red]"
            )
            yield make_chunk(
                f"\n\n> ❌ **{role}** model failed to load: `{e}`\n"
                f"> Skipping this executive.\n\n"
            )
            continue

        if actual_model != target_model:
            console.print(
                f"[yellow]  ⚠ {role}: requested {target_model}, "
                f"got {actual_model}[/yellow]"
            )
            yield make_chunk(
                f"*⚠️ Requested `{target_model}` — "
                f"using `{actual_model}`*\n\n"
            )
        else:
            yield make_chunk(f"*✅ `{actual_model}` ready*\n\n")

        try:
            had_output = False
            for sse_chunk in stream_agent(
                actual_model,
                role,
                f"Original request: {user_message}\n\n"
                f"{exec_def['prompt']}",
                exec_def["system"],
                temperature,
                exec_def["tokens"],
                board_mode=True,
            ):
                had_output = True
                yield sse_chunk

            if had_output:
                responses_received += 1

        except Exception as e:
            console.print(
                f"[red]  ✗ {role} stream crashed: {e}[/red]"
            )
            yield make_chunk(
                f"\n\n> ❌ **{role}** error: `{e}`\n"
                f"> Board meeting continues.\n\n"
            )

    if responses_received > 0:
        yield make_chunk(
            "\n\n---\n\n### 👑 CEO — Final Decision\n\n"
        )
        yield make_chunk(
            "*⏳ Reloading CEO for final synthesis...*\n\n"
        )

        try:
            ceo_final = ensure_model_loaded(
                cfg.ceo_model, force_exact=True
            )
            yield make_chunk(f"*✅ `{ceo_final}` ready*\n\n")

            yield from stream_agent(
                ceo_final,
                "CEO Final",
                (
                    f"Original request: {user_message}\n\n"
                    "Based on all department inputs, give the "
                    "final decision and recommended next actions:"
                ),
                "You are the CEO. Synthesize all inputs and "
                "make the call. Be decisive and specific.",
                cfg.ceo_temperature,
                1000,
                board_mode=True,
            )
        except Exception as e:
            console.print(
                f"[red]  ✗ CEO Final failed: {e}[/red]"
            )
            yield make_chunk(
                f"\n\n> ❌ **CEO Final** failed: `{e}`\n\n"
            )
    else:
        yield make_chunk(
            "\n\n---\n\n"
            "> ⚠️ **No executives responded.**\n"
            "> Try the FAST or GEMMA12B preset.\n"
        )

    yield make_chunk(
        f"\n\n---\n\n"
        f"*Board meeting complete — "
        f"{responses_received}/{len(_EXECUTIVES)} responded.*\n"
    )


def gen_ceo(user_message: str, requested_model: str):
    model = ensure_model_loaded(cfg.ceo_model)
    yield from stream_agent(
        model, "CEO", user_message,
        "You are the CEO. Strategic decisions. Clear and decisive.",
        cfg.ceo_temperature, 2000,
    )


def gen_cto(user_message: str, requested_model: str):
    model = ensure_model_loaded(cfg.cto_model)
    yield from stream_agent(
        model, "CTO", user_message,
        "You are the CTO. Architecture, code quality, security, "
        "performance. Be specific with technologies.",
        cfg.cto_temperature, 2000,
    )


def gen_cfo(user_message: str, requested_model: str):
    model = ensure_model_loaded(cfg.cfo_model)
    yield from stream_agent(
        model, "CFO", user_message,
        "You are the CFO. Cost, risk, ROI, financial feasibility. "
        "Think step by step and quantify everything.",
        cfg.cfo_temperature, 2000,
    )


def gen_cpo(user_message: str, requested_model: str):
    model = ensure_model_loaded(cfg.cpo_model)
    yield from stream_agent(
        model, "CPO", user_message,
        "You are the CPO. User experience, product strategy, "
        "feature prioritization, market fit.",
        cfg.cpo_temperature, 2000,
    )


def gen_coo(user_message: str, requested_model: str):
    model = ensure_model_loaded(cfg.coo_model)
    yield from stream_agent(
        model, "COO", user_message,
        "You are the COO. Concrete tasks, timelines, "
        "owners, dependencies. Think in sprints.",
        cfg.coo_temperature, 2000,
    )


def gen_vision(user_message: str, requested_model: str):
    model = ensure_model_loaded(cfg.vision_model)
    yield make_chunk("## 👁️ Vision Analysis\n\n")
    yield from stream_agent(
        model, "VISION",
        f"Analyze this: {user_message}",
        "You are a visual analysis expert. Be specific and detailed.",
        0.3, 1500,
    )


def gen_direct(user_message: str, requested_model: str):
    if requested_model == "executive-swarm":
        target = cfg.coo_model
        console.print(
            f"[dim]  executive-swarm → resolving to "
            f"COO: {target}[/dim]"
        )
    else:
        target = get_actual_model(requested_model)

    actual = ensure_model_loaded(target)
    console.print(f"[dim]  gen_direct using: {actual}[/dim]")

    if actual != target:
        console.print(
            f"[cyan]  ↳ Requested {target} → "
            f"using loaded {actual} (compatible)[/cyan]"
        )

    yield from stream_agent(
        actual, "Assistant", user_message,
        "You are a highly skilled AI assistant. "
        "Be specific and practical.",
        cfg.ceo_temperature, 2000,
    )


def gen_general(user_message: str, requested_model: str):
    if is_casual_message(user_message):
        model = ensure_model_loaded(cfg.coo_model)
        console.print(
            f"[dim]  Casual → COO ({cfg.coo_model})[/dim]"
        )
    else:
        model = ensure_model_loaded(cfg.ceo_model)

    yield from stream_agent(
        model, "Assistant", user_message,
        "You are a highly skilled AI assistant. "
        "Be specific and practical.",
        cfg.ceo_temperature, 2000,
    )


ROUTE_GENERATORS = {
    "full_board": gen_full_board,
    "ceo":        gen_ceo,
    "cto":        gen_cto,
    "cfo":        gen_cfo,
    "cpo":        gen_cpo,
    "coo":        gen_coo,
    "vision":     gen_vision,
    "direct":     gen_direct,
}


# ══════════════════════════════════════════════════════════════════
# SHARED CHAT LOGIC
# ══════════════════════════════════════════════════════════════════

def extract_messages(messages: list) -> tuple[str, str]:
    def extract_content(msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(content)

    user_message = next(
        (
            extract_content(msg)
            for msg in reversed(messages)
            if msg.get("role") == "user"
        ),
        "",
    )
    system_message = next(
        (
            extract_content(msg)
            for msg in messages
            if msg.get("role") == "system"
        ),
        "",
    )
    return user_message, system_message


def get_generator(requested_model: str, user_message: str):
    model_route = get_model_route(requested_model)

    if requested_model == "executive-swarm":

        if is_casual_message(user_message):
            console.print(
                "[cyan]  Intent: casual → direct (COO)[/cyan]"
            )
            return gen_direct

        intent = detect_intent(user_message)
        console.print(f"[cyan]  Intent: {intent}[/cyan]")

        if intent == "full_board" and len(user_message.strip()) < 200:
            console.print(
                "[cyan]  Short message → redirecting "
                "full_board → direct[/cyan]"
            )
            return gen_direct

        model_route = intent

    else:
        console.print(
            f"[cyan]  Route: {model_route} "
            f"(model={requested_model})[/cyan]"
        )

    return ROUTE_GENERATORS.get(model_route, gen_general)


def collect_full_response(
    generator_fn, user_message: str, requested_model: str
) -> str:
    parts: list[str] = []
    for chunk in generator_fn(user_message, requested_model):
        if chunk.startswith("data: ") and "[DONE]" not in chunk:
            try:
                payload = json.loads(chunk[6:])
                content = (
                    payload
                    .get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content", "")
                )
                if content:
                    parts.append(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
    return "".join(parts)


def run_preflight_and_respond(
    requested_model: str,
    user_message: str,
    do_stream: bool,
    response_format: str = "openai",
):
    clear_abort()

    route     = get_model_route(requested_model)
    preflight = run_preflight(requested_model, route)

    if not preflight.ok:
        error_content = preflight.error

        if response_format == "ollama":
            def ollama_error():
                created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                yield json.dumps({
                    "model":      requested_model,
                    "created_at": created_at,
                    "message": {
                        "role": "assistant",
                        "content": error_content,
                    },
                    "done": False,
                }) + "\n"
                yield json.dumps({
                    "model":      requested_model,
                    "created_at": created_at,
                    "message": {
                        "role": "assistant",
                        "content": "",
                    },
                    "done":        True,
                    "done_reason": "stop",
                }) + "\n"
            return Response(
                stream_with_context(ollama_error()),
                mimetype="application/x-ndjson",
                headers={
                    "Cache-Control":     "no-cache",
                    "Connection":        "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            if do_stream:
                def openai_error():
                    yield make_role_chunk(requested_model)
                    yield make_chunk(error_content)
                    yield make_done(requested_model)
                return Response(
                    stream_with_context(openai_error()),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control":     "no-cache",
                        "Connection":        "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Content-Type":
                            "text/event-stream; charset=utf-8",
                    },
                )
            else:
                return jsonify(make_error_response(error_content))

    actual_model = (
        preflight.model_to_use or get_actual_model(requested_model)
    )
    console.print(f"[green]  ✓ Using: {actual_model}[/green]")

    target = get_actual_model(requested_model)
    if actual_model != target and route != "full_board":
        console.print(
            f"[cyan]  ↳ Requested {target} → "
            f"using loaded {actual_model} (compatible)[/cyan]"
        )

    # ── Response cache check ───────────────────────────────────────
    if _is_cacheable(route, user_message):
        cached = _get_cached(route, user_message)
        if cached:
            console.print("[dim]  ⚡ Cache hit — skipping generation[/dim]")
            if response_format == "ollama":
                if do_stream:
                    def _cached_ollama_stream():
                        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        yield json.dumps({
                            "model": actual_model, "created_at": created_at,
                            "message": {"role": "assistant", "content": cached},
                            "done": False,
                        }) + "\n"
                        yield json.dumps({
                            "model": actual_model, "created_at": created_at,
                            "message": {"role": "assistant", "content": ""},
                            "done": True, "done_reason": "stop",
                        }) + "\n"
                    return Response(
                        stream_with_context(_cached_ollama_stream()),
                        mimetype="application/x-ndjson",
                        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                    )
                return jsonify({
                    "model": actual_model,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": cached},
                    "done": True, "done_reason": "stop",
                })
            else:  # openai
                if do_stream:
                    def _cached_openai_stream():
                        yield make_role_chunk(actual_model)
                        yield make_chunk(cached, actual_model)
                        yield make_done(actual_model)
                    return Response(
                        stream_with_context(_cached_openai_stream()),
                        mimetype="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache", "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                            "Content-Type": "text/event-stream; charset=utf-8",
                        },
                    )
                return jsonify(make_non_stream_response(cached, actual_model))
    # ── End cache check ────────────────────────────────────────────

    generator_fn = get_generator(actual_model, user_message)

    warning_chunks = []
    if preflight.warning:
        warning_chunks.append(f"> {preflight.warning}\n\n")
    if preflight.suggestion:
        warning_chunks.append(f"> {preflight.suggestion}\n\n")
    if (
        actual_model != target
        and route != "full_board"
        and not preflight.warning
    ):
        warning_chunks.append(
            f"*ℹ️ Using **`{actual_model}`** "
            f"(already loaded, compatible with "
            f"`{target}`)*\n\n"
        )

    # ── OpenAI format ──────────────────────────────────────────────
    if response_format == "openai":
        if not do_stream:
            full   = collect_full_response(
                generator_fn, user_message, actual_model
            )
            prefix = "".join(warning_chunks)
            result = prefix + full
            if _is_cacheable(route, user_message) and full:
                _set_cached(route, user_message, result)
            return jsonify(make_non_stream_response(result, actual_model))

        def openai_stream():
            yield make_role_chunk(actual_model)
            for w in warning_chunks:
                yield make_chunk(w)
            try:
                full_parts: list[str] = []
                for sse in generator_fn(user_message, actual_model):
                    if (
                        _is_cacheable(route, user_message)
                        and sse.startswith("data: ")
                        and "[DONE]" not in sse
                    ):
                        try:
                            _c = json.loads(sse[6:])
                            _t = _c.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if _t:
                                full_parts.append(_t)
                        except Exception:
                            pass
                    yield sse
                yield make_done(actual_model)
                if full_parts:
                    _set_cached(route, user_message, "".join(full_parts))
                console.print("[green]  ✓ Stream complete[/green]")
                _print_waiting()
            except GeneratorExit:
                signal_abort()
                _print_waiting()
                return
            except Exception as e:
                console.print(f"[red]  ✗ Stream error: {e}[/red]")
                yield make_chunk(f"\n\n[ERROR] {e}")
                yield make_done(actual_model)
                _print_waiting()

        return Response(
            stream_with_context(openai_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "Connection":        "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type":
                    "text/event-stream; charset=utf-8",
            },
        )

    # ── Ollama format ──────────────────────────────────────────────
    if not do_stream:
        full   = collect_full_response(
            generator_fn, user_message, actual_model
        )
        prefix = "".join(warning_chunks)
        return jsonify({
            "model":      actual_model,
            "created_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "message": {
                "role": "assistant",
                "content": prefix + full,
            },
            "done":        True,
            "done_reason": "stop",
        })

    def ollama_stream():
        # Compute timestamp once — no strftime per token
        created_at   = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cache_parts: list[str] = []

        try:
            for w in warning_chunks:
                yield json.dumps({
                    "model":      actual_model,
                    "created_at": created_at,
                    "message":    {"role": "assistant", "content": w},
                    "done":       False,
                }) + "\n"

            for chunk in generator_fn(user_message, actual_model):
                if chunk.startswith(": keepalive"):
                    continue
                if chunk.startswith("data: ") and "[DONE]" not in chunk:
                    try:
                        payload = json.loads(chunk[6:])
                        content = (
                            payload
                            .get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            if _is_cacheable(route, user_message):
                                cache_parts.append(content)
                            yield json.dumps({
                                "model":      actual_model,
                                "created_at": created_at,
                                "message":    {"role": "assistant", "content": content},
                                "done":       False,
                            }) + "\n"
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

            if cache_parts:
                _set_cached(route, user_message, "".join(cache_parts))

            yield json.dumps({
                "model":             actual_model,
                "created_at":        created_at,
                "message":           {"role": "assistant", "content": ""},
                "done":              True,
                "done_reason":       "stop",
                "total_duration":    0,
                "load_duration":     0,
                "prompt_eval_count": 0,
                "eval_count":        0,
                "eval_duration":     0,
            }) + "\n"

            console.print("[green]  ✓ Ollama stream complete[/green]")
            _print_waiting()

        except GeneratorExit:
            signal_abort()
            _print_waiting()
            return
        except Exception as e:
            console.print(f"[red]  ✗ Ollama stream error: {e}[/red]")
            _print_waiting()

    return Response(
        stream_with_context(ollama_stream()),
        mimetype="application/x-ndjson",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════
# TOOL CALL PASSTHROUGH  (for Cline / function-calling clients)
# ══════════════════════════════════════════════════════════════════

def passthrough_tool_call(
    messages:        list,
    tools:           list,
    tool_choice,
    requested_model: str,
    do_stream:       bool,
):
    """Proxy tool-call requests straight to LM Studio, bypassing board routing."""
    import requests as _req

    actual_model = cfg.ceo_model
    console.print(
        f"[cyan]  🔧 Tool call passthrough → {actual_model} "
        f"({len(tools)} tool(s))[/cyan]"
    )

    payload: dict = {
        "model":    actual_model,
        "messages": messages,
        "tools":    tools,
        "stream":   do_stream,
    }
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    lm_url = f"{get_base_url()}/chat/completions"

    if do_stream:
        def _tool_stream():
            try:
                resp = _req.post(lm_url, json=payload, stream=True, timeout=(10, 600))
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    decoded = raw_line.decode("utf-8")
                    yield decoded + "\n\n"
                    if decoded.strip() == "data: [DONE]":
                        break
                else:
                    yield "data: [DONE]\n\n"
            except Exception as e:
                console.print(f"[red]  ✗ Tool passthrough stream error: {e}[/red]")
                yield make_chunk(f"\n\n[Tool passthrough error] {e}")
                yield make_done(actual_model)
            _print_waiting()

        return Response(
            stream_with_context(_tool_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "Connection":        "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type":      "text/event-stream; charset=utf-8",
            },
        )

    # Non-streaming: forward JSON response as-is
    try:
        resp = _req.post(lm_url, json=payload, timeout=(10, 600))
        resp.raise_for_status()
        return Response(
            resp.content,
            status=resp.status_code,
            mimetype="application/json",
        )
    except Exception as e:
        console.print(f"[red]  ✗ Tool passthrough error: {e}[/red]")
        return jsonify({"error": str(e)}), 500


# OPENAI ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat():
    global _request_counter

    data            = request.json or {}
    messages        = data.get("messages", [])
    requested_model = data.get("model", "executive-swarm")
    do_stream       = data.get("stream", True)
    tools           = data.get("tools")
    tool_choice     = data.get("tool_choice")

    # ── Tool-call passthrough (Cline / function-calling clients) ──
    if tools:
        return passthrough_tool_call(
            messages        = messages,
            tools           = tools,
            tool_choice     = tool_choice,
            requested_model = requested_model,
            do_stream       = do_stream,
        )
    # ── End tool-call passthrough ──────────────────────────────────

    user_message, _ = extract_messages(messages)

    if not user_message:
        return jsonify({"error": "No user message found"}), 400

    # ── Short-circuit OpenWebUI background requests ────────────────
    # These fire automatically (follow-up suggestions, title gen…)
    # and would compete with the real user request for the model.
    if _is_openwebui_system_request(user_message):
        console.print(
            "[dim]  ⚡ OpenWebUI system request — skipped "
            "(not counted)[/dim]"
        )
        if do_stream:
            def _noop_openai_stream():
                yield make_role_chunk(requested_model)
                yield make_done(requested_model)
            return Response(
                stream_with_context(_noop_openai_stream()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control":     "no-cache",
                    "Connection":        "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Content-Type":
                        "text/event-stream; charset=utf-8",
                },
            )
        return jsonify(make_non_stream_response("", requested_model))
    # ── End short-circuit ──────────────────────────────────────────

    _request_counter += 1

    console.print(
        f"\n[bold white on blue]"
        f"  ──── Chat Request #{_request_counter} "
        f"──────────────────────────────"
        f" [/bold white on blue]"
    )
    console.print(
        f"[bold cyan]  → Model:[/bold cyan] {requested_model}"
    )
    console.print(
        f"[bold cyan]  → Task:[/bold cyan]  "
        f"{user_message[:120]}"
        + ("..." if len(user_message) > 120 else "")
    )
    console.print(
        f"[dim]  → Time:  {time.strftime('%H:%M:%S')}[/dim]"
    )
    console.print()

    return run_preflight_and_respond(
        requested_model=requested_model,
        user_message=user_message,
        do_stream=do_stream,
        response_format="openai",
    )


@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    now = int(time.time())
    models = [
        {
            "id":             m["id"],
            "object":         "model",
            "created":        now,
            "owned_by":       "ai-executive-team",
            "name":           m.get("name", m["id"]),
            "description":    m.get("description", ""),
            "context_length": m.get("context_length", 8192),
            "capabilities":   {
                cap: True
                for cap in m.get("capabilities", ["chat"])
            },
        }
        for m in BRIDGE_MODEL_LIST
    ]
    return jsonify({"object": "list", "data": models})


@app.route("/v1/models/<path:model_id>", methods=["GET"])
def openai_get_model(model_id: str):
    now = int(time.time())
    m   = _MODEL_MAP.get(
        model_id, {"id": model_id, "name": model_id}
    )
    return jsonify({
        "id":             m["id"],
        "object":         "model",
        "created":        now,
        "owned_by":       "ai-executive-team",
        "name":           m.get("name", m["id"]),
        "description":    m.get("description", ""),
        "context_length": m.get("context_length", 8192),
    })


# ══════════════════════════════════════════════════════════════════
# OLLAMA ENDPOINTS
# ══════════════════════════════════════════════════════════════════

def model_to_ollama(m: dict, learned: dict | None = None) -> dict:
    short_name = MODEL_NAME_MAP_REVERSE.get(m["id"], m["id"])
    meta = _OLLAMA_META.get(short_name, {
        "family":         "llama",
        "families":       ["llama"],
        "size":           8000000000,
        "parameter_size": m.get("size_label", "unknown"),
    })
    tuned       = (learned or {}).get(m["id"], {})
    quant_label = (tuned.get("k_cache") or "f16").upper()
    return {
        "name":        short_name,
        "model":       short_name,
        "modified_at": "2025-01-01T00:00:00.000Z",
        "size":        meta["size"],
        "digest":      stable_digest(m["id"]),
        "details": {
            "parent_model":       "",
            "format":             "gguf",
            "family":             meta["family"],
            "families":           meta["families"],
            "parameter_size":     meta["parameter_size"],
            "quantization_level": quant_label,
        },
    }


@app.route("/api/version", methods=["GET"])
def ollama_version():
    return jsonify({"version": "0.3.0"})


def _ollama_chat_model_list():
    """Shared helper for all Ollama model-list endpoints."""
    chat_models = [
        m for m in BRIDGE_MODEL_LIST
        if "embeddings" not in m.get("capabilities", [])
    ]
    learned = _read_learned_settings()
    return jsonify({"models": [model_to_ollama(m, learned) for m in chat_models]})


@app.route("/api/tags", methods=["GET"])
def ollama_tags():
    return _ollama_chat_model_list()


@app.route("/api/v0/models", methods=["GET"])
@app.route("/api/v1/models", methods=["GET"])
def ollama_compat_models():
    return _ollama_chat_model_list()


@app.route("/api/ps", methods=["GET"])
def ollama_ps():
    loaded_status = get_loaded_model_status()
    primary       = loaded_status.get("primary")

    if primary:
        short_name = MODEL_NAME_MAP_REVERSE.get(primary, primary)
        meta = _OLLAMA_META.get(short_name, {
            "family":         "llama",
            "families":       ["llama"],
            "size":           8000000000,
            "parameter_size": "unknown",
        })
        return jsonify({
            "models": [{
                "name":       short_name,
                "model":      short_name,
                "size":       meta["size"],
                "digest":     stable_digest(primary),
                "details": {
                    "parent_model":       "",
                    "format":             "gguf",
                    "family":             meta["family"],
                    "families":           meta["families"],
                    "parameter_size":     meta["parameter_size"],
                    "quantization_level": "Q4_K_M",
                },
                "expires_at": "2099-12-31T00:00:00.000Z",
                "size_vram":  meta["size"],
            }]
        })

    return jsonify({
        "models": [{
            "name":       "executive-swarm",
            "model":      "executive-swarm",
            "size":       19000000000,
            "digest":     stable_digest("executive-swarm"),
            "details": {
                "parent_model":       "",
                "format":             "gguf",
                "family":             "llama",
                "families":           ["llama"],
                "parameter_size":     "32B",
                "quantization_level": "Q4_K_M",
            },
            "expires_at": "2099-12-31T00:00:00.000Z",
            "size_vram":  19000000000,
        }]
    })


@app.route("/api/show", methods=["POST"])
def ollama_show():
    data     = request.json or {}
    model_id = data.get("name") or data.get(
        "model", "executive-swarm"
    )

    real_id    = MODEL_NAME_MAP.get(model_id, model_id)
    short_name = MODEL_NAME_MAP_REVERSE.get(real_id, model_id)
    m = _MODEL_MAP.get(real_id, {
        "id":          real_id,
        "name":        real_id,
        "description": "AI Executive Team model",
        "size_label":  "unknown",
    })
    meta = _OLLAMA_META.get(short_name, {
        "family":         "llama",
        "families":       ["llama"],
        "parameter_size": m.get("size_label", "unknown"),
    })

    learned     = _read_learned_settings()
    tuned       = learned.get(real_id, {})
    actual_ctx  = tuned.get("context", 8192)
    actual_k    = tuned.get("k_cache", "f16")
    quant_label = actual_k.upper() if actual_k else "F16"

    return jsonify({
        "modelfile": (
            f"# {m.get('name', model_id)}\n"
            f"# {m.get('description', '')}"
        ),
        "parameters": f"temperature 0.7\nnum_ctx {actual_ctx}",
        "template":   "{{ .System }}\n\n{{ .Prompt }}",
        "details": {
            "parent_model":       "",
            "format":             "gguf",
            "family":             meta["family"],
            "families":           meta["families"],
            "parameter_size":     meta["parameter_size"],
            "quantization_level": quant_label,
        },
        "model_info": {
            "general.architecture":    meta["family"],
            "general.parameter_count": meta.get("size", 0),
            "general.name":            m.get("name", model_id),
            "llm.context_length":      actual_ctx,
        },
    })


@app.route("/api/chat", methods=["POST"])
def ollama_chat():
    global _request_counter

    data            = request.json or {}
    messages        = data.get("messages", [])
    requested_model = data.get("model", "executive-swarm")
    do_stream       = data.get("stream", True)

    user_message, _ = extract_messages(messages)

    if not user_message:
        return jsonify({"error": "No user message"}), 400

    # ── Short-circuit OpenWebUI background requests ────────────────
    if _is_openwebui_system_request(user_message):
        console.print(
            "[dim]  ⚡ OpenWebUI system request — skipped "
            "(not counted)[/dim]"
        )
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if do_stream:
            def _noop_ollama_stream():
                yield json.dumps({
                    "model":      requested_model,
                    "created_at": created_at,
                    "message":    {"role": "assistant", "content": ""},
                    "done":       True,
                    "done_reason": "stop",
                }) + "\n"
            return Response(
                stream_with_context(_noop_ollama_stream()),
                mimetype="application/x-ndjson",
                headers={
                    "Cache-Control":     "no-cache",
                    "Connection":        "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        return jsonify({
            "model":       requested_model,
            "created_at":  created_at,
            "message":     {"role": "assistant", "content": ""},
            "done":        True,
            "done_reason": "stop",
        })
    # ── End short-circuit ──────────────────────────────────────────

    _request_counter += 1

    console.print(
        f"\n[bold white on blue]"
        f"  ──── Chat Request #{_request_counter} "
        f"──────────────────────────────"
        f" [/bold white on blue]"
    )
    console.print(
        f"[bold cyan]  → Model:[/bold cyan] {requested_model}"
    )
    console.print(
        f"[bold cyan]  → Task:[/bold cyan]  "
        f"{user_message[:120]}"
        + ("..." if len(user_message) > 120 else "")
    )
    console.print(
        f"[dim]  → Time:  {time.strftime('%H:%M:%S')}[/dim]"
    )
    console.print()

    return run_preflight_and_respond(
        requested_model=requested_model,
        user_message=user_message,
        do_stream=do_stream,
        response_format="ollama",
    )


@app.route("/api/generate", methods=["POST"])
def ollama_generate():
    global _request_counter
    _request_counter += 1

    data            = request.json or {}
    prompt          = data.get("prompt", "")
    requested_model = data.get("model", "executive-swarm")
    do_stream       = data.get("stream", True)

    if not prompt:
        return jsonify({"error": "No prompt"}), 400

    console.print(
        f"\n[bold white on blue]"
        f"  ──── Generate Request #{_request_counter} "
        f"────────────────────────── [/bold white on blue]"
    )
    console.print(
        f"[bold cyan]  → Model:[/bold cyan] {requested_model}"
    )
    console.print(
        f"[bold cyan]  → Task:[/bold cyan]  "
        f"{prompt[:120]}"
        + ("..." if len(prompt) > 120 else "")
    )
    console.print()

    return run_preflight_and_respond(
        requested_model=requested_model,
        user_message=prompt,
        do_stream=do_stream,
        response_format="ollama",
    )


# ══════════════════════════════════════════════════════════════════
# UTILITY ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    lm_up         = check_lm_studio_health()
    lm_models     = list_local_models() if lm_up else []
    loaded_status = (
        get_loaded_model_status() if lm_up else {}
    )
    learned       = _read_learned_settings()

    board_models = {
        "ceo": cfg.ceo_model,
        "cto": cfg.cto_model,
        "cfo": cfg.cfo_model,
        "cpo": cfg.cpo_model,
        "coo": cfg.coo_model,
    }
    tuner_state = {}
    for role, model_id in board_models.items():
        entry = learned.get(model_id, {})
        if entry:
            tuner_state[role] = {
                "model":    model_id,
                "context":  entry.get("context"),
                "k_cache":  entry.get("k_cache"),
                "v_cache":  entry.get("v_cache"),
                "fa":       entry.get("flash_attn"),
                "used":     entry.get("success_count", 0),
                "verified": bool(entry.get("applied_config")),
            }

    return jsonify({
        "bridge":  "running",
        "version": "2.1.0",
        "preset":  cfg.preset_name,
        "requests_served": _request_counter,
        "lm_studio": {
            "status":           "up" if lm_up else "down",
            "loaded_models":    lm_models,
            "model_count":      len(lm_models),
            "currently_loaded": loaded_status,
        },
        "board":   board_models,
        "utility": {
            "vision":   cfg.vision_model,
            "fallback": cfg.fallback_model,
        },
        "tuner":          tuner_state,
        "bridge_models":  len(BRIDGE_MODEL_LIST),
        "timeout_secs":   FIRST_TOKEN_TIMEOUT,
        "model_timeouts": MODEL_TIMEOUTS,
    })


@app.route("/v1/cache", methods=["GET"])
def cache_status():
    now = time.monotonic()
    with _response_cache_lock:
        entries = [
            {
                "key":        k,
                "expires_in": round(v[1] - now, 1),
                "length":     len(v[0]),
            }
            for k, v in _response_cache.items()
            if now < v[1]
        ]
    return jsonify({
        "cached_entries": len(entries),
        "max_size":       _RESPONSE_CACHE_MAXSIZE,
        "ttl_secs":       _RESPONSE_CACHE_TTL,
        "entries":        entries,
    })


@app.route("/v1/cache", methods=["DELETE"])
def cache_clear():
    with _response_cache_lock:
        count = len(_response_cache)
        _response_cache.clear()
    console.print(f"[yellow]  Cache cleared ({count} entries)[/yellow]")
    return jsonify({"cleared": count})


@app.route("/v1/preset", methods=["GET"])
def get_preset():
    preset_dir = Path(__file__).parent / "presets"
    available = (
        [p.stem for p in preset_dir.glob("*.yaml")]
        if preset_dir.exists()
        else []
    )
    return jsonify({
        "active_preset":    cfg.preset_name,
        "description":      cfg.preset_description,
        "available_presets": available,
    })


@app.route("/v1/preset/<preset_name>", methods=["POST"])
def set_preset(preset_name: str):
    success = cfg.switch_preset(preset_name)
    if success:
        console.print(
            f"[green]  ✓ Preset switched to: "
            f"{preset_name}[/green]"
        )
        return jsonify({
            "switched_to": preset_name,
            "status":      "ok",
            "board": {
                "ceo": cfg.ceo_model,
                "cto": cfg.cto_model,
                "cfo": cfg.cfo_model,
                "cpo": cfg.cpo_model,
                "coo": cfg.coo_model,
            },
        })
    return jsonify({
        "error": f"Preset '{preset_name}' not found"
    }), 404


@app.route("/v1/board", methods=["GET"])
def board_status():
    loaded_status = get_loaded_model_status()
    return jsonify({
        "preset":      cfg.preset_name,
        "description": cfg.preset_description,
        "requests_served": _request_counter,
        "executives": [
            {
                "role":    "CEO",
                "model":   cfg.ceo_model,
                "retail":  "GPT-4o",
                "timeout": get_model_timeout(cfg.ceo_model),
            },
            {
                "role":    "CTO",
                "model":   cfg.cto_model,
                "retail":  "Claude Sonnet",
                "timeout": get_model_timeout(cfg.cto_model),
            },
            {
                "role":    "CFO",
                "model":   cfg.cfo_model,
                "retail":  "o1 Preview",
                "timeout": get_model_timeout(cfg.cfo_model),
            },
            {
                "role":    "CPO",
                "model":   cfg.cpo_model,
                "retail":  "Claude Sonnet",
                "timeout": get_model_timeout(cfg.cpo_model),
            },
            {
                "role":    "COO",
                "model":   cfg.coo_model,
                "retail":  "Copilot Pro",
                "timeout": get_model_timeout(cfg.coo_model),
            },
        ],
        "utility": {
            "vision":   cfg.vision_model,
            "fallback": cfg.fallback_model,
        },
        "currently_loaded": loaded_status,
    })


# ══════════════════════════════════════════════════════════════════
# PRESET SWITCHER + HOTKEYS
# ══════════════════════════════════════════════════════════════════

def run_preset_switcher():
    PRESET_MAP = {
        "1": "fastest",
        "2": "fast",
        "3": "balanced",
        "4": "smart",
        "5": "nuclear",
        "6": "gemma12b",
    }

    def show_menu():
        current = cfg.preset_name.upper()
        print("\n")
        print("  ==========================================")
        print("    SWITCH PRESET -- Bridge keeps running")
        print("  ==========================================")
        print(f"  Current: {current}")
        print("  ------------------------------------------")
        print("  [1] FASTEST   Tiny GPU only    ~30s-1min")
        print("  [2] FAST      All GPU models   ~2-4 min")
        print("  [3] BALANCED  Best daily       ~4-7 min")
        print("  [4] SMART     Max reasoning    ~8-15 min")
        print("  [5] NUCLEAR   Everything maxed ~20-40 min")
        print("  [6] GEMMA12B  OG single model  ~2-4 min")
        print("  ------------------------------------------")
        print("  [ESC / Q] Cancel")
        print("  [X]       Abort current generation")
        print("  ==========================================")
        print("  Your choice: ", end="", flush=True)

    def show_ready_banner(extra_line: str = ""):
        """
        Prints a compact "ready to chat" summary after any model operation.
        Shows current preset + active model config + key controls.
        """
        from load_model import load_learned_settings as _load_ls
        learned     = _load_ls()
        model_id    = cfg.ceo_model
        entry       = learned.get(model_id, {})
        ctx         = entry.get("context",    "?")
        k_cache     = entry.get("k_cache",    "?")
        v_cache     = entry.get("v_cache",    "?")
        fa          = entry.get("flash_attn", False)
        used        = entry.get("success_count", 0)

        fa_str  = "[green]on[/green]" if fa else "[dim]off[/dim]"
        ctx_k   = f"{ctx // 1024}K" if isinstance(ctx, int) else str(ctx)

        console.print(
            "\n[bold green]══════════════════════════════════════"
            "══════════════[/bold green]"
        )
        console.print(
            "[bold green]  ◉ READY — send a message to start chatting"
            "[/bold green]"
        )
        console.print(
            "[bold green]══════════════════════════════════════"
            "══════════════[/bold green]"
        )
        console.print(
            f"[cyan]  Preset  :[/cyan]  {cfg.preset_name.upper()}"
        )
        console.print(
            f"[cyan]  Model   :[/cyan]  {model_id}"
        )
        console.print(
            f"[cyan]  Context :[/cyan]  {ctx_k} ({ctx:,} tokens)"
            if isinstance(ctx, int) else
            f"[cyan]  Context :[/cyan]  {ctx}"
        )
        console.print(
            f"[cyan]  K cache :[/cyan]  {k_cache}   "
            f"[cyan]V cache:[/cyan]  {v_cache}   "
            f"[cyan]FA:[/cyan]  {fa_str}"
        )
        if used:
            console.print(f"[dim]  Loaded {used}x — settings learned[/dim]")
        if extra_line:
            console.print(f"[dim]  {extra_line}[/dim]")
        console.print(
            "[dim]  Keys: SPACE=preset  C=context  F=flash-attn  "
            "T=tune  O=optimize  X=abort  S=status[/dim]"
        )
        console.print(
            "[bold green]══════════════════════════════════════"
            "══════════════[/bold green]\n"
        )

    def do_switch(preset_name: str):
        if preset_name == cfg.preset_name.lower():
            console.print(
                f"\n[yellow]  Already on "
                f"{preset_name.upper()} -- no change[/yellow]\n"
            )
            return

        console.print(
            f"\n[bold yellow]  Switching to "
            f"{preset_name.upper()}...[/bold yellow]"
        )
        success = cfg.switch_preset(preset_name)

        if success:
            console.print(
                f"[bold green]  Now on "
                f"{preset_name.upper()}[/bold green]"
            )
            console.print(
                f"[dim]  CEO      -> {cfg.ceo_model}[/dim]"
            )
            console.print(
                f"[dim]  CTO      -> {cfg.cto_model}[/dim]"
            )
            console.print(
                f"[dim]  CFO      -> {cfg.cfo_model}[/dim]"
            )
            console.print(
                f"[dim]  CPO      -> {cfg.cpo_model}[/dim]"
            )
            console.print(
                f"[dim]  COO      -> {cfg.coo_model}[/dim]"
            )
            console.print(
                f"[dim]  Fallback -> {cfg.fallback_model}[/dim]"
            )
            console.print(
                f"\n[dim]  Takes effect on next request[/dim]\n"
            )
        else:
            console.print(
                f"[red]  Preset '{preset_name}' not found[/red]"
            )
            console.print(
                f"[dim]  Check your presets/ folder[/dim]\n"
            )

    def do_optimize():
        console.print(
            f"\n[bold yellow]  ⟳ Optimizing "
            f"{cfg.ceo_model}...[/bold yellow]"
        )
        console.print(
            "[dim]  Running load_model.py to "
            "apply best settings...[/dim]"
        )
        try:
            result = subprocess.run(
                [sys.executable, "load_model.py", cfg.ceo_model],
                capture_output=False,
                timeout=120,
            )
            if result.returncode == 0:
                console.print(
                    "[bold green]  ✓ Model optimized"
                    "[/bold green]\n"
                )
            else:
                console.print(
                    f"[red]  ✗ Optimize returned code "
                    f"{result.returncode}[/red]\n"
                )
        except subprocess.TimeoutExpired:
            console.print(
                "[red]  ✗ Optimize timed out "
                "after 120s[/red]\n"
            )
        except Exception as e:
            console.print(
                f"[red]  ✗ Optimize failed: {e}[/red]\n"
            )

        # ── Performance snapshot ───────────────────────────────────
        # Show a live summary of what the system has learned so far
        # about every model — speeds, timeouts, reliability scores.
        log = load_log()
        if not log:
            console.print(
                "[dim]  No performance data yet — "
                "run some requests first.[/dim]\n"
            )
            return

        board_models = {
            "CEO":       cfg.ceo_model,
            "CTO":       cfg.cto_model,
            "CFO":       cfg.cfo_model,
            "CPO":       cfg.cpo_model,
            "COO":       cfg.coo_model,
        }

        console.print(
            "\n[bold cyan]  ══════════════════════════════════════"
            "══════════════[/bold cyan]"
        )
        console.print(
            "[bold cyan]  📊 PERFORMANCE SNAPSHOT — "
            f"Preset: {cfg.preset_name.upper()}[/bold cyan]"
        )
        console.print(
            "[bold cyan]  ══════════════════════════════════════"
            "══════════════[/bold cyan]"
        )

        for role, model_id in board_models.items():
            entry = log.get(model_id)
            if not entry:
                console.print(
                    f"[dim]  {role:5s}  {model_id}  "
                    f"— no data yet[/dim]"
                )
                continue

            runs         = max(entry.get("total_runs", 0), 1)
            success_rate = entry.get("successful_runs", 0) / runs * 100
            timeout_rate = entry.get("timeout_count",   0) / runs * 100
            tps          = entry.get("avg_tok_per_sec", 0.0)
            avg_ttft     = entry.get("avg_ttft_secs",   0.0)
            reliability  = get_reliability_score(model_id)
            learned_to   = get_suggested_timeout(model_id)

            timeout_color = (
                "red"    if timeout_rate > 30 else
                "yellow" if timeout_rate > 10 else
                "green"
            )
            reliability_str = (
                f"{reliability:.1f}" if reliability >= 0 else "—"
            )

            console.print(
                f"  [bold]{role:5s}[/bold]  "
                f"[cyan]{model_id}[/cyan]\n"
                f"         Runs: {entry.get('total_runs', 0):>4}  "
                f"│  Speed: {tps:>5.1f} tok/s  "
                f"│  TTFT: {avg_ttft:.1f}s  "
                f"│  Success: {success_rate:.0f}%  "
                f"│  Timeout: [{timeout_color}]{timeout_rate:.0f}%[/{timeout_color}]"
                + (f"  │  Learned TO: {learned_to}s" if learned_to else "")
                + (f"  │  Reliability: {reliability_str}" if reliability >= 0 else "")
            )

        # Also show any non-board models that have data
        board_model_ids = set(board_models.values())
        extras = [
            (mid, entry) for mid, entry in log.items()
            if mid not in board_model_ids
        ]
        if extras:
            console.print(
                "\n[dim]  ── Other models ────────────────────────"
                "──────────────[/dim]"
            )
            for model_id, entry in extras:
                runs         = max(entry.get("total_runs", 0), 1)
                timeout_rate = entry.get("timeout_count", 0) / runs * 100
                tps          = entry.get("avg_tok_per_sec", 0.0)
                timeout_color = (
                    "red"    if timeout_rate > 30 else
                    "yellow" if timeout_rate > 10 else
                    "green"
                )
                console.print(
                    f"  [dim]{model_id}  "
                    f"Runs: {entry.get('total_runs', 0):>4}  "
                    f"│  {tps:.1f} tok/s  "
                    f"│  Timeout: [{timeout_color}]{timeout_rate:.0f}%"
                    f"[/{timeout_color}][/dim]"
                )

        console.print(
            "\n[dim]  Full report: python model_performance_log.py[/dim]"
        )
        console.print(
            "[bold cyan]  ══════════════════════════════════════"
            "══════════════[/bold cyan]\n"
        )
        show_ready_banner("Model reloaded and optimized")

    def do_status():
        loaded = get_loaded_model_status()
        models = loaded.get("loaded_models", [])
        console.print(
            "\n[bold cyan]  ════════════════════════"
            "════════════════[/bold cyan]"
        )
        console.print("[bold cyan]  BRIDGE STATUS[/bold cyan]")
        console.print(
            "[bold cyan]  ════════════════════════"
            "════════════════[/bold cyan]"
        )
        console.print(
            f"[dim]  Preset   : "
            f"{cfg.preset_name.upper()}[/dim]"
        )
        console.print(
            f"[dim]  Requests : {_request_counter}[/dim]"
        )
        console.print(
            f"[dim]  Loaded   : {len(models)} model(s)[/dim]"
        )
        for m in models:
            console.print(f"[dim]    • {m}[/dim]")
        dups = loaded.get("duplicates", [])
        if dups:
            console.print(
                f"[yellow]  Dupes    : {dups}[/yellow]"
            )
        console.print(
            "[bold cyan]  ════════════════════════"
            "════════════════[/bold cyan]\n"
        )

    def do_set_context():
        """
        Cycle context window: 2048 → 4096 → 8192 → 16384 → 2048.
        Writes config, reloads model, saves to learned_settings.
        """
        from load_model import (
            write_config_everywhere,
            unload_model                as _unload,
            cleanup_duplicate_instances as _cleanup_dups,
            resolve_model_id,
            get_lm_studio_model_ids,
            get_model_info              as _get_info,
            save_learned_setting,
            run_load_attempt,
            load_learned_settings       as _load_ls,
            LM_STUDIO_BASE              as _LMS_BASE,
        )

        CONTEXT_STEPS = [2048, 4096, 8192, 16384]

        loaded_status  = get_loaded_model_status()
        model_id       = loaded_status.get("primary") or cfg.ceo_model
        info           = _get_info(model_id)
        name           = info.get("name", model_id)
        learned        = _load_ls().get(model_id, {})

        current_ctx    = learned.get("context", 4096)
        k_cache        = learned.get("k_cache",  "q8_0")
        v_cache        = learned.get("v_cache",  "f16")
        flash_attn     = learned.get("flash_attn", False)

        # Pick next step in the cycle
        try:
            idx = CONTEXT_STEPS.index(current_ctx)
        except ValueError:
            idx = 0
        new_ctx = CONTEXT_STEPS[(idx + 1) % len(CONTEXT_STEPS)]

        console.print(
            f"\n[bold yellow]  ⟳ Context: "
            f"{current_ctx} → [bold]{new_ctx}[/bold] tokens[/bold yellow]"
        )
        console.print(f"[dim]  Model: {name}[/dim]")

        try:
            available_ids = get_lm_studio_model_ids()
            lm_id         = resolve_model_id(model_id, available_ids)
        except Exception as e:
            console.print(f"[red]  ✗ Cannot resolve model ID: {e}[/red]\n")
            return

        try:
            _cleanup_dups(lm_id)
            _unload(lm_id)
            import time as _time; _time.sleep(1.5)
            write_config_everywhere(lm_id, new_ctx, k_cache, v_cache, flash_attn)
            _time.sleep(0.5)

            attempt = {
                "attempt":      1,
                "context":      new_ctx,
                "k_cache":      k_cache,
                "v_cache":      v_cache,
                "flash_attn":   flash_attn,
                "label":        f"Manual ctx={new_ctx}",
                "unload_first": False,
            }
            result = run_load_attempt(lm_id, attempt, info)

            if result.get("status") == "ok":
                save_learned_setting(
                    model_id, new_ctx, k_cache, v_cache,
                    result.get("elapsed", 0), flash_attn,
                    config_paths=result.get("confirmed_config_paths"),
                )
                console.print(
                    f"[bold green]  ✓ Context set to {new_ctx} tokens "
                    f"and saved[/bold green]\n"
                )
                show_ready_banner(f"Context: {new_ctx:,} tokens")
            else:
                console.print(
                    f"[yellow]  ⚠ Load failed — reverted? "
                    f"Try O to reload.[/yellow]\n"
                )
        except Exception as e:
            console.print(f"[red]  ✗ Context change failed: {e}[/red]\n")

    def do_toggle_fa():
        """
        Toggle Flash Attention on ↔ off.
        When enabling FA, switches V cache to q8_0 (required for FA).
        When disabling FA, switches V cache back to f16.
        Writes config, reloads model, saves to learned_settings.
        """
        from load_model import (
            write_config_everywhere,
            unload_model                as _unload,
            cleanup_duplicate_instances as _cleanup_dups,
            resolve_model_id,
            get_lm_studio_model_ids,
            get_model_info              as _get_info,
            save_learned_setting,
            run_load_attempt,
            load_learned_settings       as _load_ls,
            LM_STUDIO_BASE              as _LMS_BASE,
        )

        loaded_status  = get_loaded_model_status()
        model_id       = loaded_status.get("primary") or cfg.ceo_model
        info           = _get_info(model_id)
        name           = info.get("name", model_id)
        learned        = _load_ls().get(model_id, {})

        current_fa  = learned.get("flash_attn", False)
        context     = learned.get("context",    4096)
        k_cache     = learned.get("k_cache",   "q8_0")
        new_fa      = not current_fa

        # FA requires q8/q8 KV; without FA use q8/f16
        new_v_cache = "q8_0" if new_fa else "f16"

        console.print(
            f"\n[bold yellow]  ⟳ Flash Attention: "
            f"{'off' if current_fa else 'on'} → "
            f"[bold]{'on' if new_fa else 'off'}[/bold][/bold yellow]"
        )
        if new_fa:
            console.print(
                f"[dim]  V cache: f16 → q8_0  "
                f"(required for FA)[/dim]"
            )
        else:
            console.print(
                f"[dim]  V cache: q8_0 → f16  "
                f"(restored for FA=off)[/dim]"
            )
        console.print(f"[dim]  Model: {name}[/dim]")

        try:
            available_ids = get_lm_studio_model_ids()
            lm_id         = resolve_model_id(model_id, available_ids)
        except Exception as e:
            console.print(f"[red]  ✗ Cannot resolve model ID: {e}[/red]\n")
            return

        try:
            _cleanup_dups(lm_id)
            _unload(lm_id)
            import time as _time; _time.sleep(1.5)
            write_config_everywhere(lm_id, context, k_cache, new_v_cache, new_fa)
            _time.sleep(0.5)

            attempt = {
                "attempt":      1,
                "context":      context,
                "k_cache":      k_cache,
                "v_cache":      new_v_cache,
                "flash_attn":   new_fa,
                "label":        f"Manual FA={'on' if new_fa else 'off'}",
                "unload_first": False,
            }
            result = run_load_attempt(lm_id, attempt, info)

            if result.get("status") == "ok":
                save_learned_setting(
                    model_id, context, k_cache, new_v_cache,
                    result.get("elapsed", 0), new_fa,
                    config_paths=result.get("confirmed_config_paths"),
                )
                console.print(
                    f"[bold green]  ✓ Flash Attention "
                    f"{'enabled' if new_fa else 'disabled'} "
                    f"and saved[/bold green]\n"
                )
                show_ready_banner(f"Flash Attention {'on' if new_fa else 'off'}, V cache → {new_v_cache}")
            else:
                console.print(
                    f"[yellow]  ⚠ Load failed — try O to reload.[/yellow]\n"
                )
        except Exception as e:
            console.print(f"[red]  ✗ FA toggle failed: {e}[/red]\n")

    def _get_ram_pct() -> float:
        """Returns system RAM usage % using the Windows API (no psutil needed)."""
        try:
            import ctypes
            class _MEMSTATEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength",                ctypes.c_ulong),
                    ("dwMemoryLoad",            ctypes.c_ulong),
                    ("ullTotalPhys",            ctypes.c_ulonglong),
                    ("ullAvailPhys",            ctypes.c_ulonglong),
                    ("ullTotalPageFile",        ctypes.c_ulonglong),
                    ("ullAvailPageFile",        ctypes.c_ulonglong),
                    ("ullTotalVirtual",         ctypes.c_ulonglong),
                    ("ullAvailVirtual",         ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            _s = _MEMSTATEX()
            _s.dwLength = ctypes.sizeof(_s)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(_s))
            return float(_s.dwMemoryLoad)
        except Exception:
            return 0.0

    def do_tune():
        """
        Auto-tuner — tries a matrix of LM Studio configs for the
        current preset's CEO model, benchmarks each one with a live
        generation, picks the fastest, and saves it to
        learned_settings.json so the O key uses it from now on.

        What gets tuned:
          • Context window size  (2048 / 4096 / 8192 / 16384 / 32768)
          • KV cache quantization  (q8_0 / q4_0 / f16)
          • Flash Attention  (on / off — benchmark decides, failures saved)

        FAST mode (default): 4 configs, ~15 min hybrid / ~5 min GPU-only.
        SLOW mode (press S): 10 configs, ~45 min hybrid / ~12 min GPU-only.
          Full cross-product of ctx × FA × k/v cache quant.

        Each config: unload → write config → load → benchmark → record.
        Winner = highest tok/s.  Saved → loaded → done.
        """
        # ── Lazy imports — only pay the cost when T is pressed ────
        import requests as _req
        from load_model import (
            write_config_everywhere,
            unload_model                as _unload,
            cleanup_duplicate_instances as _cleanup_dups,
            resolve_model_id,
            get_lm_studio_model_ids,
            get_model_info              as _get_info,
            save_learned_setting,
            run_load_attempt,
            LM_STUDIO_BASE              as _LMS_BASE,
            is_known_failed,
            save_failed_config,
            get_confirmed_config_paths,
        )

        # ── Which model to tune? — let user pick ─────────────────
        loaded_status  = get_loaded_model_status()
        current_loaded = loaded_status.get("primary") or cfg.ceo_model

        # Build a deduplicated ordered list of preset models with roles
        _role_map: dict[str, list[str]] = {}
        for role, mid in [
            ("CEO",    cfg.ceo_model),
            ("CTO",    cfg.cto_model),
            ("CFO",    cfg.cfo_model),
            ("CPO",    cfg.cpo_model),
            ("COO",    cfg.coo_model),
        ]:
            _role_map.setdefault(mid, []).append(role)

        _tune_options = list(_role_map.keys())   # unique models, insertion order

        console.print(
            "\n[bold yellow]  ══════════════════════════════════"
            "══════════════════[/bold yellow]"
        )
        console.print("[bold yellow]  🔬 AUTO-TUNER — Select model[/bold yellow]")
        console.print(
            "[bold yellow]  ══════════════════════════════════"
            "══════════════════[/bold yellow]"
        )
        for _i, _mid in enumerate(_tune_options, 1):
            _roles  = "/".join(_role_map[_mid])
            _marker = "  ◉ " if _mid == current_loaded else "    "
            console.print(
                f"[cyan]{_marker}[{_i}][/cyan]  "
                f"[bold]{_mid}[/bold]  [dim]({_roles})[/dim]"
            )
        console.print(
            f"\n[dim]  Currently loaded: {current_loaded}[/dim]"
        )
        console.print(
            "[dim]  Press 1-"
            f"{len(_tune_options)} to choose, or ENTER / wait 10s "
            "to tune the loaded model[/dim]\n"
        )

        _pick = [current_loaded]
        _pick_done = [False]

        def _pick_countdown():
            for _r in range(10, 0, -1):
                if _pick_done[0]:
                    return
                sys.stderr.write(
                    f"\r  Tuning: {_pick[0]}  ({_r}s)  "
                )
                sys.stderr.flush()
                time.sleep(1)
            _pick_done[0] = True

        _pt = threading.Thread(target=_pick_countdown, daemon=True)
        _pt.start()

        while not _pick_done[0]:
            if msvcrt.kbhit():
                _k = msvcrt.getwch()
                _ki = _k if isinstance(_k, str) else _k.decode("ascii", errors="ignore")
                if _ki in [str(n) for n in range(1, len(_tune_options) + 1)]:
                    _pick[0] = _tune_options[int(_ki) - 1]
                    _pick_done[0] = True
                    sys.stderr.write(
                        f"\r  ✓ [{_ki}] → {_pick[0]}"
                        f"                              \n"
                    )
                    sys.stderr.flush()
                    break
                elif _ki in ("\r", "\n"):
                    _pick_done[0] = True
                    sys.stderr.write(
                        f"\r  ✓ [ENTER] → {_pick[0]}"
                        f"                              \n"
                    )
                    sys.stderr.flush()
                    break
            time.sleep(0.05)

        _pt.join(timeout=12)
        sys.stderr.write("\n")
        sys.stderr.flush()

        model_id = _pick[0]

        # ── Fast or Slow tune? ────────────────────────────────────
        console.print(
            "\n[bold yellow]  ════════════════════════════════════"
            "════════════════[/bold yellow]"
        )
        console.print("[bold yellow]  🔬 TUNE DEPTH[/bold yellow]")
        console.print(
            "[bold yellow]  ════════════════════════════════════"
            "════════════════[/bold yellow]"
        )
        console.print(
            "[cyan]  [F] FAST  — 4 configs   (~15 min hybrid / ~5 min GPU)[/cyan]"
        )
        console.print(
            "[dim]      FA on/off × 2-3 context sizes — quick sanity check[/dim]"
        )
        console.print(
            "[cyan]  [S] SLOW  — 10 configs  (~45 min hybrid / ~12 min GPU)[/cyan]"
        )
        console.print(
            "[dim]      Full grid: ctx × FA × k/v cache quant — thorough search[/dim]"
        )
        console.print("[dim]  Press F/S or wait 5s for FAST...[/dim]\n")

        _tune_fast    = [True]
        _speed_done   = [False]

        def _speed_countdown():
            for _r in range(5, 0, -1):
                if _speed_done[0]:
                    return
                sys.stderr.write(f"\r  Depth: FAST  ({_r}s)  ")
                sys.stderr.flush()
                time.sleep(1)
            _speed_done[0] = True

        _spt = threading.Thread(target=_speed_countdown, daemon=True)
        _spt.start()

        while not _speed_done[0]:
            if msvcrt.kbhit():
                _sk  = msvcrt.getwch()
                _ski = _sk if isinstance(_sk, str) else _sk.decode("ascii", errors="ignore")
                if _ski in ("s", "S"):
                    _tune_fast[0]  = False
                    _speed_done[0] = True
                    sys.stderr.write(
                        "\r  ✓ SLOW tune selected                    \n"
                    )
                    sys.stderr.flush()
                    break
                elif _ski in ("f", "F", "\r", "\n"):
                    _speed_done[0] = True
                    sys.stderr.write(
                        "\r  ✓ FAST tune selected                    \n"
                    )
                    sys.stderr.flush()
                    break
            time.sleep(0.05)

        _spt.join(timeout=7)
        sys.stderr.write("\n")
        sys.stderr.flush()
        is_fast_tune = _tune_fast[0]

        info    = _get_info(model_id)
        gpu_fit = info.get("gpu_fit", True)
        size_gb = info.get("size_gb", 8.0)
        name    = info.get("name", model_id)

        # ── Build test matrix ─────────────────────────────────────
        # FAST: 4 configs — FA on/off × 2-3 context sizes.
        # SLOW: 10 configs — full cross-product of ctx × FA × cache quant.
        #
        # FA=on configs that are incompatible with the hardware/model will
        # fail the readiness probe and be saved to failed_configs so they're
        # permanently skipped on future tune runs.  Let the benchmark decide
        # rather than hard-coding FA=off everywhere.
        #
        # GPU-fit  (< 10 GB): all layers on GPU — can push higher context.
        # Hybrid   (≥ 10 GB): layers split GPU+RAM — conservative on context.

        if is_fast_tune:
            if gpu_fit and size_gb < 10:
                test_matrix = [
                    {"context": 8192,  "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "8K  q8/q8  FA=on"},
                    {"context": 8192,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "8K  q8/f16 FA=off"},
                    {"context": 16384, "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "16K q8/q8  FA=on"},
                    {"context": 4096,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "4K  q8/f16 FA=off"},
                ]
            else:
                # Hybrid fast — FA on/off at baseline + next step up
                test_matrix = [
                    {"context": 4096,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "4K  q8/f16 FA=off"},
                    {"context": 4096,  "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "4K  q8/q8  FA=on"},
                    {"context": 8192,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "8K  q8/f16 FA=off"},
                    {"context": 2048,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "2K  q8/f16 FA=off"},
                ]
        else:
            # SLOW — full cross-product: ctx × FA × k/v cache quant
            if gpu_fit and size_gb < 10:
                test_matrix = [
                    {"context": 16384, "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "16K q8/q8  FA=on"},
                    {"context": 16384, "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "16K q8/f16 FA=off"},
                    {"context": 16384, "k_cache": "q4_0", "v_cache": "f16",  "flash_attn": False, "label": "16K q4/f16 FA=off"},
                    {"context": 8192,  "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "8K  q8/q8  FA=on"},
                    {"context": 8192,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "8K  q8/f16 FA=off"},
                    {"context": 8192,  "k_cache": "q4_0", "v_cache": "f16",  "flash_attn": False, "label": "8K  q4/f16 FA=off"},
                    {"context": 4096,  "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "4K  q8/q8  FA=on"},
                    {"context": 4096,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "4K  q8/f16 FA=off"},
                    {"context": 32768, "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "32K q8/q8  FA=on (ambitious)"},
                ]
            else:
                # Hybrid slow — conservative on ctx, thorough on FA + cache
                test_matrix = [
                    {"context": 2048,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "2K  q8/f16 FA=off"},
                    {"context": 2048,  "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "2K  q8/q8  FA=on"},
                    {"context": 4096,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "4K  q8/f16 FA=off"},
                    {"context": 4096,  "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "4K  q8/q8  FA=on"},
                    {"context": 4096,  "k_cache": "q4_0", "v_cache": "f16",  "flash_attn": False, "label": "4K  q4/f16 FA=off"},
                    {"context": 8192,  "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "8K  q8/f16 FA=off"},
                    {"context": 8192,  "k_cache": "q8_0", "v_cache": "q8_0", "flash_attn": True,  "label": "8K  q8/q8  FA=on"},
                    {"context": 8192,  "k_cache": "q4_0", "v_cache": "f16",  "flash_attn": False, "label": "8K  q4/f16 FA=off"},
                    {"context": 16384, "k_cache": "q8_0", "v_cache": "f16",  "flash_attn": False, "label": "16K q8/f16 FA=off"},
                    {"context": 16384, "k_cache": "q4_0", "v_cache": "f16",  "flash_attn": False, "label": "16K q4/f16 FA=off"},
                ]

        # ── Benchmark prompt — same every time for fair comparison ─
        BENCH_PROMPT = "List every prime number between 1 and 50, one per line."
        BENCH_SYSTEM = "You are a helpful assistant. Be concise and direct."
        BENCH_TOKENS = 30

        est_per   = info.get("est_load_s", 30) + 25   # load + benchmark
        est_total = len(test_matrix) * est_per

        # ── Check if already tuned ────────────────────────────────
        from load_model import load_learned_settings as _load_ls
        _prev = _load_ls().get(model_id, {})
        _already_tuned = bool(_prev.get("context"))

        console.print(
            "\n[bold yellow]  ══════════════════════════════════"
            "══════════════════[/bold yellow]"
        )
        console.print(
            "[bold yellow]  🔬 AUTO-TUNER[/bold yellow]"
        )
        console.print(
            "[bold yellow]  ══════════════════════════════════"
            "══════════════════[/bold yellow]"
        )
        console.print(f"[cyan]  Model    : {name}[/cyan]")
        console.print(
            f"[cyan]  Depth    : {'FAST' if is_fast_tune else 'SLOW'}[/cyan]"
        )
        console.print(
            f"[cyan]  GPU fit  : "
            f"{'Yes ✅' if gpu_fit else 'Hybrid ⚠️'}  "
            f"({size_gb} GB)[/cyan]"
        )
        console.print(
            f"[cyan]  Configs  : {len(test_matrix)}  "
            f"× ~{est_per}s each  "
            f"≈ {est_total // 60}m {est_total % 60}s total[/cyan]"
        )
        if _already_tuned:
            console.print(
                f"[green]  Previous : ctx={_prev['context']}  "
                f"k={_prev.get('k_cache','?')}  "
                f"v={_prev.get('v_cache','?')}  "
                f"({_prev.get('success_count', 1)}x used)[/green]"
            )
            console.print(
                "[dim]  Re-tuning will overwrite the saved result "
                "if a better config is found.[/dim]"
            )
        else:
            console.print(
                "[dim]  First tune — no previous settings for this "
                "model.[/dim]"
            )
        console.print(
            f"[dim]  Benchmark : \"{BENCH_PROMPT}\"[/dim]"
        )
        console.print(
            "[dim]  Same prompt every config → fair tok/s "
            "comparison.[/dim]"
        )
        console.print(
            "[dim]  Run once per model — winner saved permanently "
            "to learned_settings.json.[/dim]"
        )
        console.print(
            "[dim]  O key will use the winner automatically from "
            "now on.[/dim]"
        )
        console.print(
            "[yellow]  ⚠ Bridge paused during tuning  "
            "│  Press ESC between configs to stop early[/yellow]\n"
        )

        # ── Resolve LM Studio model ID ────────────────────────────
        try:
            available_ids = get_lm_studio_model_ids()
            lm_id = resolve_model_id(model_id, available_ids)
        except Exception as e:
            console.print(f"[red]  ✗ Cannot reach LM Studio: {e}[/red]\n")
            return

        if lm_id != model_id:
            console.print(f"[dim]  LM ID: {lm_id}[/dim]\n")

        # Clean up any orphaned duplicate instances left by a previous tune run
        _cleanup_dups(lm_id)

        tune_results: list[dict] = []
        _stop_early = False

        for i, cfg_entry in enumerate(test_matrix):
            # ── ESC check between configs ─────────────────────────
            # Drain any keys that were buffered while the last
            # config was loading/benchmarking.
            while msvcrt.kbhit():
                _k = msvcrt.getch()
                if _k in (b"\x1b", b"q", b"Q"):
                    _stop_early = True
                    break
            if _stop_early:
                console.print(
                    "\n[yellow]  ⏹ Stopped early — "
                    "using best result so far.[/yellow]"
                )
                break

            label      = cfg_entry["label"]
            context    = cfg_entry["context"]
            k_cache    = cfg_entry["k_cache"]
            v_cache    = cfg_entry["v_cache"]
            flash_attn = cfg_entry["flash_attn"]

            console.print(
                f"[bold]  [{i + 1}/{len(test_matrix)}] {label}[/bold]"
            )

            # Step 0: Skip configs that previously failed on this hardware.
            # Saves ~140s per known-bad config and avoids re-discovering pain.
            _prev_fail = is_known_failed(
                model_id, context, k_cache, v_cache, flash_attn
            )
            if _prev_fail:
                _fc = _prev_fail.get("fail_count", 1)
                _fr = _prev_fail.get("reason", "unknown")
                _ft = _prev_fail.get("last_tried", "unknown")
                console.print(
                    f"[yellow]    Skipping — failed {_fc}x before "
                    f"(last: {_ft})[/yellow]"
                )
                console.print(f"[dim]    Reason: {_fr}[/dim]")
                console.print(
                    "[dim]    To retry, clear failed_configs for this model "
                    "in learned_settings.json[/dim]"
                )
                tune_results.append({
                    "cfg":     cfg_entry,
                    "tps":     0.0,
                    "ok":      False,
                    "skipped": True,
                    "reason":  f"previously failed {_fc}x: {_fr}",
                })
                continue

            # Step 1: Unload
            console.print("[dim]    → Unloading...[/dim]")
            try:
                _unload(lm_id)
                time.sleep(2)
            except Exception as e:
                console.print(f"[yellow]    ⚠ Unload warning: {e}[/yellow]")

            # Step 2: Write config
            try:
                write_config_everywhere(
                    lm_id, context, k_cache, v_cache, flash_attn
                )
                time.sleep(0.5)
            except Exception as e:
                console.print(f"[red]    ✗ Config write failed: {e}[/red]")
                tune_results.append(
                    {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                     "reason": f"config write failed: {e}"}
                )
                continue

            # Step 2.5: RAM pressure check — abort before loading if system is near-full.
            # At 92%+ RAM the OS starts thrashing and results are meaningless.
            _ram_pct = _get_ram_pct()
            if _ram_pct >= 92:
                console.print(
                    f"[red]    !! RAM at {_ram_pct:.0f}% — skipping load "
                    f"to avoid system freeze[/red]"
                )
                tune_results.append({
                    "cfg": cfg_entry, "tps": 0.0, "ok": False,
                    "reason": f"RAM pressure {_ram_pct:.0f}%",
                })
                continue
            elif _ram_pct >= 82:
                console.print(
                    f"[yellow]    RAM at {_ram_pct:.0f}% — loading may be slow[/yellow]"
                )

            # Step 3: Load — pass known config paths so we skip rediscovery
            attempt = {
                "attempt":     i + 1,
                "context":     context,
                "k_cache":     k_cache,
                "v_cache":     v_cache,
                "flash_attn":  flash_attn,
                "label":       label,
                "unload_first": False,   # already unloaded above
            }
            _known_paths = get_confirmed_config_paths(model_id)
            try:
                load_result = run_load_attempt(
                    lm_id, attempt, info,
                    known_paths=_known_paths,
                )
            except Exception as e:
                console.print(f"[red]    ✗ Load error: {e}[/red]")
                save_failed_config(
                    model_id, context, k_cache, v_cache, flash_attn,
                    f"load error: {e}"[:120],
                )
                tune_results.append(
                    {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                     "reason": str(e)}
                )
                continue

            if load_result.get("status") != "ok":
                reason = load_result.get("message", "unknown")[:80]
                console.print(f"[yellow]    ✗ Load failed: {reason}[/yellow]")
                save_failed_config(
                    model_id, context, k_cache, v_cache, flash_attn,
                    f"load failed: {reason}",
                )
                tune_results.append(
                    {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                     "reason": reason}
                )
                continue

            load_time = load_result.get("elapsed", 0)

            # Step 3.5: Readiness probe — tiny 1-token request to verify the
            # model is *actually* responsive at this config before committing
            # 140s to a real benchmark. If the model can't return 1 token in
            # the probe budget, it's not viable at these settings on this
            # hardware (RAM bottleneck on prompt processing, OOM, etc.).
            _probe_budget = 30 if gpu_fit else 75
            console.print(
                f"[dim]    Probing readiness (1-token, budget {_probe_budget}s)...[/dim]"
            )
            _probe_start = time.time()
            _probe_ok    = False
            _probe_err   = None
            try:
                _pr = _req.post(
                    f"{_LMS_BASE}/v1/chat/completions",
                    json={
                        "model":       lm_id,
                        "messages":    [{"role": "user", "content": "hi"}],
                        "max_tokens":  1,
                        "temperature": 0,
                        "stream":      False,
                    },
                    timeout=_probe_budget,
                )
                if _pr.status_code == 200:
                    _probe_ok = True
                else:
                    _probe_err = f"HTTP {_pr.status_code}"
            except Exception as _pex:
                _probe_err = str(_pex)[:100]

            _probe_secs = time.time() - _probe_start
            if not _probe_ok:
                _reason = (
                    f"probe unresponsive after {_probe_secs:.0f}s "
                    f"({_probe_err})"
                )
                console.print(
                    f"[yellow]    ✗ Model didn't respond to probe: "
                    f"{_probe_err}[/yellow]"
                )
                console.print(
                    "[yellow]    Config not viable at this hardware — "
                    "skipping full benchmark.[/yellow]"
                )
                save_failed_config(
                    model_id, context, k_cache, v_cache, flash_attn, _reason
                )
                tune_results.append(
                    {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                     "reason": _reason}
                )
                continue
            console.print(
                f"[green]    Probe OK in {_probe_secs:.1f}s — model is responsive[/green]"
            )

            # Step 4: Benchmark — NON-STREAMING, threaded with live elapsed time.
            # We reverted from SSE streaming because the LM Studio stream format
            # was producing "no tokens returned" false negatives (delta events
            # without content fields, usage chunks parsed as no-ops, etc.) that
            # caused us to mark the working config as failed. Non-streaming uses
            # usage.completion_tokens which is reliable.
            #
            # The probe already gave us TTFT (time-to-first-token at this config),
            # so we don't need streaming for that. The benchmark now measures
            # steady-state generation speed only (model is already warm from probe).
            _BENCH_TIMEOUT = 45 if gpu_fit else 120

            # Visible settle countdown — KV cache + warmup finalize
            _settle = 2 if gpu_fit else 4
            console.print(
                f"[dim]    Finalizing post-probe (KV cache settle) — {_settle}s[/dim]"
            )
            for _s in range(_settle, 0, -1):
                console.print(f"[dim]    ... finalizing {_s}s[/dim]")
                time.sleep(1)

            console.print(
                f"[dim]    Benchmark — generating {BENCH_TOKENS} tokens "
                f"(limit {_BENCH_TIMEOUT}s)[/dim]"
            )
            bench_start  = time.time()
            _bench_resp  = [None]
            _bench_exc   = [None]
            _bench_done  = threading.Event()

            def _run_bench():
                try:
                    _bench_resp[0] = _req.post(
                        f"{_LMS_BASE}/v1/chat/completions",
                        json={
                            "model":       lm_id,
                            "messages":    [
                                {"role": "system", "content": BENCH_SYSTEM},
                                {"role": "user",   "content": BENCH_PROMPT},
                            ],
                            "max_tokens":  BENCH_TOKENS,
                            "temperature": 0,
                            "stream":      False,
                        },
                        timeout=_BENCH_TIMEOUT,
                    )
                except Exception as exc:
                    _bench_exc[0] = exc
                finally:
                    _bench_done.set()

            _bt = threading.Thread(target=_run_bench, daemon=True)
            _bt.start()

            _next_tick = 3
            while not _bench_done.wait(timeout=1.0):
                _elapsed = int(time.time() - bench_start)
                if _elapsed >= _next_tick:
                    _pct   = min(99, int(_elapsed / _BENCH_TIMEOUT * 100))
                    _fill  = "=" * (_pct // 5)
                    _empty = "." * (20 - _pct // 5)
                    console.print(
                        f"[dim]    |{_fill}{_empty}| {_elapsed}s / "
                        f"{_BENCH_TIMEOUT}s — generating tokens...[/dim]"
                    )
                    _next_tick = _elapsed + 3
                if _elapsed >= _BENCH_TIMEOUT:
                    break
            _bench_done.wait(timeout=5)
            bench_secs = time.time() - bench_start
            resp = _bench_resp[0]
            _ex  = _bench_exc[0]

            if _ex is not None:
                console.print(f"[red]    ✗ Benchmark error: {_ex}[/red]")
                # Don't save as failed — transient errors shouldn't trash settings
                tune_results.append(
                    {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                     "reason": str(_ex)}
                )
            elif resp is None:
                _to_reason = f"benchmark timeout after {bench_secs:.0f}s"
                console.print(
                    f"[yellow]    ✗ Benchmark timed out after {bench_secs:.0f}s[/yellow]"
                )
                save_failed_config(
                    model_id, context, k_cache, v_cache, flash_attn, _to_reason
                )
                tune_results.append(
                    {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                     "reason": _to_reason}
                )
            elif resp.status_code == 200:
                data  = resp.json()
                usage = data.get("usage", {}) or {}
                ntok  = int(usage.get("completion_tokens", 0))

                if ntok > 0:
                    # Pure generation tok/s (warm model — KV cache already built
                    # during the probe, so this measures real steady-state speed)
                    gen_tps = ntok / max(bench_secs, 0.1)
                    # End-to-end first-request feel = probe time (TTFT) + generation
                    e2e_secs = _probe_secs + bench_secs
                    e2e_tps  = ntok / max(e2e_secs, 0.1)
                    console.print(
                        f"[green]    OK {ntok} tokens — "
                        f"first token {_probe_secs:.1f}s (from probe), "
                        f"generation [bold]{gen_tps:.1f} tok/s[/bold], "
                        f"end-to-end {e2e_tps:.1f} tok/s[/green]"
                    )
                    tune_results.append({
                        "cfg":        cfg_entry,
                        "tps":        gen_tps,         # for ranking
                        "gen_tps":    gen_tps,
                        "e2e_tps":    e2e_tps,
                        "ttft":       _probe_secs,
                        "bench_secs": bench_secs,
                        "load_secs":  load_time,
                        "tokens":     ntok,
                        "ok":         True,
                    })
                else:
                    # HTTP 200 but no completion tokens — odd, but DON'T save as
                    # failed since this is likely a measurement artifact, not a
                    # real config failure (the probe just confirmed responsiveness).
                    console.print(
                        f"[yellow]    ✗ HTTP 200 but completion_tokens=0 "
                        f"(measurement artifact — not marked as failed)[/yellow]"
                    )
                    tune_results.append(
                        {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                         "reason": "HTTP 200 but no completion tokens (artifact)"}
                    )
            else:
                console.print(
                    f"[yellow]    ✗ Benchmark HTTP {resp.status_code}[/yellow]"
                )
                save_failed_config(
                    model_id, context, k_cache, v_cache, flash_attn,
                    f"HTTP {resp.status_code}",
                )
                tune_results.append(
                    {"cfg": cfg_entry, "tps": 0.0, "ok": False,
                     "reason": f"HTTP {resp.status_code}"}
                )

        # Unload whatever is still loaded — clean state after tuning
        console.print("[dim]  Unloading after tune...[/dim]")
        try:
            _unload(lm_id)
            time.sleep(1)
        except Exception as _ue:
            console.print(f"[yellow]  Unload warning: {_ue}[/yellow]")

        # ── Results table ─────────────────────────────────────────
        successful = [r for r in tune_results if r.get("ok") and r["tps"] > 0]

        console.print(
            "\n[bold cyan]  ══════════════════════════════════"
            "══════════════════[/bold cyan]"
        )
        console.print("[bold cyan]  🏁 TUNE RESULTS[/bold cyan]")
        console.print(
            "[bold cyan]  ══════════════════════════════════"
            "══════════════════[/bold cyan]"
        )

        if not successful:
            console.print(
                "[red]  ✗ No configs completed successfully.[/red]\n"
            )
            # Diagnose why — give specific feedback instead of "check logs"
            _reasons    = [r.get("reason", "") for r in tune_results if not r.get("ok")]
            _reasons_lc = " ".join(_reasons).lower()

            if "ram pressure" in _reasons_lc:
                _ram_vals = [r for r in _reasons if "ram pressure" in r.lower()]
                console.print(
                    f"[red]  Cause: System RAM pressure ({_ram_vals[0] if _ram_vals else 'high'})[/red]"
                )
                console.print(
                    "[yellow]  Fix: Close other apps to free RAM, or tune a GPU-only model first.[/yellow]"
                )
            elif any(x in _reasons_lc for x in ["out of memory", "oom", "failed to allocate"]):
                console.print("[red]  Cause: Out of memory (VRAM or RAM full)[/red]")
                console.print("[yellow]  Fix: Try a smaller context size or a GPU-only model.[/yellow]")
            elif "timeout" in _reasons_lc or "timed out" in _reasons_lc or "connectionerror" in _reasons_lc:
                _n_to = sum(1 for r in _reasons if "timeout" in r.lower() or "timed out" in r.lower())
                console.print(
                    f"[red]  Cause: Benchmark timed out ({_n_to}/{len(_reasons)} configs)[/red]"
                )
                console.print(
                    "[yellow]  This is a RAM-offload model — first token can take 60-120s.[/yellow]"
                )
                console.print(
                    "[yellow]  Tune a GPU-only model first, or run this tune overnight.[/yellow]"
                )
            elif any(x in _reasons_lc for x in ["not found", "no such file", "model_not_found"]):
                console.print("[red]  Cause: Model file not found by LM Studio[/red]")
                console.print("[yellow]  Fix: Open LM Studio and confirm the model is visible.[/yellow]")
            elif "load failed" in _reasons_lc or "http 5" in _reasons_lc:
                console.print("[red]  Cause: LM Studio rejected the load request[/red]")
                for _r in _reasons[:3]:
                    console.print(f"[red]    - {_r}[/red]")
                console.print("[yellow]  Load the model manually in LM Studio to see the full error.[/yellow]")
            else:
                for _r in _reasons[:4]:
                    console.print(f"[red]    - {_r}[/red]")
                console.print(
                    "[yellow]  Load the model manually in LM Studio to see the full error.[/yellow]"
                )
            console.print()
            return

        successful.sort(key=lambda r: r["tps"], reverse=True)
        best     = successful[0]
        best_tps = best["tps"]

        # ── Annotated results table ───────────────────────────────
        for r in successful:
            is_best = r is best
            marker  = "🏆" if is_best else "  "
            cfg_r   = r["cfg"]
            lbl     = cfg_r["label"]
            tps     = r["tps"]
            pct     = tps / best_tps * 100

            # Per-row note — explains WHY this config performed as it did
            notes = []
            # Find the matching FA=off peer to judge whether FA actually helped
            peer_tps = next(
                (x["tps"] for x in successful
                 if x["cfg"]["context"] == cfg_r["context"]
                 and x["cfg"]["flash_attn"] != cfg_r["flash_attn"]
                 and x["tps"] > 0),
                None,
            )
            if cfg_r["flash_attn"]:
                if is_best:
                    notes.append("Flash Attention helped — keep it on")
                elif peer_tps and tps >= peer_tps * 0.98:
                    notes.append("FA matched FA=off — model may have disabled it internally")
                else:
                    notes.append("Flash Attention hurt this model on your GPU")
            else:
                if not is_best and peer_tps and peer_tps > tps * 1.02:
                    notes.append("FA=on was faster at this context size")
            if cfg_r["context"] >= 16384 and is_best:
                notes.append("max context + fast — great result")
            if cfg_r["context"] <= 2048 and not is_best:
                notes.append("small context limits usefulness")
            if not is_best and pct >= 95:
                notes.append(f"only {100 - pct:.0f}% slower — viable alternative")
            if not is_best and pct < 70:
                notes.append(f"{100 - pct:.0f}% slower than winner")

            note_str = f"  [dim]← {notes[0]}[/dim]" if notes else ""
            color    = "bold green" if is_best else "white"

            console.print(
                f"  {marker} [{color}]{tps:>6.1f} tok/s[/{color}]  "
                f"load {r.get('load_secs', 0):>3}s  "
                f"{lbl}"
                + note_str
            )

        for r in tune_results:
            if not r.get("ok"):
                reason = r.get("reason", "failed")
                # Plain-English reason
                if "flash" in reason.lower() or "v cache" in reason.lower():
                    hint = "Flash Attention not supported with this KV config"
                elif "memory" in reason.lower() or "oom" in reason.lower():
                    hint = "Not enough VRAM for this context size"
                elif "not found" in reason.lower():
                    hint = "Model not found on disk"
                elif "timeout" in reason.lower():
                    hint = "Took too long to load"
                else:
                    hint = reason[:60]
                console.print(
                    f"  ✗  [dim]{r['cfg']['label']:30s}"
                    f" — {hint}[/dim]"
                )

        # ── Winner summary ────────────────────────────────────────
        best_cfg = best["cfg"]
        console.print(
            f"\n[bold green]  🏆 Winner: {best_cfg['label']}[/bold green]  "
            f"[bold]{best_tps:.1f} tok/s[/bold]"
        )
        console.print()

        # Plain-English explanation of what the winner settings mean
        ctx_k    = best_cfg["context"] // 1024
        k_cache  = best_cfg["k_cache"]
        v_cache  = best_cfg["v_cache"]
        fa       = best_cfg["flash_attn"]
        worst    = successful[-1]["tps"] if len(successful) > 1 else best_tps
        gain_pct = int((best_tps - worst) / max(worst, 0.1) * 100)

        console.print(
            f"[dim]  Context window : {best_cfg['context']:,} tokens "
            f"({ctx_k}K) — how much conversation history fits in memory[/dim]"
        )
        console.print(
            f"[dim]  K cache quant  : {k_cache} — "
            + ("compressed key cache, saves VRAM" if k_cache == "q8_0" else "full precision key cache")
            + "[/dim]"
        )
        console.print(
            f"[dim]  V cache quant  : {v_cache} — "
            + ("compressed value cache, saves VRAM" if v_cache == "q8_0" else "full precision value cache, required by some models")
            + "[/dim]"
        )
        console.print(
            f"[dim]  Flash Attention: {'on' if fa else 'off'} — "
            + ("enabled, speeds up attention on compatible GPUs" if fa else "disabled, required for this model's KV config")
            + "[/dim]"
        )
        if gain_pct > 5 and len(successful) > 1:
            console.print(
                f"\n[dim]  Best config was "
                f"[bold]{gain_pct}% faster[/bold] than the slowest "
                f"tested — worth tuning.[/dim]"
            )
        console.print()

        # ── Apply winner + save ───────────────────────────────────
        console.print("[dim]  → Applying winner config...[/dim]")
        try:
            _unload(lm_id)
            time.sleep(2)
            write_config_everywhere(
                lm_id,
                best_cfg["context"],
                best_cfg["k_cache"],
                best_cfg["v_cache"],
                best_cfg["flash_attn"],
            )
            time.sleep(0.5)

            final_attempt = {
                "attempt":     1,
                "context":     best_cfg["context"],
                "k_cache":     best_cfg["k_cache"],
                "v_cache":     best_cfg["v_cache"],
                "flash_attn":  best_cfg["flash_attn"],
                "label":       best_cfg["label"],
                "unload_first": False,
            }
            final = run_load_attempt(lm_id, final_attempt, info)

            if final.get("status") == "ok":
                save_learned_setting(
                    model_id,
                    best_cfg["context"],
                    best_cfg["k_cache"],
                    best_cfg["v_cache"],
                    final.get("elapsed", 0),
                    best_cfg["flash_attn"],
                    config_paths=final.get("confirmed_config_paths"),
                )
                console.print(
                    "[bold green]  ✓ Saved to learned_settings.json[/bold green]"
                )
                console.print(
                    f"[bold green]  ✓ {name} tuned for your RTX 3060[/bold green]"
                )
                console.print(
                    f"[dim]  O key will now use: {best_cfg['label']}[/dim]\n"
                )
            else:
                console.print(
                    "[yellow]  ⚠ Winner load failed — "
                    "settings saved anyway, apply with O key[/yellow]"
                )
                # Still save — next O key press will load correctly
                save_learned_setting(
                    model_id,
                    best_cfg["context"],
                    best_cfg["k_cache"],
                    best_cfg["v_cache"],
                    0,
                    best_cfg["flash_attn"],
                )

        except Exception as e:
            console.print(f"[red]  ✗ Failed to apply winner: {e}[/red]\n")

        # Final sweep — eject any :N orphans left over from this or prior runs
        _cleanup_dups(lm_id)
        show_ready_banner("Tuned — winner config active")

    in_menu = False
    _unknown_key_hint_shown = False

    _CMD_KEYS = (
        b" ", b"x", b"X", b"o", b"O", b"s", b"S",
        b"t", b"T", b"c", b"C", b"f", b"F",
        b"\x1b", b"\r", b"\n",
        b"\x03",  # Ctrl+C
    )

    try:
        while True:
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.1)
                    continue

                key = msvcrt.getch()

                if not in_menu:
                    if key == b" ":
                        in_menu = True
                        show_menu()
                    elif key in (b"x", b"X"):
                        console.print(
                            "\n[bold red]  ⛔ MANUAL ABORT -- "
                            "stopping generation...[/bold red]"
                        )
                        signal_abort()
                        time.sleep(0.5)
                        console.print(
                            "[green]  ✓ Generation "
                            "stopped[/green]\n"
                        )
                    elif key in (b"o", b"O"):
                        do_optimize()
                    elif key in (b"s", b"S"):
                        do_status()
                    elif key in (b"t", b"T"):
                        do_tune()
                    elif key in (b"c", b"C"):
                        do_set_context()
                    elif key in (b"f", b"F"):
                        do_toggle_fa()
                    else:
                        # Any non-command key gets silently eaten by getch().
                        # Show a one-time hint, then echo every keystroke so the
                        # user can at least see what they're typing.
                        if not _unknown_key_hint_shown and key not in _CMD_KEYS:
                            _unknown_key_hint_shown = True
                            console.print(
                                "\n[dim]  (Heads up: this window intercepts single "
                                "keys for commands — anything you type here won't "
                                "do anything. Use SPACE/X/O/T/C/F/S only. Send "
                                "chats from OpenWebUI/Continue/AnythingLLM "
                                "instead. Typed characters are echoed below so "
                                "you can see what you typed.)[/dim]\n"
                            )
                        # Echo printable chars + handle Enter/Backspace so the
                        # terminal feels alive even though input is no-op.
                        try:
                            if key == b"\r" or key == b"\n":
                                sys.stdout.write("\n")
                            elif key == b"\x08":  # backspace
                                sys.stdout.write("\b \b")
                            else:
                                ch = key.decode("utf-8", errors="ignore")
                                if ch and ch.isprintable():
                                    sys.stdout.write(ch)
                            sys.stdout.flush()
                        except Exception:
                            pass
                    continue

                if key in (b"\x1b", b"q", b"Q"):
                    in_menu = False
                    console.print(
                        f"[dim]  Cancelled -- keeping "
                        f"{cfg.preset_name.upper()}[/dim]\n"
                    )
                    continue

                if key in (b"x", b"X"):
                    in_menu = False
                    console.print(
                        "\n[bold red]  ⛔ MANUAL ABORT -- "
                        "stopping generation...[/bold red]"
                    )
                    signal_abort()
                    time.sleep(0.5)
                    console.print(
                        "[green]  ✓ Generation "
                        "stopped[/green]\n"
                    )
                    continue

                char = key.decode("ascii", errors="ignore")
                if char in PRESET_MAP:
                    in_menu = False
                    do_switch(PRESET_MAP[char])
                    continue

            except Exception:
                time.sleep(0.1)

    except KeyboardInterrupt:
        console.print(
            "\n\n[bold yellow]  Shutting down "
            "bridge...[/bold yellow]"
        )
        console.print(
            "[green]  Bridge stopped cleanly.[/green]\n"
        )
        sys.exit(0)


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    flask_thread = threading.Thread(
        target=lambda: app.run(
            host=HOST,
            port=PORT,
            debug=False,
            threaded=True,
            use_reloader=False,
        ),
        daemon=True,
        name="flask-server",
    )
    flask_thread.start()

    time.sleep(1.5)

    console.print(f"""
[bold green]════════════════════════════════════════════════════════[/bold green]
[bold green]  AI Executive Team Bridge v2.0.3 — READY[/bold green]
[bold green]════════════════════════════════════════════════════════[/bold green]

[bold cyan]  Preset:   {cfg.preset_name.upper()}[/bold cyan]
[dim]  Fallback: {cfg.fallback_model}[/dim]
[dim]  Timeout:  {FIRST_TOKEN_TIMEOUT}s (hybrid: 180s)[/dim]

[bold yellow]════════════════════════════════════════════════════════[/bold yellow]
[bold yellow]  CONTROLS[/bold yellow]
[bold yellow]════════════════════════════════════════════════════════[/bold yellow]
[bold white]  SPACE[/bold white]   →  Switch preset (menu appears)
[bold white]  X[/bold white]       →  Abort current generation
[bold white]  O[/bold white]       →  Optimize / reload current model + perf report
[bold white]  T[/bold white]       →  Tune — benchmark configs, save best to learned_settings
[bold white]  C[/bold white]       →  Cycle context window  2048→4096→8192→16384→...
[bold white]  F[/bold white]       →  Toggle Flash Attention on/off (adjusts V cache too)
[bold white]  S[/bold white]       →  Show bridge status
[bold white]  Ctrl+C[/bold white]  →  Stop bridge completely
[bold yellow]════════════════════════════════════════════════════════[/bold yellow]

[bold cyan]  ◉ Ready — waiting for first chat request[/bold cyan]
""")

    run_preset_switcher()