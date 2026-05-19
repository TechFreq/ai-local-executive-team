# routers/local_router.py
# ══════════════════════════════════════════════════════
# Handles all communication with LM Studio and Ollama
#
# v7 — Abort signal support
#
# Changes from v6:
#   - Added signal_abort() / clear_abort() functions
#   - Added abort check inside call_local_stream_safe()
#     fetch thread — closes the HTTP response when abort
#     is signalled so LM Studio stops generating tokens
#   - stop_event now also checks _abort_event so pressing
#     X in terminal or Stop in OpenWebUI both work
# ══════════════════════════════════════════════════════

import json
import time
import queue
import threading
import requests
from rich.console import Console
from core.config_loader import cfg

# ── Reusable HTTP session — connection pooling across all requests ──
_session = requests.Session()

console = Console()

LM_STUDIO_URL  = cfg.lm_studio_url
OLLAMA_URL     = cfg.ollama_url
BACKEND        = cfg.backend
LM_STUDIO_BASE = LM_STUDIO_URL.rstrip("/v1").rstrip("/")

FIRST_TOKEN_TIMEOUT = cfg.first_token_secs
KEEPALIVE_INTERVAL  = cfg.keepalive_interval
HARD_TIMEOUT        = cfg.hard_timeout_secs

_DONE  = object()
_ERROR = object()

KEEPALIVE_SIGNAL = "__KEEPALIVE__"
TIMEOUT_SIGNAL   = "__TIMEOUT__"


# ══════════════════════════════════════════════════════
# ABORT SIGNAL
#
# signal_abort()  — called by bridge when client
#                   disconnects or user presses X
# clear_abort()   — called at start of every new request
#
# The fetch thread checks _abort_event every chunk
# and closes the HTTP response immediately when set.
# This stops LM Studio from generating more tokens.
# ══════════════════════════════════════════════════════

_abort_event = threading.Event()


def signal_abort() -> None:
    """
    Signal all active streaming to stop immediately.
    Safe to call from any thread.
    Closes the HTTP connection to LM Studio so it
    stops generating tokens for the current request.
    """
    _abort_event.set()
    console.print(
        "[bold red]  ⛔ Abort signal received — "
        "closing LM Studio connection[/bold red]"
    )


def clear_abort() -> None:
    """
    Clear the abort signal at the start of a new request.
    Must be called before each new generation starts.
    """
    _abort_event.clear()


# ══════════════════════════════════════════════════════
# CONFIRMED ID MAP
# ══════════════════════════════════════════════════════

CONFIRMED_ID_MAP: dict[str, str] = {
    "qwen/qwen2.5-coder-14b-instruct":       "qwen2.5-coder-14b-instruct",
    "qwen/qwq-32b":                           "qwq-32b",
    "deepseek/deepseek-r1-distill-qwen-32b":  "deepseek-r1-distill-qwen-32b",
    "qwen/qwen2.5-vl-7b-instruct":            "qwen2.5-vl-7b-instruct",
    "nomic-ai/nomic-embed-text-v1.5":         "text-embedding-nomic-embed-text-v1.5",
}

_REVERSE_ID_MAP: dict[str, str] = {
    lm_id: our_id
    for our_id, lm_id in CONFIRMED_ID_MAP.items()
}

_runtime_id_cache: dict[str, str] = {}


def resolve_lm_id(our_id: str) -> str:
    """
    Converts our registry model ID to the LM Studio model ID.

    Order:
    1. Hardcoded confirmed map
    2. Runtime cache
    3. Exact match against /v1/models
    4. Prefix strip match
    5. Fuzzy word overlap match
    6. Return unchanged
    """
    if our_id in CONFIRMED_ID_MAP:
        return CONFIRMED_ID_MAP[our_id]

    if our_id in _runtime_id_cache:
        return _runtime_id_cache[our_id]

    try:
        resp = _session.get(f"{LM_STUDIO_URL}/models", timeout=5)
        if resp.status_code == 200:
            lm_ids = [
                m.get("id", "")
                for m in resp.json().get("data", [])
                if m.get("id")
            ]

            if our_id in lm_ids:
                _runtime_id_cache[our_id] = our_id
                return our_id

            if "/" in our_id:
                without_prefix = our_id.split("/", 1)[1]
                if without_prefix in lm_ids:
                    _runtime_id_cache[our_id] = without_prefix
                    console.print(
                        f"[dim]  ID resolved: {our_id} "
                        f"→ {without_prefix}[/dim]"
                    )
                    return without_prefix

            clean_lm_ids = [
                lid for lid in lm_ids
                if not (
                    ":" in lid
                    and lid.split(":")[-1].isdigit()
                )
            ]
            our_words = set(
                our_id.lower()
                .replace("/", "-")
                .replace("_", "-")
                .split("-")
            )
            best_match = None
            best_score = 0

            for lm_id in clean_lm_ids:
                lm_words = set(
                    lm_id.lower()
                    .replace("/", "-")
                    .replace("_", "-")
                    .split("-")
                )
                our_sig = {w for w in our_words if len(w) > 2}
                lm_sig  = {w for w in lm_words  if len(w) > 2}
                overlap = len(our_sig & lm_sig)
                if overlap > best_score:
                    best_score = overlap
                    best_match = lm_id

            if best_match and best_score >= 3:
                _runtime_id_cache[our_id] = best_match
                console.print(
                    f"[dim]  ID fuzzy resolved: {our_id} "
                    f"→ {best_match} (score={best_score})[/dim]"
                )
                return best_match

    except Exception as e:
        console.print(
            f"[yellow]  ID resolution failed for {our_id}: "
            f"{e}[/yellow]"
        )

    # Dynamic fallback: strip publisher prefix rather than returning a 404-prone ID.
    # Models downloaded from non-lmstudio-community HF orgs (Qwen, DeepSeek, etc.)
    # appear in LM Studio as bare name with no publisher prefix.
    if "/" in our_id:
        name_only = our_id.split("/", 1)[1]
        console.print(
            f"[dim]  ID fallback: {our_id} -> {name_only}[/dim]"
        )
        return name_only

    return our_id


def our_id_from_lm_id(lm_id: str) -> str:
    base_lm_id = lm_id.split(":")[0] if ":" in lm_id else lm_id

    if base_lm_id in _REVERSE_ID_MAP:
        return _REVERSE_ID_MAP[base_lm_id]

    for our_id, mapped_lm_id in _runtime_id_cache.items():
        if mapped_lm_id == base_lm_id:
            return our_id

    return base_lm_id


# ══════════════════════════════════════════════════════
# LOADED STATE — via /api/v0/models
# ══════════════════════════════════════════════════════

_instance_id_cache: dict[str, str] = {}

# TTL cache for LM Studio state — avoids repeated HTTP calls
# within the same request cycle (preflight + ensure_model_loaded)
_state_cache:      dict[str, dict] = {}
_state_cache_time: float           = 0.0
_STATE_CACHE_TTL:  float           = 2.5   # seconds


def _invalidate_state_cache() -> None:
    """Force the next get_lm_studio_state() call to hit LM Studio.
    Call this after any load or eject so callers see the new state immediately."""
    global _state_cache_time
    _state_cache_time = 0.0


def get_lm_studio_state() -> dict[str, dict]:
    """
    Queries /api/v0/models for full model state.
    Returns dict keyed by our model ID.
    Results are cached for _STATE_CACHE_TTL seconds to avoid
    redundant HTTP calls within the same request cycle.
    """
    global _state_cache, _state_cache_time
    now = time.monotonic()
    if now - _state_cache_time < _STATE_CACHE_TTL and _state_cache:
        return _state_cache

    try:
        resp = _session.get(
            f"{LM_STUDIO_BASE}/api/v0/models",
            timeout=5,
        )
        if resp.status_code != 200:
            console.print(
                f"[yellow]  ⚠ /api/v0/models returned "
                f"{resp.status_code}[/yellow]"
            )
            return {}

        data   = resp.json()
        models = data.get("data", [])
        result = {}

        for m in models:
            lm_id = m.get("id", "")
            if not lm_id:
                continue

            state  = m.get("state", "not-loaded")
            is_dup = ":" in lm_id and lm_id.split(":")[-1].isdigit()
            our_id = our_id_from_lm_id(lm_id)

            result[our_id] = {
                "lm_id":          lm_id,
                "state":          state,
                "type":           m.get("type", "llm"),
                "arch":           m.get("arch", ""),
                "loaded_context": m.get("loaded_context_length", 0),
                "is_duplicate":   is_dup,
                "capabilities":   m.get("capabilities", []),
            }

        _state_cache      = result
        _state_cache_time = now
        return result

    except Exception as e:
        console.print(
            f"[yellow]  ⚠ Could not get LM Studio state: {e}[/yellow]"
        )
        return {}


def get_loaded_models() -> list[str]:
    state  = get_lm_studio_state()
    return [
        our_id for our_id, info in state.items()
        if info["state"] == "loaded" and not info["is_duplicate"]
    ]


def get_loaded_duplicates() -> list[dict]:
    state = get_lm_studio_state()
    return [
        info for info in state.values()
        if info["state"] == "loaded" and info["is_duplicate"]
    ]


def get_primary_loaded_model() -> str | None:
    loaded = get_loaded_models()
    return loaded[0] if loaded else None


def is_model_loaded(our_id: str) -> bool:
    state = get_lm_studio_state()
    info  = state.get(our_id)
    if info:
        return info["state"] == "loaded" and not info["is_duplicate"]
    return False


# ══════════════════════════════════════════════════════
# COMPATIBLE MODEL GROUPS
# ══════════════════════════════════════════════════════

COMPATIBLE_GROUPS = [
    {
        "deepseek/deepseek-r1-distill-qwen-32b",
        "qwen/qwq-32b",
        "microsoft/phi-4-reasoning-plus",
        "deepseek/deepseek-r1-0528-qwen3-8b",
    },
    {
        "qwen/qwen3-coder-30b",
        "qwen/qwen2.5-coder-14b-instruct",
    },
    {
        "qwen/qwen3.5-9b",
        "google/gemma-3-12b",
        "google/gemma-4-e2b",
    },
    {
        "google/gemma-4-31b",
        "google/gemma-4-26b-a4b",
    },
]

EJECT_WAIT_SECS      = 2
LOAD_CONFIRM_TIMEOUT = 60


def models_are_compatible(model_a: str, model_b: str) -> bool:
    if model_a == model_b:
        return True
    for group in COMPATIBLE_GROUPS:
        if model_a in group and model_b in group:
            return True
    return False


def find_compatible_loaded(requested_model: str) -> str | None:
    loaded = get_loaded_models()
    if requested_model in loaded:
        return requested_model
    for loaded_model in loaded:
        if models_are_compatible(loaded_model, requested_model):
            return loaded_model
    return None


# ══════════════════════════════════════════════════════
# EJECT MODEL
# ══════════════════════════════════════════════════════

def eject_model(our_id: str) -> bool:
    lm_id = resolve_lm_id(our_id)

    if not is_model_loaded(our_id):
        console.print(
            f"[dim]  {our_id} is not loaded — nothing to eject[/dim]"
        )
        return True

    instance_id = _instance_id_cache.get(our_id)

    if instance_id:
        try:
            resp = requests.post(
                f"{LM_STUDIO_BASE}/api/v1/models/unload",
                json={"instance_id": instance_id},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                console.print(
                    f"[dim]  ✓ Ejected {our_id} "
                    f"(instance_id={instance_id[:8]}...)[/dim]"
                )
                _instance_id_cache.pop(our_id, None)
                _invalidate_state_cache()
                time.sleep(EJECT_WAIT_SECS)
                return True
            else:
                console.print(
                    f"[yellow]  Unload returned {resp.status_code}: "
                    f"{resp.text[:100]}[/yellow]"
                )
        except Exception as e:
            console.print(
                f"[yellow]  instance_id unload failed: {e}[/yellow]"
            )
    else:
        console.print(
            f"[yellow]  No instance_id for {our_id} — "
            f"using TTL trick[/yellow]"
        )

    try:
        requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model":             lm_id,
                "messages":          [{"role": "user", "content": "x"}],
                "max_tokens":        1,
                "temperature":       0,
                "model_ttl_seconds": 0,
            },
            timeout=15,
        )
        console.print(
            f"[dim]  ✓ Ejected {our_id} (TTL trick)[/dim]"
        )
        _instance_id_cache.pop(our_id, None)
        _invalidate_state_cache()
        time.sleep(EJECT_WAIT_SECS)
        return True
    except Exception as e:
        console.print(f"[red]  ✗ Could not eject {our_id}: {e}[/red]")

    return False


def eject_duplicate(lm_id: str, instance_id: str = None) -> bool:
    console.print(
        f"[yellow]  Auto-ejecting duplicate: {lm_id}[/yellow]"
    )

    if instance_id:
        try:
            resp = requests.post(
                f"{LM_STUDIO_BASE}/api/v1/models/unload",
                json={"instance_id": instance_id},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                console.print(
                    f"[dim]  ✓ Ejected duplicate {lm_id}[/dim]"
                )
                return True
        except Exception:
            pass

    try:
        requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model":             lm_id,
                "messages":          [{"role": "user", "content": "x"}],
                "max_tokens":        1,
                "temperature":       0,
                "model_ttl_seconds": 0,
            },
            timeout=15,
        )
        console.print(
            f"[dim]  ✓ Ejected duplicate {lm_id} (TTL trick)[/dim]"
        )
        return True
    except Exception as e:
        console.print(
            f"[yellow]  Could not eject duplicate {lm_id}: {e}[/yellow]"
        )
    return False


def cleanup_duplicates():
    duplicates = get_loaded_duplicates()
    if not duplicates:
        return
    console.print(
        f"[yellow]  Found {len(duplicates)} duplicate instance(s) — "
        f"cleaning up...[/yellow]"
    )
    for dup in duplicates:
        eject_duplicate(dup["lm_id"])
        time.sleep(1)


# ══════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════

def load_model(our_id: str) -> bool:
    if is_model_loaded(our_id):
        console.print(
            f"[green]  ✓ {our_id} already loaded — skipping[/green]"
        )
        return True

    cleanup_duplicates()

    lm_id = resolve_lm_id(our_id)
    console.print(f"[dim]  Loading: {our_id}[/dim]")
    if lm_id != our_id:
        console.print(f"[dim]    LM Studio ID: {lm_id}[/dim]")

    try:
        resp = requests.post(
            f"{LM_STUDIO_BASE}/api/v1/models/load",
            json={"model": lm_id},
            timeout=LOAD_CONFIRM_TIMEOUT,
        )

        if resp.status_code in (200, 201):
            console.print(f"[dim]  ✓ Loaded {our_id}[/dim]")
            try:
                rdata       = resp.json()
                instance_id = (
                    rdata.get("instance_id") or
                    rdata.get("data", {}).get("instance_id", "")
                )
                if instance_id:
                    _instance_id_cache[our_id] = instance_id
                    console.print(
                        f"[dim]    instance_id cached: "
                        f"{instance_id[:8]}...[/dim]"
                    )
                else:
                    console.print(
                        f"[dim]    No instance_id — "
                        f"unload will use TTL trick[/dim]"
                    )
            except Exception:
                pass
            _invalidate_state_cache()
            return True
        else:
            console.print(
                f"[yellow]  Load endpoint: {resp.status_code} — "
                f"{resp.text[:150]}[/yellow]"
            )

    except Exception as e:
        console.print(f"[yellow]  Load endpoint failed: {e}[/yellow]")

    try:
        console.print(f"[dim]  Auto-load trigger for {lm_id}...[/dim]")
        requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model":       lm_id,
                "messages":    [{"role": "user", "content": "ready"}],
                "max_tokens":  1,
                "temperature": 0,
            },
            timeout=LOAD_CONFIRM_TIMEOUT,
        )
        console.print(
            f"[dim]  ✓ Loaded {our_id} (auto-load trigger)[/dim]"
        )
        _invalidate_state_cache()
        return True
    except Exception as e:
        console.print(f"[yellow]  Auto-load trigger failed: {e}[/yellow]")

    console.print(
        f"[yellow]  ⚠ Could not confirm load of {our_id}[/yellow]"
    )
    return False


# ══════════════════════════════════════════════════════
# SMART LOADER
# ══════════════════════════════════════════════════════

def ensure_model_loaded(
    requested_model: str,
    force_exact: bool = False,
) -> str:
    state  = get_lm_studio_state()
    loaded = [
        our_id for our_id, info in state.items()
        if info["state"] == "loaded" and not info["is_duplicate"]
    ]

    console.print(
        f"[dim]  Currently loaded ({len(loaded)}): "
        f"{', '.join(loaded) if loaded else 'none'}[/dim]"
    )

    duplicates = [
        info["lm_id"] for info in state.values()
        if info["is_duplicate"] and info["state"] == "loaded"
    ]
    if duplicates:
        console.print(
            f"[yellow]  Duplicate instances detected: "
            f"{duplicates}[/yellow]"
        )

    if requested_model in loaded:
        console.print(
            f"[green]  ✓ {requested_model} already loaded[/green]"
        )
        return requested_model

    if not force_exact:
        for loaded_model in loaded:
            if models_are_compatible(loaded_model, requested_model):
                console.print(
                    f"[cyan]  ✓ Using {loaded_model} "
                    f"(compatible with {requested_model})[/cyan]"
                )
                return loaded_model

    console.print(
        f"[yellow]  {requested_model} not loaded → "
        f"loading...[/yellow]"
    )

    cleanup_duplicates()

    large_hybrid = {
        "google/gemma-4-31b",
        "qwen/qwen3-coder-30b",
        "deepseek/deepseek-r1-distill-qwen-32b",
        "google/gemma-4-26b-a4b",
        "qwen/qwq-32b",
    }

    board_models = {
        cfg.ceo_model,
        cfg.cto_model,
        cfg.cfo_model,
        cfg.cpo_model,
        cfg.coo_model,
    }

    if requested_model in large_hybrid:
        for candidate in loaded:
            if candidate not in board_models:
                console.print(
                    f"[yellow]  Freeing VRAM: ejecting "
                    f"{candidate}[/yellow]"
                )
                eject_model(candidate)
                break

    load_model(requested_model)
    return requested_model


# ══════════════════════════════════════════════════════
# URL + BACKEND HELPERS
# ══════════════════════════════════════════════════════

def get_base_url() -> str:
    return OLLAMA_URL if BACKEND == "ollama" else LM_STUDIO_URL


# ══════════════════════════════════════════════════════
# NON-STREAMING CALL
# ══════════════════════════════════════════════════════

def call_local(
    prompt: str,
    system_message: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int    = 2000,
) -> str:
    lm_id    = resolve_lm_id(model)
    base_url = get_base_url()
    payload  = {
        "model":       lm_id,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt},
        ]
    }
    try:
        console.print(f"[dim]  → [{BACKEND}] {lm_id}[/dim]")
        r = _session.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=(10, 900),
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] No local backend available.\n"
            "Fix: LM Studio → Local Server → Start Server"
        )
    except requests.exceptions.Timeout:
        return "[ERROR] Model timed out."
    except Exception as e:
        console.print(f"[red][ERROR] {e}[/red]")
        return f"[ERROR] {e}"


# ══════════════════════════════════════════════════════
# SAFE STREAMING CALL — with abort support
#
# The fetch thread now checks _abort_event on every
# chunk. When set, it calls response.close() which
# drops the TCP connection to LM Studio immediately.
# LM Studio detects the closed connection and stops
# generating tokens — no more wasted compute.
#
# Two ways abort gets triggered:
#   1. User presses Stop in OpenWebUI / Continue
#      → GeneratorExit in stream_agent
#      → signal_abort() called
#      → _abort_event set
#      → fetch thread closes response next chunk
#
#   2. User presses X in terminal
#      → signal_abort() called directly
#      → same path from step above
# ══════════════════════════════════════════════════════

def call_local_stream_safe(
    prompt: str,
    system_message: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int    = 2000,
    timeout: int       = None,
):
    """
    Streaming with keepalive, per-model timeout, and abort support.
    """
    first_token_timeout = (
        timeout if timeout is not None else FIRST_TOKEN_TIMEOUT
    )
    lm_id        = resolve_lm_id(model)
    result_queue = queue.Queue()
    stop_event   = threading.Event()

    def fetch():
        payload = {
            "model":       lm_id,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      True,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": prompt},
            ]
        }
        response = None
        try:
            console.print(f"[dim]    [{BACKEND}] {lm_id}[/dim]")
            response = _session.post(
                f"{get_base_url()}/chat/completions",
                json=payload,
                stream=True,
                timeout=(10, 600),
            )
            response.raise_for_status()

            for line in response.iter_lines():
                # ── Abort check ────────────────────────────────────
                # Check both the local stop_event (timeout)
                # and the global _abort_event (client disconnect / X key)
                if stop_event.is_set() or _abort_event.is_set():
                    console.print(
                        f"[yellow]  ⚡ Fetch thread stopping "
                        f"for {lm_id}[/yellow]"
                    )
                    try:
                        response.close()  # drops TCP → LM Studio stops
                    except Exception:
                        pass
                    break
                # ── End abort check ────────────────────────────────

                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                data = decoded[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk   = json.loads(data)
                    content = (
                        chunk.get("choices", [{}])[0]
                             .get("delta", {})
                             .get("content", "")
                    )
                    if content:
                        result_queue.put(content)
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            # Don't report error if we aborted intentionally
            if not (stop_event.is_set() or _abort_event.is_set()):
                result_queue.put((_ERROR, str(e)))
        finally:
            # Always close response on exit
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            result_queue.put(_DONE)

    thread = threading.Thread(target=fetch, daemon=True)
    thread.start()

    elapsed         = 0
    got_first_token = False

    while elapsed < HARD_TIMEOUT:
        # ── Check abort in the main yield loop too ─────────────────
        if _abort_event.is_set():
            console.print(
                f"[yellow]  ⚡ Abort detected in stream loop "
                f"for {lm_id} — stopping yield[/yellow]"
            )
            stop_event.set()
            break
        # ── End abort check ────────────────────────────────────────

        try:
            item = result_queue.get(timeout=KEEPALIVE_INTERVAL)

            if item is _DONE:
                break

            if isinstance(item, tuple) and item[0] is _ERROR:
                # Only surface error if not an intentional abort
                if not _abort_event.is_set():
                    yield f"[ERROR] {item[1]}"
                break

            if not got_first_token:
                console.print(
                    f"[dim]    First token from {lm_id} "
                    f"after {elapsed:.0f}s[/dim]"
                )
            got_first_token = True
            yield item

        except queue.Empty:
            elapsed += KEEPALIVE_INTERVAL

            # Check abort during empty wait too
            if _abort_event.is_set():
                stop_event.set()
                break

            if not got_first_token and elapsed >= first_token_timeout:
                console.print(
                    f"[yellow]  ⏱ No response from {lm_id} "
                    f"after {elapsed:.0f}s — signalling fallback[/yellow]"
                )
                stop_event.set()
                yield TIMEOUT_SIGNAL
                break

            yield KEEPALIVE_SIGNAL

    thread.join(timeout=10)


# ══════════════════════════════════════════════════════
# BASIC STREAMING (no timeout, no abort)
# ══════════════════════════════════════════════════════

def call_local_stream(
    prompt: str,
    system_message: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int    = 2000,
):
    lm_id   = resolve_lm_id(model)
    payload = {
        "model":       lm_id,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      True,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt},
        ]
    }
    try:
        response = _session.post(
            f"{get_base_url()}/chat/completions",
            json=payload,
            stream=True,
            timeout=(10, 600),
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data = decoded[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = (
                        chunk.get("choices", [{}])[0]
                             .get("delta", {})
                             .get("content", "")
                    )
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.ConnectionError:
        yield "[ERROR] Cannot connect to LM Studio."
    except requests.exceptions.Timeout:
        yield "[ERROR] Model timed out."
    except Exception as e:
        yield f"[ERROR] {e}"


# ══════════════════════════════════════════════════════
# HEALTH CHECKS
# ══════════════════════════════════════════════════════

def check_lm_studio_health() -> bool:
    try:
        r = _session.get(f"{LM_STUDIO_URL}/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_ollama_health() -> bool:
    try:
        r = _session.get(f"{OLLAMA_URL}/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_any_local_health() -> bool:
    return check_lm_studio_health() or check_ollama_health()


def list_local_models() -> list:
    for url in [LM_STUDIO_URL, OLLAMA_URL]:
        try:
            r = _session.get(f"{url}/models", timeout=3)
            if r.status_code == 200:
                return [
                    m["id"]
                    for m in r.json().get("data", [])
                ]
        except Exception:
            continue
    return []


def get_loaded_model_status() -> dict:
    """
    Returns loaded model status for the /health endpoint.
    """
    loaded = get_loaded_models()
    state  = get_lm_studio_state()

    loaded_details = {
        our_id: {
            "lm_id":   info["lm_id"],
            "context": info["loaded_context"],
            "type":    info["type"],
        }
        for our_id, info in state.items()
        if info["state"] == "loaded" and not info["is_duplicate"]
    }

    duplicates = [
        info["lm_id"]
        for info in state.values()
        if info["is_duplicate"] and info["state"] == "loaded"
    ]

    return {
        "loaded_models":    loaded,
        "count":            len(loaded),
        "primary":          loaded[0] if loaded else None,
        "details":          loaded_details,
        "duplicates":       duplicates,
        "id_map_overrides": {
            k: v for k, v in CONFIRMED_ID_MAP.items()
        },
    }