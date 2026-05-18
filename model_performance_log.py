# model_performance_log.py
# ══════════════════════════════════════════════════════
# Tracks model performance — learns timeouts, speeds,
# reliability, and TTFT from real usage over time.
#
# New in v2:
#   - Tracks timeout_count per model
#   - Tracks avg_ttft_secs (time-to-first-token)
#   - get_suggested_timeout() — learned timeout from TTFT
#   - get_timeout_rate()      — fraction of runs that timed out
#   - get_reliability_score() — composite score (speed × success × no-timeouts)
#   - get_best_model_for_role() now penalises high-timeout models
#   - In-memory cache — no disk I/O on every generation
# ══════════════════════════════════════════════════════

import json
import time
import atexit
import threading
from pathlib import Path
from rich.console import Console
from core.config_loader import cfg

console  = Console()
LOG_FILE = Path(cfg.performance_file)

# ── In-memory cache — avoids disk read/write on every generation ──
_log_cache:   dict | None = None
_cache_dirty: bool        = False
_cache_lock                = threading.Lock()


def _load_from_disk() -> dict:
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _flush_to_disk():
    """Write the in-memory cache to disk. Called at exit and on demand."""
    global _cache_dirty
    with _cache_lock:
        if _log_cache is not None and _cache_dirty:
            try:
                with open(LOG_FILE, "w") as f:
                    json.dump(_log_cache, f, indent=2)
                _cache_dirty = False
            except Exception as e:
                console.print(
                    f"[yellow]  ⚠ Could not flush perf log: {e}[/yellow]"
                )


# Flush on clean exit so no data is lost
atexit.register(_flush_to_disk)


def load_log() -> dict:
    global _log_cache
    with _cache_lock:
        if _log_cache is None:
            _log_cache = _load_from_disk()
        return _log_cache


def save_log(log: dict):
    """Mark cache dirty — actual write happens at exit or on demand."""
    global _log_cache, _cache_dirty
    with _cache_lock:
        _log_cache   = log
        _cache_dirty = True


def _ensure_entry(log: dict, model_id: str, role: str) -> dict:
    """Return the log entry for model_id, creating it if missing."""
    if model_id not in log:
        log[model_id] = {
            "role":            role,
            "total_runs":      0,
            "successful_runs": 0,
            "timeout_count":   0,
            "total_tokens":    0,
            "total_duration":  0.0,
            "avg_tok_per_sec": 0.0,
            "ttft_samples":    0,
            "avg_ttft_secs":   0.0,
            "first_used":      time.time(),
            "last_used":       time.time(),
            "preset":          cfg.preset_name,
        }
    else:
        # Back-fill new fields into old entries
        entry = log[model_id]
        entry.setdefault("timeout_count",   0)
        entry.setdefault("ttft_samples",    0)
        entry.setdefault("avg_ttft_secs",   0.0)
    return log[model_id]


# ══════════════════════════════════════════════════════
# RECORD FUNCTIONS
# ══════════════════════════════════════════════════════

def record_generation(
    model_id:      str,
    role:          str,
    tokens:        int,
    duration_secs: float,
    success:       bool,
):
    if not cfg.track_performance:
        return

    log   = load_log()
    entry = _ensure_entry(log, model_id, role)

    entry["total_runs"]     += 1
    entry["total_tokens"]   += tokens
    entry["total_duration"] += duration_secs
    entry["last_used"]       = time.time()
    entry["role"]            = role
    entry["preset"]          = cfg.preset_name

    if success:
        entry["successful_runs"] += 1

    if entry["total_duration"] > 0:
        entry["avg_tok_per_sec"] = round(
            entry["total_tokens"] / entry["total_duration"], 2
        )

    save_log(log)


def record_timeout(model_id: str, role: str) -> None:
    """
    Record that a model timed out without producing any output.
    This is tracked separately so get_suggested_timeout() can
    learn that a model's timeout value needs to be higher.
    """
    if not cfg.track_performance:
        return

    log   = load_log()
    entry = _ensure_entry(log, model_id, role)

    entry["timeout_count"] += 1
    entry["total_runs"]    += 1
    entry["last_used"]      = time.time()
    entry["role"]           = role

    console.print(
        f"[dim]  📊 Recorded timeout for {model_id} "
        f"(total: {entry['timeout_count']})[/dim]"
    )
    save_log(log)


def record_ttft(model_id: str, ttft_secs: float) -> None:
    """
    Record time-to-first-token for a model.
    Uses a running average so the learned value improves over time.
    """
    if not cfg.track_performance:
        return
    if ttft_secs <= 0:
        return

    log   = load_log()
    entry = log.get(model_id)
    if entry is None:
        return  # don't create an entry just for TTFT

    entry.setdefault("ttft_samples",  0)
    entry.setdefault("avg_ttft_secs", 0.0)

    samples = entry["ttft_samples"]
    avg     = entry["avg_ttft_secs"]

    # Running average — new data weighs more (recency bias)
    weight = min(samples, 10)   # cap old weight so recent runs matter more
    entry["avg_ttft_secs"] = round(
        (avg * weight + ttft_secs) / (weight + 1), 2
    )
    entry["ttft_samples"] = samples + 1

    save_log(log)


# ══════════════════════════════════════════════════════
# QUERY FUNCTIONS — used by the bridge server
# ══════════════════════════════════════════════════════

def get_suggested_timeout(model_id: str, floor: int = 30) -> int | None:
    """
    Returns a learned timeout for a model based on observed TTFT.
    Formula: avg_ttft × 4  (generous margin), minimum = floor.

    Returns None if fewer than 3 TTFT samples exist
    (not enough data to trust the learned value yet).

    The bridge server uses this to supplement hardcoded MODEL_TIMEOUTS
    so timeouts automatically improve with use.
    """
    log     = load_log()
    entry   = log.get(model_id, {})
    samples = entry.get("ttft_samples", 0)

    if samples < 3:
        return None

    avg_ttft  = entry.get("avg_ttft_secs", 0.0)
    suggested = max(int(avg_ttft * 4), floor)
    return suggested


def get_timeout_rate(model_id: str) -> float:
    """
    Returns the fraction of runs that timed out (0.0 – 1.0).
    A rate above 0.3 means the model is unreliable at its current timeout.
    """
    log   = load_log()
    entry = log.get(model_id, {})
    runs  = entry.get("total_runs", 0)
    if runs == 0:
        return 0.0
    return entry.get("timeout_count", 0) / runs


def get_reliability_score(model_id: str) -> float:
    """
    Composite reliability score (0.0 – ∞, higher is better).
    Combines speed × success_rate × (1 - timeout_penalty).
    Used by get_best_model_for_role() to rank models.
    """
    log   = load_log()
    entry = log.get(model_id, {})
    runs  = entry.get("total_runs", 0)
    if runs < 3:
        return -1.0

    success_rate  = entry.get("successful_runs", 0) / runs
    tok_per_sec   = entry.get("avg_tok_per_sec",  0.0)
    timeout_rate  = entry.get("timeout_count", 0) / runs

    # Penalise timeouts heavily — a model that times out 50% of the
    # time is much less useful even if it's fast when it does respond
    timeout_penalty = min(timeout_rate * 2.0, 1.0)
    return success_rate * tok_per_sec * (1.0 - timeout_penalty)


def get_best_model_for_role(role: str) -> str | None:
    """
    Returns the model with the best reliability score for a given role.
    Requires at least 3 runs. Penalises high-timeout models.
    """
    log        = load_log()
    best_model = None
    best_score = -1.0

    for model_id, stats in log.items():
        if stats.get("role") != role:
            continue
        score = get_reliability_score(model_id)
        if score > best_score:
            best_score = score
            best_model = model_id

    return best_model


# ══════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════

def print_performance_report():
    log = load_log()

    if not log:
        console.print("[dim]No performance data yet.[/dim]")
        return

    console.print("\n[bold cyan]Model Performance Report[/bold cyan]")
    console.print(f"[dim]Active preset: {cfg.preset_name}[/dim]\n")

    # Sort by reliability score descending
    sorted_models = sorted(
        log.items(),
        key=lambda x: get_reliability_score(x[0]),
        reverse=True,
    )

    for model_id, stats in sorted_models:
        runs         = max(stats.get("total_runs", 0), 1)
        success_rate = stats.get("successful_runs", 0) / runs * 100
        timeout_rate = stats.get("timeout_count",   0) / runs * 100
        reliability  = get_reliability_score(model_id)
        suggested_to = get_suggested_timeout(model_id)
        avg_ttft     = stats.get("avg_ttft_secs", 0.0)

        timeout_color = (
            "red"    if timeout_rate > 30 else
            "yellow" if timeout_rate > 10 else
            "green"
        )

        console.print(
            f"  [cyan]{model_id}[/cyan]\n"
            f"    Role:          {stats.get('role', '?')}\n"
            f"    Runs:          {stats.get('total_runs', 0)}\n"
            f"    Success rate:  {success_rate:.0f}%\n"
            f"    Timeout rate:  [{timeout_color}]{timeout_rate:.0f}%[/{timeout_color}]\n"
            f"    Speed:         {stats.get('avg_tok_per_sec', 0):.1f} tok/s\n"
            f"    Avg TTFT:      {avg_ttft:.1f}s "
            f"({stats.get('ttft_samples', 0)} samples)\n"
            f"    Learned timeout: "
            + (f"{suggested_to}s" if suggested_to else "not enough data yet")
            + f"\n"
            f"    Reliability:   {reliability:.1f}\n"
            f"    Preset:        {stats.get('preset', 'unknown')}\n"
        )


if __name__ == "__main__":
    print_performance_report()
