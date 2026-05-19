# sync_models.py
# ══════════════════════════════════════════════════════
# AUTO-SYNC + AUTO-ASSIGN LM STUDIO MODELS
#
# Detects new/removed models in LM Studio, scores each
# one against every role using preset-aware heuristics,
# and auto-assigns the best fit — no manual editing needed.
#
# Usage:
#   python sync_models.py             → preview only (no writes)
#   python sync_models.py --apply     → update configs + auto-assign roles
#   python sync_models.py --configure → also write LM Studio load settings
#   python sync_models.py --full      → --apply + --configure together
#
# Sources (tried in order):
#   1. LM Studio API at localhost:1234  (most accurate model IDs)
#   2. Filesystem scan of ~/.lmstudio/models  (LM Studio closed)
#
# Auto-assignment rules:
#   - Scores each new model 0-100 against every role
#   - Adjusts scoring based on active preset (fast = prefer GPU,
#     smart/nuclear = prefer quality/size)
#   - Only assigns if new model scores higher than the preset default
#   - Never overwrites a role you've already manually set
#   - Adds everything else as commented options so you can still swap
# ══════════════════════════════════════════════════════

import os
import re
import sys
import json
import subprocess
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from rich.console import Console
    console = Console()
    def p(msg):   console.print(msg)
    def pr(msg):  console.print(msg, style="green")
    def py(msg):  console.print(msg, style="yellow")
    def pd(msg):  console.print(msg, style="dim")
    def pe(msg):  console.print(msg, style="red")
    def pb(msg):  console.print(msg, style="bold")
except ImportError:
    def p(msg):  print(msg)
    def pr(msg): print(msg)
    def py(msg): print(msg)
    def pd(msg): print(msg)
    def pe(msg): print(msg)
    def pb(msg): print(msg)

ROOT             = Path(__file__).parent
CONFIGURE_PY     = ROOT / "configure_models.py"
MY_MODELS        = ROOT / "config" / "my_models.yaml"
CONFIG_YAML      = ROOT / "config.yaml"
PRESETS_DIR      = ROOT / "presets"
LEARNED_SETTINGS = ROOT / "learned_settings.json"
ARCHIVE_FILE     = ROOT / "config" / "model_archive.json"
LMS_URL          = "http://localhost:1234/v1"
LMS_MODELS   = Path.home() / ".lmstudio" / "models"


# ══════════════════════════════════════════════════════
# ROLE DEFINITIONS
# ══════════════════════════════════════════════════════

ALL_ROLES = [
    ("board",   "ceo"),
    ("board",   "cto"),
    ("board",   "cfo"),
    ("board",   "cpo"),
    ("board",   "coo"),
    ("utility", "vision"),
    ("utility", "autocomplete"),
    ("fallback","primary"),
    ("fallback","general"),
    ("fallback","fast_reasoning"),
    ("fallback","last_resort"),
]

ROLE_DISPLAY = {
    "ceo":           "CEO  (strategic lead)",
    "cto":           "CTO  (code & architecture)",
    "cfo":           "CFO  (reasoning & analysis)",
    "cpo":           "CPO  (product & creativity)",
    "coo":           "COO  (operations, fast output)",
    "vision":        "Vision (image analysis)",
    "autocomplete":  "Autocomplete (tiny+fast)",
    "primary":       "Fallback primary",
    "general":       "Fallback general",
    "fast_reasoning":"Fallback fast reasoning",
    "last_resort":   "Fallback last resort",
}

# Publisher inference from HuggingFace org names
PUBLISHER_PREFIXES = {
    "gemma":    "google",
    "llama":    "meta-llama",
    "mistral":  "mistralai",
    "mixtral":  "mistralai",
    "phi":      "microsoft",
    "qwen":     "qwen",
    "deepseek": "deepseek",
    "nemotron": "nvidia",
    "nomic":    "nomic-ai",
    "falcon":   "tiiuae",
    "yi":       "01-ai",
    "stablelm": "stabilityai",
    "codellama":"meta-llama",
    "starcoder":"bigcode",
}

COMMUNITY_ORGS = {
    "lmstudio-community", "unsloth", "bartowski", "TheBloke",
    "mlabonne", "QuantFactory", "MaziyarPanahi", "Donnyed",
    "NousResearch", "openfree",
}


# ══════════════════════════════════════════════════════
# SCORING ENGINE
# ══════════════════════════════════════════════════════

def _model_tags(name: str) -> dict:
    n = name.lower()
    return {
        "is_vision":    any(x in n for x in ["vl-", "-vl", "vision", "vl7", "vl2"]),
        "is_embed":     any(x in n for x in ["embed", "embedding"]),
        "is_coder":     any(x in n for x in ["coder", "-code-", "starcoder"]),
        "is_reasoning": any(x in n for x in ["r1", "qwq", "reasoning", "thinking", "reason"]),
        "is_moe":       any(x in n for x in ["-moe", "moe-", "a3b", "a4b", "-omni"]),
        "is_instruct":  any(x in n for x in ["instruct", "chat", "it"]),
    }


def score_model(name: str, size_gb: float, role: str, preset_name: str) -> tuple:
    """
    Returns (score 0-100, reason string).
    Higher score = better fit for this role in this preset.
    """
    tags    = _model_tags(name)
    gpu_fit = size_gb < 11.5 and size_gb > 0
    reasons = []
    score   = 50

    # ── Hard exclusions ───────────────────────────────
    if role == "vision" and not tags["is_vision"]:
        return 0, "not a vision model"
    if role != "vision" and tags["is_vision"]:
        return 5, "vision model, not useful for text roles"
    if role == "embed" and not tags["is_embed"]:
        return 10, "not an embedding model"
    if tags["is_embed"] and role != "embed":
        return 5, "embedding model, cannot generate text"

    # ── Preset context modifier ────────────────────────
    # Fast presets favour GPU-fit small models
    # Smart/nuclear favour quality/size
    if preset_name in ("fastest",):
        if gpu_fit:
            score += 25; reasons.append("GPU-only (fastest preset)")
        else:
            score -= 40; reasons.append("too large for fastest preset")
    elif preset_name in ("fast",):
        if gpu_fit:
            score += 15; reasons.append("GPU-fit (fast preset)")
        elif size_gb > 0:
            score -= 20; reasons.append("RAM offload slow for fast preset")
    elif preset_name in ("smart", "nuclear"):
        if size_gb >= 15:
            score += 15; reasons.append("large model suits quality preset")
        elif size_gb < 7 and role in ("ceo","cto","cfo","cpo"):
            score -= 25; reasons.append("too small for quality preset board role")

    # ── Role-specific scoring ─────────────────────────

    if role == "ceo":
        # Wants: large, general, good reasoning, NOT coding-specific
        if size_gb >= 25:   score += 30; reasons.append(f"{size_gb:.0f}GB flagship")
        elif size_gb >= 18: score += 20; reasons.append(f"{size_gb:.0f}GB large")
        elif size_gb >= 12: score += 5
        elif size_gb >  0:  score -= 25; reasons.append("too small for CEO")
        if tags["is_coder"] and not tags["is_reasoning"]:
            score -= 15; reasons.append("coder model — CTO is a better fit")
        if tags["is_reasoning"]: score += 8;  reasons.append("reasoning bonus")
        if tags["is_moe"]:       score += 5;  reasons.append("MoE efficiency")

    elif role == "cto":
        # Wants: coder specialization, decent size
        if tags["is_coder"]: score += 35; reasons.append("coder model — ideal for CTO")
        if size_gb >= 18:    score += 15; reasons.append(f"{size_gb:.0f}GB large coder")
        elif size_gb >= 10:  score += 10
        elif size_gb < 6:    score -= 20; reasons.append("too small for CTO")
        if tags["is_reasoning"]: score += 5

    elif role == "cfo":
        # Wants: reasoning first, decent size second
        if tags["is_reasoning"]: score += 38; reasons.append("reasoning model — ideal for CFO")
        if size_gb >= 18:   score += 12; reasons.append(f"{size_gb:.0f}GB heavy reasoner")
        elif size_gb >= 10: score += 5
        elif size_gb < 6:   score -= 25; reasons.append("too small for deep CFO analysis")
        if tags["is_coder"] and not tags["is_reasoning"]:
            score -= 10; reasons.append("pure coder, not ideal for CFO")

    elif role == "cpo":
        # Wants: general quality, creativity, NOT coder-specific, NOT pure reasoner
        if size_gb >= 18:   score += 20; reasons.append(f"{size_gb:.0f}GB quality model")
        elif size_gb >= 12: score += 12
        elif size_gb < 6:   score -= 20; reasons.append("too small for CPO quality bar")
        if tags["is_coder"] and not tags["is_reasoning"]:
            score -= 12; reasons.append("coder model better at CTO/COO")
        if tags["is_reasoning"]: score -= 5; reasons.append("reasoning model less creative")
        if tags["is_moe"]:       score += 8; reasons.append("MoE good balance for CPO")

    elif role == "coo":
        # Wants: fast (GPU-fit), structured output, coder is a plus
        if gpu_fit:          score += 28; reasons.append("GPU-fit = fast COO")
        else:                score -= 18; reasons.append("RAM offload makes COO slow")
        if tags["is_coder"]: score += 20; reasons.append("coder good for structured tasks")
        if size_gb <= 10:    score += 10; reasons.append("small = fast responses")
        if size_gb > 16:     score -= 15; reasons.append("oversized for COO speed requirement")
        if tags["is_reasoning"]: score -= 12; reasons.append("reasoning overhead slows COO")

    elif role == "autocomplete":
        # Wants: smallest possible, GPU-only
        if size_gb < 5:       score += 40; reasons.append("tiny — ideal autocomplete speed")
        elif size_gb < 7:     score += 15
        elif size_gb < 10:    score -= 10
        else:                 score -= 35; reasons.append("too large for autocomplete")
        if not gpu_fit:       score -= 20; reasons.append("must fit GPU for autocomplete")

    elif role == "vision":
        if tags["is_vision"]: score += 50; reasons.append("vision model")

    elif role in ("primary", "fast_reasoning"):
        # Fallback: GPU-fit preferred, reasoning is a bonus
        if gpu_fit:              score += 22; reasons.append("GPU-fit fallback")
        if tags["is_reasoning"]: score += 18; reasons.append("reasoning fallback")
        if size_gb <= 10:        score += 8

    elif role in ("general", "last_resort"):
        # Fast general fallback
        if gpu_fit:    score += 25; reasons.append("GPU-fit")
        if size_gb < 10: score += 10
        if size_gb > 15: score -= 20; reasons.append("too large for fallback speed")

    return max(0, min(100, score)), "; ".join(reasons) if reasons else "general capability"


def best_role_for_model(name: str, size_gb: float, preset_name: str) -> tuple:
    """
    Returns (section, role_key, score, reason) for the single best role.
    Vision/embed models are forced to their dedicated slots.
    """
    tags = _model_tags(name)
    if tags["is_vision"]:
        return "utility", "vision", 99, "vision model — dedicated slot"
    if tags["is_embed"]:
        return "utility", "embed", 99, "embedding model — dedicated slot"

    best_score  = -1
    best_result = ("board", "coo", 0, "default")

    for section, role in ALL_ROLES:
        if role in ("vision",):
            continue
        s, reason = score_model(name, size_gb, role, preset_name)
        if s > best_score:
            best_score  = s
            best_result = (section, role, s, reason)

    return best_result


def score_known_model(model_id: str, role: str, preset_name: str) -> int:
    """
    Score an existing model (from the preset) using the same function.
    Infers size from the model name since we don't always have it on disk.
    """
    SIZE_HINTS = {
        "gemma-4-31b": 19.9,  "gemma-4-26b": 18.0,   "gemma-4-e2b": 4.4,
        "gemma-4-e4b": 6.3,   "gemma-3-12b": 8.2,    "qwen3-coder-30b": 18.6,
        "qwen3.6-27b": 17.5,  "qwen3.5-9b": 6.5,     "qwen2.5-coder-14b": 9.0,
        "qwen2.5-vl-7b": 7.2, "qwq-32b": 19.9,       "deepseek-r1-distill-qwen-32b": 19.9,
        "deepseek-r1-0528-qwen3-8b": 5.0,             "phi-4-reasoning-plus": 9.1,
        "nemotron-3-nano-omni": 26.1,                  "nomic-embed": 0.3,
    }
    name  = model_id.split("/")[-1].lower()
    size  = 0.0
    for hint_key, hint_size in SIZE_HINTS.items():
        if hint_key.lower() in name:
            size = hint_size
            break
    if size == 0:
        # Guess from size tag in name
        for tag, gb in [("70b",43),("32b",20),("31b",20),("30b",19),
                        ("27b",18),("26b",18),("14b",9),("12b",8),
                        ("9b",7),("8b",5),("7b",5),("4b",4),("3b",3)]:
            if tag in name:
                size = gb; break
    score, _ = score_model(name, size, role, preset_name)
    return score


# ══════════════════════════════════════════════════════
# CONFIG READERS
# ══════════════════════════════════════════════════════

def get_active_preset_name() -> str:
    if not CONFIG_YAML.exists() or not HAS_YAML:
        return "balanced"
    try:
        with open(CONFIG_YAML, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("active_preset", "balanced")
    except Exception:
        return "balanced"


def get_preset_models(preset_name: str) -> dict:
    """Returns {role_key: model_id} for the given preset."""
    preset_file = PRESETS_DIR / f"{preset_name}.yaml"
    if not preset_file.exists() or not HAS_YAML:
        return {}
    try:
        with open(preset_file, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        result = {}
        for section in ("board", "utility", "fallback"):
            for role, model in (data.get(section) or {}).items():
                result[role] = model
        return result
    except Exception:
        return {}


def get_active_overrides() -> dict:
    """
    Reads config/my_models.yaml and returns {role_key: model_id}
    for roles that have ACTIVE (uncommented) assignments.
    """
    if not MY_MODELS.exists():
        return {}
    overrides = {}
    try:
        lines = MY_MODELS.read_text(encoding="utf-8").splitlines()
        current_section = None
        for line in lines:
            stripped = line.strip()
            # Track section headers
            if stripped in ("board:", "utility:", "fallback:"):
                current_section = stripped.rstrip(":")
                continue
            # Skip comments and blank lines
            if stripped.startswith("#") or not stripped:
                continue
            # Active assignment: "  role: model/id"
            if ":" in stripped and current_section:
                parts = stripped.split(":", 1)
                role  = parts[0].strip()
                model = parts[1].strip()
                if model and not model.startswith("#"):
                    overrides[role] = model
    except Exception:
        pass
    return overrides


def get_configured_models() -> list:
    if not CONFIGURE_PY.exists():
        return []
    text = CONFIGURE_PY.read_text(encoding="utf-8")
    match = re.search(r'YOUR_MODELS\s*=\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return []
    pairs = re.findall(
        r'"publisher"\s*:\s*"([^"]+)".*?"name"\s*:\s*"([^"]+)"',
        match.group(1)
    )
    return [f"{pub}/{name}" for pub, name in pairs]


def get_vram_table() -> dict:
    if not CONFIGURE_PY.exists():
        return {}
    text  = CONFIGURE_PY.read_text(encoding="utf-8")
    match = re.search(r'MODEL_WEIGHT_VRAM_MB\s*=\s*\{(.*?)\}', text, re.DOTALL)
    if not match:
        return {}
    result = {}
    for m in re.finditer(r'"([^"]+)"\s*:\s*(\d+)', match.group(1)):
        result[m.group(1)] = int(m.group(2))
    return result


# ══════════════════════════════════════════════════════
# DISCOVERY
# ══════════════════════════════════════════════════════

def discover_from_api():
    if not HAS_REQUESTS:
        return None
    try:
        resp = requests.get(f"{LMS_URL}/models", timeout=3)
        if resp.status_code != 200:
            return None
        data   = resp.json().get("data", [])
        models = [{"id": m["id"], "size_gb": 0.0, "source": "api"}
                  for m in data if m.get("id")]
        return models or None
    except Exception:
        return None


def _infer_publisher(hf_org: str, model_name: str) -> str:
    if hf_org not in COMMUNITY_ORGS:
        return hf_org
    for prefix, pub in PUBLISHER_PREFIXES.items():
        if model_name.lower().startswith(prefix):
            return pub
    return hf_org


def discover_from_filesystem() -> list:
    if not LMS_MODELS.exists():
        return []
    found = []
    for hf_org in LMS_MODELS.iterdir():
        if not hf_org.is_dir() or hf_org.name.startswith("."):
            continue
        for model_dir in hf_org.iterdir():
            if not model_dir.is_dir():
                continue
            has_gguf = any(f.suffix.lower() == ".gguf"
                           for f in model_dir.iterdir())
            if not has_gguf:
                continue
            model_name = model_dir.name
            publisher  = _infer_publisher(hf_org.name, model_name)
            size_gb    = sum(
                f.stat().st_size for f in model_dir.iterdir()
                if f.suffix.lower() == ".gguf"
            ) / (1024 ** 3)
            found.append({
                "id":      f"{publisher}/{model_name}",
                "size_gb": round(size_gb, 1),
                "source":  "fs",
            })
    return found


# ══════════════════════════════════════════════════════
# DIFF
# ══════════════════════════════════════════════════════

def _normalise(model_id: str) -> str:
    mid = model_id.lower()
    if "/" in mid:
        mid = mid.split("/", 1)[1]
    mid = re.sub(r'-q\d[_k_smq0-9]*$', '', mid)
    mid = re.sub(r'-(gguf|ggml)$', '', mid)
    mid = re.sub(r'^text-embedding-', '', mid)
    return mid.strip("-")


def diff_models(discovered: list, configured: list) -> tuple:
    disc_norm = {_normalise(m["id"]): m for m in discovered}
    conf_norm = {_normalise(cid): cid  for cid in configured}
    new_keys  = set(disc_norm) - set(conf_norm)
    rem_keys  = set(conf_norm) - set(disc_norm)
    return ([disc_norm[k] for k in new_keys],
            [conf_norm[k] for k in rem_keys])


# ══════════════════════════════════════════════════════
# ASSIGNMENT PLANNER
# ══════════════════════════════════════════════════════

def plan_assignments(new_models: list, preset_name: str) -> list:
    """
    For each new model, decide:
      - best_role      : the role it scores highest in
      - score          : how well it fits (0-100)
      - reason         : why
      - preset_score   : how the preset's current model scores in that same role
      - should_assign  : True if new model clearly beats the preset default
                         and no manual override exists for that role
    Returns list of assignment dicts.
    """
    preset_models    = get_preset_models(preset_name)
    active_overrides = get_active_overrides()
    plans            = []

    for model in new_models:
        name    = model["id"].split("/")[-1]
        size_gb = model["size_gb"]

        section, role, score, reason = best_role_for_model(
            name, size_gb, preset_name
        )

        # Score the current preset model in the same role
        current_model = preset_models.get(role, "")
        preset_score  = score_known_model(current_model, role, preset_name) \
                        if current_model else 0

        # Assign if: new model clearly wins AND role not manually set
        role_overridden = role in active_overrides
        wins_by         = score - preset_score
        should_assign   = (
            not role_overridden
            and score >= 55         # minimum quality bar
            and wins_by >= 8        # meaningful improvement
        )

        plans.append({
            "model":          model,
            "section":        section,
            "role":           role,
            "score":          score,
            "reason":         reason,
            "current_model":  current_model,
            "preset_score":   preset_score,
            "wins_by":        wins_by,
            "role_overridden":role_overridden,
            "should_assign":  should_assign,
        })

    return plans


# ══════════════════════════════════════════════════════
# FILE UPDATERS
# ══════════════════════════════════════════════════════

def _vram_estimate(size_gb: float) -> int:
    return int(size_gb * 1024)


def update_configure_models(new_models: list) -> bool:
    if not new_models:
        return True
    text       = CONFIGURE_PY.read_text(encoding="utf-8")
    vram_table = get_vram_table()

    # Add to MODEL_WEIGHT_VRAM_MB
    vram_lines = []
    for m in new_models:
        name = m["id"].split("/")[-1]
        if name not in vram_table:
            vram_mb = _vram_estimate(m["size_gb"]) if m["size_gb"] > 0 else 8000
            pad     = max(1, 38 - len(name))
            vram_lines.append(f'    "{name}":{" " * pad}{vram_mb},')

    if vram_lines:
        block = "\n".join(vram_lines)
        text  = re.sub(
            r'(MODEL_WEIGHT_VRAM_MB\s*=\s*\{.*?)(^\})',
            lambda mo: mo.group(1) + block + "\n" + mo.group(2),
            text, flags=re.DOTALL | re.MULTILINE,
        )

    # Add to YOUR_MODELS
    configured = get_configured_models()
    conf_norm  = {_normalise(cid) for cid in configured}
    your_lines = []
    for m in new_models:
        if _normalise(m["id"]) in conf_norm:
            continue
        parts     = m["id"].split("/", 1)
        publisher = parts[0] if len(parts) == 2 else "unknown"
        name      = parts[1] if len(parts) == 2 else parts[0]
        pad       = max(1, 12 - len(publisher))
        your_lines.append(
            f'    {{"publisher": "{publisher}",{" " * pad}'
            f'"name": "{name}"}},  # {m["size_gb"]:.1f} GB'
        )

    if your_lines:
        block = "\n".join(your_lines)
        text  = re.sub(
            r'(YOUR_MODELS\s*=\s*\[.*?)(^\])',
            lambda mo: mo.group(1) + block + "\n" + mo.group(2),
            text, flags=re.DOTALL | re.MULTILINE,
        )

    CONFIGURE_PY.write_text(text, encoding="utf-8")
    return True


# Role comment markers for inserting into my_models.yaml sections
ROLE_MARKERS = {
    "ceo":           "# ── CEO",
    "cto":           "# ── CTO",
    "cfo":           "# ── CFO",
    "cfo_backup":    "# ── CFO_BACKUP",
    "cpo":           "# ── CPO",
    "coo":           "# ── COO",
    "vision":        "# ── Vision",
    "autocomplete":  "# ── Autocomplete",
    "embed":         "# ── Embed",
    "primary":       "# ── Primary fallback",
    "general":       "# ── General fallback",
    "fast_reasoning":"# ── Fast reasoning",
    "last_resort":   "# ── Last resort",
}


def _find_role_block(lines: list, role: str) -> tuple:
    """Returns (start_idx, end_idx) of the role's comment block."""
    marker = ROLE_MARKERS.get(role, "")
    if not marker:
        return -1, -1
    for i, line in enumerate(lines):
        if marker.lower() in line.lower():
            # Find where this block ends (next role marker or section)
            for j in range(i + 1, len(lines)):
                s = lines[j].strip()
                if s.startswith("# ──") or s.startswith("# ══"):
                    return i, j
            return i, len(lines)
    return -1, -1


def update_my_models_yaml(plans: list) -> bool:
    """
    For auto-assigned models: writes an ACTIVE (uncommented) line.
    For comment-only models: inserts a commented option.
    Respects existing content — never duplicates.
    """
    if not MY_MODELS.exists() or not plans:
        return True

    lines = MY_MODELS.read_text(encoding="utf-8").splitlines()

    for plan in plans:
        mid     = plan["model"]["id"]
        size_gb = plan["model"]["size_gb"]
        role    = plan["role"]
        section = plan["section"]
        score   = plan["score"]
        reason  = plan["reason"]

        # Skip if already in file (any form)
        if any(mid in ln for ln in lines):
            continue

        start, end = _find_role_block(lines, role)
        if start == -1:
            continue

        if plan["should_assign"]:
            # Write as ACTIVE assignment
            note = (f"# auto-assigned: score {score}/100 "
                    f"vs preset ({plan['preset_score']}/100) — {reason}")
            active_line  = f"  {role}: {mid}"
            comment_line = f"  {note}"

            # Check if there's already an active line in this block — if so, comment it out
            for i in range(start, end):
                s = lines[i].strip()
                if (s.startswith(f"{role}:") or
                    (not s.startswith("#") and ":" in s and
                     s.split(":")[0].strip() == role)):
                    lines[i] = "  # " + lines[i].strip()

            insert_at = start + 1
            lines.insert(insert_at,     comment_line)
            lines.insert(insert_at + 1, active_line)
        else:
            # Write as commented option
            if plan["role_overridden"]:
                note = f"# manual override active for {role} — option available"
            elif plan["wins_by"] < 0:
                note = f"# preset default scores higher ({plan['preset_score']} vs {score}) — keeping as option"
            else:
                note = f"# score {score}/100 — {reason}"
            comment_line = f"  # {role}: {mid}  {note}"

            insert_at = end - 1
            while insert_at > start and lines[insert_at].strip() == "":
                insert_at -= 1
            lines.insert(insert_at + 1, comment_line)

    MY_MODELS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


# ══════════════════════════════════════════════════════
# ARCHIVE / REMOVE / RESTORE
# ══════════════════════════════════════════════════════

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def archive_and_remove(removed_ids: list) -> int:
    """
    For each model that's no longer in LM Studio:
      1. Save its learned_settings.json entry to config/model_archive.json
      2. Save its VRAM entry from configure_models.py
      3. Remove it from YOUR_MODELS in configure_models.py
      4. Comment out any active assignment in my_models.yaml
    Returns count of models archived.
    """
    if not removed_ids:
        return 0

    learned  = _load_json(LEARNED_SETTINGS)
    archive  = _load_json(ARCHIVE_FILE)
    vram_tbl = get_vram_table()

    from datetime import date
    today = str(date.today())

    for model_id in removed_ids:
        name = model_id.split("/")[-1]

        # 1. Archive learned settings (keep forever — tiny file)
        entry = {"archived_at": today, "model_id": model_id}
        if model_id in learned:
            entry["settings"] = learned[model_id]
        if name in vram_tbl:
            entry["vram_mb"] = vram_tbl[name]
        # Infer publisher/name for re-adding later
        parts = model_id.split("/", 1)
        entry["publisher"] = parts[0] if len(parts) == 2 else "unknown"
        entry["model_name"] = parts[1] if len(parts) == 2 else parts[0]

        archive[model_id] = entry

    _save_json(ARCHIVE_FILE, archive)

    # 2. Remove from YOUR_MODELS in configure_models.py
    text = CONFIGURE_PY.read_text(encoding="utf-8")
    for model_id in removed_ids:
        name = model_id.split("/")[-1]
        # Match the dict entry line (with or without trailing comment)
        text = re.sub(
            r'^\s*\{"publisher":\s*"[^"]*",\s*"name":\s*"' + re.escape(name) + r'"[^}]*\},?[^\n]*\n',
            '',
            text,
            flags=re.MULTILINE,
        )
    CONFIGURE_PY.write_text(text, encoding="utf-8")

    # 3. Comment out active assignments in my_models.yaml
    if MY_MODELS.exists():
        lines = MY_MODELS.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Active line (not a comment) containing removed model
            if (not stripped.startswith("#")
                    and ":" in stripped
                    and any(mid.split("/")[-1] in stripped
                            or mid in stripped
                            for mid in removed_ids)):
                lines[i] = "  # [removed] " + stripped
        MY_MODELS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return len(removed_ids)


def restore_archived(new_models: list) -> int:
    """
    For each newly discovered model, check if it has archived settings.
    If yes, restore them to learned_settings.json so the model loads
    with its previously tuned context/cache settings immediately.
    Returns count of models restored.
    """
    if not new_models:
        return 0

    archive  = _load_json(ARCHIVE_FILE)
    learned  = _load_json(LEARNED_SETTINGS)
    restored = 0

    for model in new_models:
        mid = model["id"]
        if mid not in archive:
            continue
        saved = archive[mid].get("settings")
        if not saved:
            continue
        if mid not in learned:
            learned[mid] = saved
            restored += 1
            pd(f"  Restored learned settings for {mid} "
               f"(context={saved.get('context')}, "
               f"used {saved.get('success_count', 0)}x before)")

    if restored:
        _save_json(LEARNED_SETTINGS, learned)

    return restored


# ══════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════

def print_plan(plans: list, preset_name: str):
    if not plans:
        return

    p(f"\n  [bold]ASSIGNMENT PLAN[/bold] (preset: [bold cyan]{preset_name.upper()}[/bold cyan])")
    p(f"  {'Model':<36} {'Score':>5}  {'Role':<22} Decision")
    p(f"  {'-' * 36}  {'-' * 5}  {'-' * 22}  {'-' * 30}")

    for plan in plans:
        mid      = plan["model"]["id"]
        score    = plan["score"]
        role     = plan["role"]
        p_score  = plan["preset_score"]
        decision = ""

        if plan["should_assign"]:
            decision = f"[green]AUTO-ASSIGNED[/green] (beats preset by +{plan['wins_by']})"
        elif plan["role_overridden"]:
            decision = f"[yellow]commented only[/yellow] (you already set {role} manually)"
        elif plan["wins_by"] < 0:
            decision = f"[dim]commented only[/dim] (preset model scores {p_score} here)"
        else:
            decision = f"[dim]commented only[/dim] (close scores — review manually)"

        role_label = ROLE_DISPLAY.get(role, role)
        p(f"  {mid:<36} {score:>5}  {role_label:<22}  {decision}")
        pd(f"    Reason: {plan['reason']}")

    p("")


def print_status(discovered, configured, new_models, removed, source, preset):
    p("\n  [bold]" + "=" * 60 + "[/bold]")
    p(f"  [bold cyan]LM STUDIO MODEL SYNC[/bold cyan]")
    p(f"  [bold]" + "=" * 60 + "[/bold]")
    p(f"  Source    : [bold]{source}[/bold]")
    p(f"  Preset    : [bold]{preset.upper()}[/bold]")
    p(f"  Found     : [bold]{len(discovered)}[/bold] models in LM Studio")
    p(f"  Tracked   : [bold]{len(configured)}[/bold] models in configure_models.py")

    if not new_models and not removed:
        pr("  Everything in sync — no new models found.")
        p("  [bold]" + "=" * 60 + "[/bold]\n")
        return

    if new_models:
        p(f"\n  [bold]NEW MODELS[/bold] ({len(new_models)})")
        for m in new_models:
            s = f"{m['size_gb']:.1f} GB" if m["size_gb"] > 0 else "? GB"
            p(f"  [green]+[/green] {m['id']:<38} {s}")

    if removed:
        p(f"\n  [bold yellow]NOT FOUND[/bold yellow] ({len(removed)}) — in config but not in LM Studio")
        for cid in removed:
            py(f"  - {cid}")
        py("  (settings will be archived, entries removed from config — run --apply)")

    p(f"\n  [bold]" + "=" * 60 + "[/bold]\n")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    apply     = "--apply"     in sys.argv or "--full" in sys.argv
    configure = "--configure" in sys.argv or "--full" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        p(__doc__)
        return

    # ── Step 1: Discover ──────────────────────────────
    pd("\nChecking LM Studio API...")
    api_models = discover_from_api()

    if api_models:
        source     = "LM Studio API"
        discovered = api_models
        pd(f"  Connected — {len(discovered)} model(s) returned")
    else:
        py("  API not available. Scanning filesystem...")
        discovered = discover_from_filesystem()
        source     = f"Filesystem scan"
        if not discovered:
            pe("  No models found. Is LM Studio installed?\n")
            return

    # ── Step 2: Diff ──────────────────────────────────
    preset_name         = get_active_preset_name()
    configured          = get_configured_models()
    new_models, removed = diff_models(discovered, configured)

    print_status(discovered, configured, new_models, removed, source, preset_name)

    if not new_models and not removed:
        return

    # ── Step 3: Plan assignments ───────────────────────
    if new_models:
        plans = plan_assignments(new_models, preset_name)
        print_plan(plans, preset_name)

        assigned = [p for p in plans if p["should_assign"]]
        comments = [p for p in plans if not p["should_assign"]]

        p(f"  Summary:")
        if assigned:
            pr(f"  {len(assigned)} model(s) will be AUTO-ASSIGNED to roles")
        if comments:
            pd(f"  {len(comments)} model(s) added as commented options only")
    else:
        plans = []

    if not apply:
        p("\n  Run with [bold]--apply[/bold] to write these changes.")
        p("  Run with [bold]--full[/bold]  to also write LM Studio load settings.\n")
        return

    # ── Step 4: Handle removed models ─────────────────
    if removed:
        pb("Archiving and removing models no longer in LM Studio...")
        count = archive_and_remove(removed)
        pr(f"  {count} model(s) archived to config/model_archive.json")
        pd("  Their learned settings (context, cache, load times) are saved.")
        pd("  Re-download any of them and settings restore automatically.")
        p("")

    # ── Step 5: Apply new models ───────────────────────
    if new_models:
        # Restore archived settings for re-downloaded models first
        restored = restore_archived(new_models)
        if restored:
            pr(f"  {restored} model(s) had archived settings — restored to learned_settings.json")
            p("")
        pb("Updating configure_models.py...")
        ok = update_configure_models(new_models)
        if ok:
            pd(f"  Added {len(new_models)} model(s) to YOUR_MODELS + VRAM table")

        pb("Updating config/my_models.yaml...")
        ok = update_my_models_yaml(plans)
        if ok:
            assigned_count = len([p for p in plans if p["should_assign"]])
            comment_count  = len([p for p in plans if not p["should_assign"]])
            if assigned_count:
                pr(f"  {assigned_count} model(s) auto-assigned and active")
            if comment_count:
                pd(f"  {comment_count} model(s) added as commented options")
            pd("  Review config/my_models.yaml to adjust any assignment")

    # ── Step 5: LM Studio settings ────────────────────
    if configure:
        pb("\nWriting LM Studio load settings...")
        pe("  LM Studio must be CLOSED for this step.")
        result = subprocess.run(
            [sys.executable, str(CONFIGURE_PY)], cwd=ROOT
        )
        if result.returncode != 0:
            pe("  configure_models.py reported errors.")
        else:
            pr("  LM Studio settings written.")

    pr("\nSync complete.\n")


if __name__ == "__main__":
    main()
