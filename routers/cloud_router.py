# routers/cloud_router.py
# Cloud is OFF by default
# Returns None so hybrid_router falls back to local
# Reads config from config.yaml via cfg

import requests
from core.config_loader import cfg
from rich.console import Console

console = Console()

OPENROUTER_KEY = cfg._raw.get("cloud", {}).get("openrouter_api_key", "")
OPENROUTER_URL = cfg._raw.get(
    "cloud", {}
).get("openrouter_base_url", "https://openrouter.ai/api/v1")


def call_cloud(
    prompt: str,
    system_message: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    if not cfg.use_cloud:
        console.print("[dim]  Cloud disabled. Using local.[/dim]")
        return None

    if not OPENROUTER_KEY:
        console.print(
            "[yellow]  No API key. Using local fallback.[/yellow]"
        )
        return None

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "http://localhost:5555",
            "X-Title":       "AI Executive Team"
        }
        payload = {
            "model":       model,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": prompt}
            ]
        }
        response = requests.post(
            f"{OPENROUTER_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        console.print(
            f"[yellow]  Cloud failed: {str(e)}. "
            f"Using local.[/yellow]"
        )
        return None


def check_openrouter_health() -> bool:
    if not cfg.use_cloud or not OPENROUTER_KEY:
        return False
    try:
        r = requests.get(
            f"{OPENROUTER_URL}/models",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            timeout=3
        )
        return r.status_code == 200
    except Exception:
        return False
