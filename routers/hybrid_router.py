# routers/hybrid_router.py
# Routes prompts to the right agent and backend
# Reads config from config.yaml via cfg

import os
import yaml
from core.config_loader import cfg
from routers.local_router import call_local, check_any_local_health
from routers.cloud_router import call_cloud, check_openrouter_health
from rich.console import Console

console = Console()

RULES_PATH = os.path.join(
    os.path.dirname(__file__), "../config/routing_rules.yaml"
)
with open(RULES_PATH, "r") as f:
    ROUTING_RULES = yaml.safe_load(f)

# Built once at import — avoids recreating the dict on every detect_intent() call
_RULE_MAP: dict[str, list[str]] = {
    "full_board": ROUTING_RULES.get("full_board_triggers", []),
    "cto":        ROUTING_RULES.get("cto_only_triggers",   []),
    "cfo":        ROUTING_RULES.get("cfo_only_triggers",   []),
    "cpo":        ROUTING_RULES.get("cpo_only_triggers",   []),
    "coo":        ROUTING_RULES.get("coo_only_triggers",   []),
    "vision":     ROUTING_RULES.get("vision_triggers",     []),
}


def detect_intent(prompt: str) -> str:
    """
    Scans prompt for keywords.
    Returns intent string.

    Returns:
        'full_board' | 'cto' | 'cfo' | 'cpo' |
        'coo' | 'vision' | 'default'
    """
    prompt_lower = prompt.lower()

    for intent, triggers in _RULE_MAP.items():
        for trigger in triggers:
            if trigger in prompt_lower:
                console.print(
                    f"[cyan]  Routing to:[/cyan] "
                    f"[bold]{intent}[/bold] "
                    f"[dim](matched: '{trigger}')[/dim]"
                )
                return intent

    return "default"


def route(
    prompt: str,
    system_message: str,
    compute: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> str:
    if compute == "cloud" and cfg.use_cloud:
        cloud_result = call_cloud(
            prompt, system_message, model, temperature, max_tokens
        )
        if cloud_result is not None:
            return cloud_result

    if check_any_local_health():
        return call_local(
            prompt, system_message, model, temperature, max_tokens
        )

    return (
        "[ERROR] No backends available.\n\n"
        "Fix:\n"
        "  LM Studio → Local Server → Start Server\n"
        "  OR: ollama serve\n\n"
        "Then try again."
    )
