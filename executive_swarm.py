# executive_swarm.py
# ══════════════════════════════════════════════════════
# Main orchestrator — runs the full executive meeting
# Reads all config from config.yaml via cfg
# ══════════════════════════════════════════════════════

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents.ceo    import ceo_assess, ceo_synthesize
from agents.cto    import cto_analyze, cto_write_code
from agents.cfo    import cfo_analyze
from agents.cpo    import cpo_analyze
from agents.coo    import coo_plan
from agents.vision import vision_analyze_file, vision_analyze_url
from routers.hybrid_router import detect_intent
from core.config_loader import cfg
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()
console = Console()


def run_executive_meeting(prompt: str) -> dict:
    """
    Full board meeting. All 5 agents weigh in.
    CEO goes first and last.
    Returns dict with all responses.
    """
    console.print(Panel(
        f"[bold white]{prompt}[/bold white]",
        title=(
            f"[bold cyan]EXECUTIVE MEETING — "
            f"{cfg.preset_name.upper()}[/bold cyan]"
        ),
        border_style="cyan"
    ))

    responses = {}

    # Phase 1 — CEO reads the room
    console.print(
        "\n[bold yellow]Phase 1 — CEO Initial Read[/bold yellow]"
    )
    responses["ceo_initial"] = ceo_assess(prompt)
    console.print(Panel(
        responses["ceo_initial"],
        title="[bold]CEO — Initial Assessment[/bold]",
        border_style="yellow"
    ))

    # Phase 2 — Department heads run in parallel (CTO, CFO, CPO, COO)
    # COO still needs cto/cfo/cpo context, so it runs after the others.
    console.print(
        "\n[bold yellow]Phase 2 — Department Briefings (parallel)[/bold yellow]"
    )

    def _run_cto():
        console.print(f"\n[cyan]CTO analyzing [{cfg.cto_model}]...[/cyan]")
        return "cto", cto_analyze(prompt, ceo_context=responses["ceo_initial"])

    def _run_cfo():
        console.print(f"\n[cyan]CFO analyzing [{cfg.cfo_model}]...[/cyan]")
        return "cfo", cfo_analyze(
            prompt,
            ceo_context=responses["ceo_initial"],
            cto_context="",          # CTO result not yet available in parallel phase
        )

    def _run_cpo():
        console.print(f"\n[cyan]CPO analyzing [{cfg.cpo_model}]...[/cyan]")
        return "cpo", cpo_analyze(prompt, ceo_context=responses["ceo_initial"])

    # CTO, CFO, CPO run concurrently
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fn) for fn in (_run_cto, _run_cfo, _run_cpo)]
        for future in as_completed(futures):
            key, result = future.result()
            responses[key] = result

    console.print(Panel(
        responses["cto"],
        title="[bold]CTO — Technical[/bold]",
        border_style="blue"
    ))
    console.print(Panel(
        responses["cfo"],
        title="[bold]CFO — Financial & Risk[/bold]",
        border_style="green"
    ))
    console.print(Panel(
        responses["cpo"],
        title="[bold]CPO — Product[/bold]",
        border_style="magenta"
    ))

    # COO runs last — it benefits from the other departments' context
    console.print(f"\n[cyan]COO planning [{cfg.coo_model}]...[/cyan]")
    responses["coo"] = coo_plan(
        prompt,
        ceo_context=responses["ceo_initial"],
        cto_context=responses["cto"],
        cfo_context=responses["cfo"],
        cpo_context=responses["cpo"]
    )
    console.print(Panel(
        responses["coo"],
        title="[bold]COO — Operations[/bold]",
        border_style="red"
    ))

    # Phase 3 — CEO final call
    console.print(
        "\n[bold yellow]Phase 3 — CEO Final Decision[/bold yellow]"
    )
    responses["ceo_final"] = ceo_synthesize(
        original_request=prompt,
        cto_response=responses["cto"],
        cfo_response=responses["cfo"],
        cpo_response=responses["cpo"],
        coo_response=responses["coo"]
    )
    console.print(Panel(
        responses["ceo_final"],
        title="[bold bright_yellow]CEO — FINAL DECISION[/bold bright_yellow]",
        border_style="bright_yellow"
    ))

    return responses


def run_single_agent(prompt: str) -> str:
    """Single agent for quick questions"""
    intent = detect_intent(prompt)

    if intent == "cto" or intent == "default":
        return cto_analyze(prompt)
    elif intent == "cfo":
        return cfo_analyze(prompt)
    elif intent == "cpo":
        return cpo_analyze(prompt)
    elif intent == "coo":
        return coo_plan(prompt)
    elif intent == "vision":
        return (
            "To analyze an image file run:\n"
            "python agents/vision.py <path_to_image> [question]\n\n"
            "Example:\n"
            "python agents/vision.py screenshot.png "
            "'What errors do you see?'"
        )
    elif intent == "full_board":
        return run_executive_meeting(prompt)["ceo_final"]
    else:
        return cto_analyze(prompt)


def smart_route(prompt: str) -> str:
    """
    Auto-decides full board vs single agent.
    This is what the bridge server calls.
    """
    intent = detect_intent(prompt)

    if intent == "full_board":
        return run_executive_meeting(prompt)["ceo_final"]
    else:
        return run_single_agent(prompt)


if __name__ == "__main__":
    cfg.summary()

    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        user_prompt = input(
            "\nWhat should the executive team discuss?\n> "
        )

    smart_route(user_prompt)
