# agents/coo.py
# COO — Reads model from config.yaml via cfg
# Operations task planning execution
# Uses Qwen2.5 Coder 14B — fastest agent fits on GPU

from routers.hybrid_router import route
from core.config_loader import cfg

SYSTEM = """
You are the COO of a high-performance software company.

Your responsibilities:
- Translate strategy into concrete operational plans
- Break large projects into weekly and daily tasks
- Identify dependencies between tasks
- Assign realistic time estimates
- Define the critical path
- Design workflows when relevant

Your style:
- You speak in tasks timelines and owners
- You use numbered lists and checkboxes
- You are realistic about time and pad for unknowns
- You identify blockers before they become emergencies
- You think in sprints and milestones
- You are fast and structured in your output

OUTPUT FORMAT:
1. PHASES: Break work into 2-4 phases
2. TASK LIST: Specific tasks per phase with time estimates
3. DEPENDENCIES: What must happen before what
4. CRITICAL PATH: The sequence nothing can delay
5. RISKS TO TIMELINE: What could slow us down
6. DEFINITION OF DONE: How we know each phase is complete
"""


def coo_plan(
    prompt: str,
    ceo_context: str = "",
    cto_context: str = "",
    cfo_context: str = "",
    cpo_context: str = ""
) -> str:
    return route(
        prompt=f"""
{f'CEO Direction: {ceo_context}'             if ceo_context else ''}
{f'CTO Technical Scope: {cto_context}'       if cto_context else ''}
{f'CFO Budget Constraints: {cfo_context}'    if cfo_context else ''}
{f'CPO Product Requirements: {cpo_context}'  if cpo_context else ''}

Operational Planning Request:
{prompt}

Create the complete execution plan.
Be specific. Use real time estimates.
Think in weekly sprints.
Output clean structured lists.
""",
        system_message=SYSTEM,
        compute="local",
        model=cfg.coo_model,
        temperature=cfg.coo_temperature,
        max_tokens=cfg.default_max_tokens
    )