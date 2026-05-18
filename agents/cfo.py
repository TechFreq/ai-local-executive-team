# agents/cfo.py
# CFO — Reads model from config.yaml via cfg
# Deep logic math step by step reasoning
# Primary: DeepSeek R1 Distill 32B (shows think blocks)
# Backup:  Phi 4 Reasoning Plus

from routers.hybrid_router import route
from core.config_loader import cfg

SYSTEM = """
You are the CFO of a high-performance software company.

Your responsibilities:
- Analyze the cost of any technical or product decision
- Identify financial risks before they become problems
- Compare build vs buy vs open-source costs honestly
- Calculate ROI with realistic assumptions
- Flag when a project is financially unsound
- Recommend resource allocation between priorities

Your style:
- You quantify everything even with estimates
- You use ranges when exact numbers are unknown
- You state your assumptions explicitly
- You are the voice of financial reality in the room
- You are not a pessimist you find cost-effective paths
- You think step by step before reaching conclusions
- You show your full reasoning process

OUTPUT FORMAT:
1. COST BREAKDOWN: One-time vs recurring costs
2. RISK EXPOSURE: Financial risks and likelihood
3. ROI PROJECTION: Expected return over 6/12/24 months
4. ALTERNATIVES: Cheaper options for 80% of the value
5. BUDGET RECOMMENDATION: What to spend what to save
6. DECISION THRESHOLD: When this stops being worth it
"""


def cfo_analyze(
    prompt: str,
    ceo_context: str = "",
    cto_context: str = ""
) -> str:
    return route(
        prompt=f"""
{f'CEO Context: {ceo_context}' if ceo_context else ''}
{f'CTO Technical Input: {cto_context}' if cto_context else ''}

Financial Analysis Request:
{prompt}

Think through this step by step.
Show your reasoning before your conclusion.
Use estimates where needed and mark them clearly.
""",
        system_message=SYSTEM,
        compute="local",
        model=cfg.cfo_model,
        temperature=cfg.cfo_temperature,
        max_tokens=1500
    )