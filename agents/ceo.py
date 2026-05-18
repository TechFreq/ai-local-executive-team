# agents/ceo.py
# CEO — Reads model from config.yaml via cfg
# Strategic synthesizer. First look last word.

from routers.hybrid_router import route
from core.config_loader import cfg

SYSTEM = """
You are the CEO of a high-performance software company.

Your responsibilities:
- Receive the raw request and identify the core strategic question
- Determine which departments need to weigh in
- Synthesize all department responses into ONE clear final decision
- Acknowledge dissenting opinions honestly
- Define the top 3 immediate next steps
- Define success metrics for any decision made

Your communication style:
- Direct and decisive you make calls you do not hedge
- You acknowledge tradeoffs honestly
- You give credit to department heads who surface good concerns

FINAL OUTPUT FORMAT:
1. DECISION: (one clear sentence)
2. RATIONALE: (why this over alternatives)
3. KEY RISKS: (top 2-3 risks acknowledged)
4. DISSENT NOTED: (any valid opposing view)
5. NEXT STEPS: (3 concrete actions)
6. SUCCESS METRICS: (how we know this worked)
"""


def ceo_assess(prompt: str) -> str:
    return route(
        prompt=f"""
New request:
{prompt}

As CEO provide a brief initial assessment (150 words max):
1. What type of problem is this?
2. Which department heads need to weigh in and why?
3. What is the key question we need to answer?
4. What constraints should all departments keep in mind?
""",
        system_message=SYSTEM,
        compute="local",
        model=cfg.ceo_model,
        temperature=cfg.ceo_temperature,
        max_tokens=400
    )


def ceo_synthesize(
    original_request: str,
    cto_response: str,
    cfo_response: str,
    cpo_response: str,
    coo_response: str
) -> str:
    return route(
        prompt=f"""
EXECUTIVE MEETING — FINAL SYNTHESIS

Original Request:
{original_request}

CTO (Technical):
{cto_response}

CFO (Cost & Risk):
{cfo_response}

CPO (Product & UX):
{cpo_response}

COO (Operations):
{coo_response}

Deliver your final decision using the structured format.
""",
        system_message=SYSTEM,
        compute="local",
        model=cfg.ceo_model,
        temperature=cfg.ceo_temperature,
        max_tokens=cfg.default_max_tokens
    )