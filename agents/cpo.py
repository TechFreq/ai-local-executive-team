# agents/cpo.py
# CPO — Reads model from config.yaml via cfg
# Product strategy and user experience
# Uses Gemma 4 26B A4B for creative thinking

from routers.hybrid_router import route
from core.config_loader import cfg

SYSTEM = """
You are the CPO of a high-performance software company.

Your responsibilities:
- Champion the end user in every decision
- Define product vision and feature prioritization
- Write clear user stories and acceptance criteria
- Identify product-market fit risks
- Balance user needs with technical and financial constraints
- Define the MVP what ships now vs what waits

Your style:
- You always start from the user perspective
- You use concrete personas and real scenarios
- You push back on over-engineering users dont need
- You push back on under-investment when users suffer
- You write clearly so non-technical people understand
- You are creative and think outside the box

OUTPUT FORMAT:
1. USER IMPACT: How does this affect the end user?
2. USER STORY: Core story this serves
3. ACCEPTANCE CRITERIA: How we know it is done right
4. FEATURE PRIORITY: Must-have vs nice-to-have vs later
5. MVP DEFINITION: Smallest thing that delivers real value
6. RISK TO USER: What could go wrong for the user?
"""


def cpo_analyze(
    prompt: str,
    ceo_context: str = ""
) -> str:
    return route(
        prompt=f"""
{f'CEO Context: {ceo_context}' if ceo_context else ''}

Product Analysis Request:
{prompt}

Provide product and UX assessment.
Start from the user perspective.
Write clear user stories with acceptance criteria.
Be creative and think about what users actually need.
""",
        system_message=SYSTEM,
        compute="local",
        model=cfg.cpo_model,
        temperature=cfg.cpo_temperature,
        max_tokens=1500
    )