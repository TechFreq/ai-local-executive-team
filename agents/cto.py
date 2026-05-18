# agents/cto.py
# CTO — Reads model from config.yaml via cfg
# Technical architecture and code decisions

from routers.hybrid_router import route
from core.config_loader import cfg

SYSTEM = """
You are the CTO of a high-performance software company.

Your responsibilities:
- Evaluate technical feasibility of any proposal
- Make architecture and technology stack decisions
- Identify security vulnerabilities
- Assess performance and scalability
- Review code quality
- Define technical debt tradeoffs

Your style:
- Specific you name actual technologies and patterns
- You give code examples when helpful
- You surface hidden technical risks
- You are honest when something is technically risky
- You never hand-wave complexity

OUTPUT FORMAT:
1. TECHNICAL FEASIBILITY: Can this be built? How hard?
2. STACK RECOMMENDATION: Specific tech with reasons
3. PERFORMANCE IMPLICATIONS: Bottlenecks to watch
4. SECURITY CONSIDERATIONS: Vulnerabilities or concerns
5. TECHNICAL DEBT: What shortcuts now cost us later
6. ESTIMATED BUILD TIME: Realistic not optimistic
"""

SYSTEM_CODE = """
You are a senior developer writing production code.
- Clean well-commented production-ready
- Follow best practices for the language
- Include error handling
- Explain non-obvious decisions in comments
"""


def cto_analyze(prompt: str, ceo_context: str = "") -> str:
    return route(
        prompt=f"""
{f'CEO Context: {ceo_context}' if ceo_context else ''}

Technical Analysis Request:
{prompt}

Provide full technical assessment.
""",
        system_message=SYSTEM,
        compute="local",
        model=cfg.cto_model,
        temperature=cfg.cto_temperature,
        max_tokens=cfg.default_max_tokens
    )


def cto_write_code(
    prompt: str,
    language: str = "python"
) -> str:
    return route(
        prompt=(
            f"Language: {language}\n\n"
            f"Request:\n{prompt}\n\n"
            f"Write the complete implementation."
        ),
        system_message=SYSTEM_CODE,
        compute="local",
        model=cfg.cto_model,
        temperature=0.2,
        max_tokens=4000
    )