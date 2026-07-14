"""LLM foresight-briefing layer.

Turns the platform's structured outputs (network KPIs, delay predictions, RL
buffer plan, 2040 scenario forecast) into a concise executive briefing for a
European aviation strategy team, using the Anthropic Claude API.

If no ``ANTHROPIC_API_KEY`` is configured, a deterministic template briefing is
produced instead — so the platform never hard-depends on network access.
"""

from __future__ import annotations

import json
import os
import textwrap

MODEL = os.environ.get("AEROFORESIGHT_LLM_MODEL", "claude-opus-4-8")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the strategy analyst for AeroForesight-EU, a European aviation
    intelligence platform. You write concise, decision-oriented briefings for
    airline network planners and airport commercial teams.

    Ground every claim in the JSON metrics provided. Use EUR. Be specific about
    hubs, carriers and scenarios. Cover: (1) current network health, (2) delay
    risk and the recommended buffer allocation, (3) the 2040 sustainability-cost
    outlook across scenarios, and (4) two concrete recommendations. Keep it under
    350 words. Do not invent numbers that are not in the data.
    """
).strip()


def _fmt(v) -> str:
    return f"{v:,}" if isinstance(v, (int, float)) else str(v)


def _template_briefing(payload: dict) -> str:
    kpis = payload.get("network_kpis", {})
    fc = payload.get("forecast_2040", {})
    rl = payload.get("rl_plan", {})
    return textwrap.dedent(
        f"""
        AeroForesight-EU — Executive Briefing (offline template)

        1. Network health
        {_fmt(kpis.get('total_flights', 'n/a'))} flights modelled; on-time rate
        {kpis.get('on_time_pct', 'n/a')}%. Hub concentration (HHI)
        {kpis.get('hub_hhi', 'n/a')}. Revenue at risk from delays:
        EUR {_fmt(kpis.get('revenue_at_risk_eur', 'n/a'))}.

        2. Delay risk & buffer allocation
        Recommended RL buffer plan: {rl.get('allocation', 'n/a')}
        (expected net benefit EUR {_fmt(rl.get('expected_net_benefit_eur', 'n/a'))}).

        3. 2040 sustainability-cost outlook
        {json.dumps(fc, indent=2)}

        4. Recommendations
        - Prioritise schedule buffer at the highest-congestion hubs identified above.
        - Hedge ETS exposure and accelerate SAF offtake if the green_push scenario firms up.

        (Set ANTHROPIC_API_KEY for a full narrative briefing.)
        """
    ).strip()


def generate_briefing(payload: dict, model: str = MODEL) -> str:
    """Generate an executive briefing. Falls back to a template offline."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_briefing(payload)

    try:
        import anthropic
    except ImportError:
        return _template_briefing(payload)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / profile
    user_msg = (
        "Write the briefing from these metrics:\n\n"
        + json.dumps(payload, indent=2, default=str)
    )
    # Stream so a long briefing never hits an HTTP timeout; adaptive thinking on.
    with client.messages.stream(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        message = stream.get_final_message()

    return "".join(b.text for b in message.content if b.type == "text").strip()
