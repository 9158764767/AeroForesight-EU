"""AeroForesight-EU dashboard (Streamlit + Plotly).

Run: ``streamlit run dashboard/streamlit_app.py``

Reads ``artifacts/foresight_report.json`` if present (run the pipeline first);
otherwise computes a live view directly from the data so it always renders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Make the src/ package importable when run via `streamlit run`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeroforesight.config import artifact_dir, load_config  # noqa: E402
from aeroforesight.data.loader import load_flights  # noqa: E402
from aeroforesight.models import business, forecast  # noqa: E402

st.set_page_config(page_title="AeroForesight-EU", layout="wide", page_icon="✈️")

st.title("✈️ AeroForesight-EU")
st.caption("European aviation intelligence & foresight — flight paths, business entities, and 2040 predictions")

cfg = load_config()
df = load_flights(cfg)

report_path = artifact_dir() / "foresight_report.json"
report = json.loads(report_path.read_text()) if report_path.exists() else None

# ---- KPI row -----------------------------------------------------------------
kpis = business.network_kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flights modelled", f"{kpis['total_flights']:,}")
c2.metric("On-time rate", f"{kpis['on_time_pct']}%")
c3.metric("Hub concentration (HHI)", f"{kpis['hub_hhi']:,}")
c4.metric("Revenue at risk (EUR)", f"{kpis['revenue_at_risk_eur']:,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Network & carriers", "Delay patterns", "2040 foresight", "Briefing"]
)

with tab1:
    left, right = st.columns(2)
    routes = business.route_summary(df)
    fig = px.bar(routes, x="route", y="revenue_eur", color="delay_rate",
                 color_continuous_scale="OrRd", title="Top routes by revenue (colour = delay rate %)")
    left.plotly_chart(fig, use_container_width=True)

    share = business.carrier_market_share(df)
    fig2 = px.pie(share, names="carrier", values="revenue_eur", title="Carrier revenue share")
    right.plotly_chart(fig2, use_container_width=True)

with tab2:
    left, right = st.columns(2)
    by_hour = df.groupby("dep_hour")["arr_delay_min"].mean().reset_index()
    fig3 = px.line(by_hour, x="dep_hour", y="arr_delay_min", markers=True,
                   title="Average arrival delay by departure hour")
    left.plotly_chart(fig3, use_container_width=True)

    hub_delay = df.groupby("origin").agg(
        avg_delay=("arr_delay_min", "mean"), congestion=("congestion", "mean")
    ).reset_index()
    fig4 = px.scatter(hub_delay, x="congestion", y="avg_delay", text="origin", size="avg_delay",
                      title="Hub congestion vs average delay")
    right.plotly_chart(fig4, use_container_width=True)

with tab3:
    fc = forecast.run_forecast(cfg)
    fig5 = px.line(fc, x="year", y="green_cost_eur_m", color="scenario", markers=True,
                   title="Sustainability cost to 2040 (carbon bill + SAF premium, EUR m)")
    st.plotly_chart(fig5, use_container_width=True)
    colA, colB = st.columns(2)
    fig6 = px.line(fc, x="year", y="co2_mt", color="scenario", markers=True, title="CO₂ (Mt)")
    colA.plotly_chart(fig6, use_container_width=True)
    fig7 = px.line(fc, x="year", y="saf_share", color="scenario", markers=True, title="SAF share")
    colB.plotly_chart(fig7, use_container_width=True)

with tab4:
    if report and report.get("briefing"):
        if report.get("rl_plan"):
            st.subheader("RL buffer-allocation plan")
            st.json(report["rl_plan"])
        st.subheader("Executive foresight briefing")
        st.markdown(report["briefing"])
    else:
        st.info("Run `python -m aeroforesight.mlops.pipeline` to generate the RL plan and LLM briefing.")
