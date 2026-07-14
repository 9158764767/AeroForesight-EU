"""Business-entity & flying-pattern analytics.

Turns the raw flight table into the metrics a commercial / network team cares
about: route-level demand, carrier market share, hub concentration (HHI),
network connectivity, and the revenue at risk from delays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def route_summary(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    g = (
        df.assign(route=df["origin"] + "-" + df["dest"])
        .groupby("route")
        .agg(
            flights=("flight_id", "count"),
            pax=("pax", "sum"),
            revenue_eur=("revenue_eur", "sum"),
            avg_delay_min=("arr_delay_min", "mean"),
            delay_rate=("delayed", "mean"),
        )
        .sort_values("revenue_eur", ascending=False)
        .head(top)
        .reset_index()
    )
    g["avg_delay_min"] = g["avg_delay_min"].round(1)
    g["delay_rate"] = (g["delay_rate"] * 100).round(1)
    return g


def carrier_market_share(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("carrier").agg(
        flights=("flight_id", "count"),
        revenue_eur=("revenue_eur", "sum"),
        avg_delay_min=("arr_delay_min", "mean"),
    )
    g["share_pct"] = (g["revenue_eur"] / g["revenue_eur"].sum() * 100).round(2)
    g["avg_delay_min"] = g["avg_delay_min"].round(1)
    return g.sort_values("revenue_eur", ascending=False).reset_index()


def hub_concentration_hhi(df: pd.DataFrame) -> float:
    """Herfindahl-Hirschman Index over origin-hub flight shares (0..10000)."""
    shares = df["origin"].value_counts(normalize=True).to_numpy()
    return float(round((shares**2).sum() * 10000, 1))


def connectivity(df: pd.DataFrame) -> pd.DataFrame:
    """Number of distinct destinations reachable from each hub (network breadth)."""
    c = df.groupby("origin")["dest"].nunique().sort_values(ascending=False)
    return c.rename("destinations_served").reset_index().rename(columns={"origin": "hub"})


def revenue_at_risk(df: pd.DataFrame, delay_penalty_per_min: float = 12.0) -> dict:
    """Estimated EUR impact of delays (EU261-style compensation + goodwill proxy)."""
    delayed = df[df["delayed"] == 1]
    exposure = float((delayed["arr_delay_min"] * delay_penalty_per_min).sum())
    return {
        "delayed_flights": int(len(delayed)),
        "delay_rate_pct": round(len(delayed) / max(len(df), 1) * 100, 2),
        "revenue_at_risk_eur": round(exposure, 2),
        "avg_delay_min_delayed": round(float(delayed["arr_delay_min"].mean() or 0), 1),
    }


def network_kpis(df: pd.DataFrame) -> dict:
    return {
        "total_flights": int(len(df)),
        "total_revenue_eur": round(float(df["revenue_eur"].sum()), 2),
        "hub_hhi": hub_concentration_hhi(df),
        "on_time_pct": round(float((1 - df["delayed"].mean()) * 100), 2),
        **revenue_at_risk(df),
    }
