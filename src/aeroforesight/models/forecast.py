"""Futuristic scenario forecasting (to 2040) — the platform's headline output.

Projects European traffic, SAF (Sustainable Aviation Fuel) adoption under
ReFuelEU mandates, EU-ETS carbon prices, and the resulting cost impact on the
modelled carriers. Deterministic scenario paths (baseline / green_push /
high_growth) make the futures explainable to business stakeholders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _interp(base_year: int, horizon: int, start: float, end: float) -> np.ndarray:
    years = np.arange(base_year, base_year + horizon + 1)
    return np.interp(years, [years[0], years[-1]], [start, end])


def project_scenario(name: str, params: dict, fcfg: dict, base_traffic: float) -> pd.DataFrame:
    base_year = int(fcfg["base_year"])
    horizon = int(fcfg["horizon_years"])
    years = np.arange(base_year, base_year + horizon + 1)

    cagr = params["traffic_cagr"]
    traffic = base_traffic * (1 + cagr) ** (years - base_year)

    saf_share = _interp(base_year, horizon, 0.06, params["saf_share_2040"])
    ets_price = _interp(base_year, horizon, 95.0, params["ets_price_eur_2040"])

    # Fuel burn (Mt) scales with traffic; ~0.28 kg CO2 per RPK-equivalent unit.
    co2_mt = traffic * 0.28 * (1 - 0.55 * saf_share)  # SAF cuts lifecycle CO2
    # Carbon bill: covered emissions * ETS price (EUR).
    carbon_bill_eur_m = co2_mt * ets_price
    # SAF premium: SAF costs ~2.5x kerosene; premium grows with adoption.
    saf_premium_eur_m = traffic * 0.20 * saf_share * 2.5 * 100
    total_green_cost = carbon_bill_eur_m + saf_premium_eur_m

    return pd.DataFrame(
        {
            "scenario": name,
            "year": years,
            "traffic_index": traffic.round(2),
            "saf_share": saf_share.round(3),
            "ets_price_eur": ets_price.round(1),
            "co2_mt": co2_mt.round(2),
            "carbon_bill_eur_m": carbon_bill_eur_m.round(1),
            "saf_premium_eur_m": saf_premium_eur_m.round(1),
            "green_cost_eur_m": total_green_cost.round(1),
        }
    )


def run_forecast(cfg: dict, base_traffic: float = 100.0) -> pd.DataFrame:
    fcfg = cfg["forecast"]
    frames = [
        project_scenario(name, params, fcfg, base_traffic)
        for name, params in fcfg["scenarios"].items()
    ]
    return pd.concat(frames, ignore_index=True)
