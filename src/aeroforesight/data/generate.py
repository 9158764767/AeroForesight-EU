"""Synthetic European flight-network generator.

Stands in for real feeds (OpenSky Network, Eurocontrol R&D data, airline OAG
schedules) so the whole platform is runnable offline with no credentials.

The generator is *causal*: delays depend on hub congestion, weather, distance,
carrier cost profile and time-of-day — so the DL model has real signal to learn
and the RL agent has a real network to optimise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import data_dir, load_config


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (vectorised)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def generate_flights(cfg: dict | None = None) -> pd.DataFrame:
    """Generate a synthetic flight table for the configured European network."""
    cfg = cfg or load_config()
    dcfg = cfg["data"]
    rng = np.random.default_rng(dcfg["seed"])

    airports = pd.DataFrame(cfg["airports"]).set_index("iata")
    carriers = pd.DataFrame(cfg["carriers"])
    iatas = airports.index.to_numpy()

    n_days = int(dcfg["n_days"])
    per_day = int(dcfg["flights_per_day"])
    n = n_days * per_day
    start = pd.Timestamp(dcfg["start_date"])

    # --- pick origin/destination (no self-loops) ---
    origin = rng.choice(iatas, size=n)
    dest = rng.choice(iatas, size=n)
    mask = origin == dest
    while mask.any():
        dest[mask] = rng.choice(iatas, size=mask.sum())
        mask = origin == dest

    carrier = rng.choice(carriers["code"].to_numpy(), size=n)
    carrier_idx = pd.Series(range(len(carriers)), index=carriers["code"])
    cost_index = carriers.set_index("code")["cost_index"].reindex(carrier).to_numpy()

    day = rng.integers(0, n_days, size=n)
    dep_hour = rng.integers(5, 24, size=n)  # 05:00–23:00 departures
    dep_dt = start + pd.to_timedelta(day, unit="D") + pd.to_timedelta(dep_hour, unit="h")
    dow = dep_dt.dayofweek.to_numpy()

    dist = _haversine(
        airports.loc[origin, "lat"].to_numpy(),
        airports.loc[origin, "lon"].to_numpy(),
        airports.loc[dest, "lat"].to_numpy(),
        airports.loc[dest, "lon"].to_numpy(),
    )

    # --- congestion: how loaded is the origin hub in that hour ---
    cap = airports.loc[origin, "capacity"].to_numpy()
    # peak factor: mornings (6-9) and evenings (17-20) are busiest
    peak = np.where(np.isin(dep_hour, [6, 7, 8, 17, 18, 19]), 1.35, 1.0)
    demand_pressure = (per_day / len(iatas)) * peak
    congestion = np.clip(demand_pressure / (cap / 24.0), 0.2, 2.5)

    # --- weather shock: seasonal + random storms ---
    season = 1.0 + 0.4 * np.sin(2 * np.pi * (day / 365.0))  # winter worse
    weather = rng.gamma(shape=2.0, scale=1.0, size=n) * season

    # --- causal arrival-delay (minutes) ---
    base = 4.0
    delay = (
        base
        + 9.0 * (congestion - 1.0)
        + 2.2 * weather
        + 0.004 * dist
        + 6.0 * (cost_index - 0.9)      # tight low-cost turnarounds propagate delay
        + 2.0 * (dow >= 5)              # weekend leisure peaks
        + rng.normal(0, 6.0, size=n)
    )
    delay = np.clip(delay, -10, 240)

    flight_time_min = dist / 12.0 + 35 + rng.normal(0, 6, size=n)  # ~720 km/h + taxi
    pax = np.clip(rng.normal(150, 40, size=n), 40, 320).astype(int)
    # revenue proxy (EUR): distance-based yield, discounted for low-cost carriers
    fare = (40 + 0.09 * dist) * np.where(cost_index < 0.8, 0.75, 1.0)
    revenue = fare * pax

    df = pd.DataFrame(
        {
            "flight_id": np.arange(n),
            "dep_time": dep_dt,
            "day": day,
            "dep_hour": dep_hour,
            "dow": dow,
            "origin": origin,
            "dest": dest,
            "carrier": carrier,
            "cost_index": cost_index.round(3),
            "distance_km": dist.round(1),
            "flight_time_min": flight_time_min.round(1),
            "congestion": congestion.round(3),
            "weather_index": weather.round(3),
            "pax": pax,
            "revenue_eur": revenue.round(2),
            "arr_delay_min": delay.round(1),
        }
    )
    df["delayed"] = (df["arr_delay_min"] > cfg["dl_model"]["delay_threshold_min"]).astype(int)
    return df


def write_dataset(cfg: dict | None = None) -> str:
    """Generate and persist the dataset as parquet (falls back to CSV)."""
    cfg = cfg or load_config()
    df = generate_flights(cfg)
    out = data_dir() / "flights.parquet"
    try:
        df.to_parquet(out, index=False)
    except Exception:  # pyarrow/fastparquet not installed
        out = data_dir() / "flights.csv"
        df.to_csv(out, index=False)
    return str(out)


if __name__ == "__main__":
    path = write_dataset()
    print(f"Wrote synthetic dataset -> {path}")
