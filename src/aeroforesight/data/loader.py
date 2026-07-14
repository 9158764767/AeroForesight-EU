"""Load the flight dataset, generating it on first use."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import data_dir, load_config
from .generate import write_dataset


def load_flights(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    pq = data_dir() / "flights.parquet"
    csv = data_dir() / "flights.csv"
    if not pq.exists() and not csv.exists():
        write_dataset(cfg)
    path = pq if pq.exists() else csv
    if Path(path).suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["dep_time"])
    return df
