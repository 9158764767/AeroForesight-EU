"""Turn the raw flight table into model-ready feature matrices.

Shared by the DL delay predictor and the monitoring layer so training and
serving see identical feature definitions (no train/serve skew).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "distance_km",
    "flight_time_min",
    "congestion",
    "weather_index",
    "cost_index",
    "dep_hour",
    "dow",
    "pax",
]
CATEGORICAL_FEATURES = ["origin", "dest", "carrier"]
TARGET = "delayed"


@dataclass
class FeatureSpec:
    """Column layout produced by :func:`build_features`, needed at serve time."""

    numeric: list[str]
    categorical_columns: list[str]  # one-hot expanded column names

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical_columns


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    """Return (X, spec). One-hot encodes categoricals, keeps numerics as-is."""
    num = df[NUMERIC_FEATURES].astype(float).reset_index(drop=True)
    cat = pd.get_dummies(df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)
    X = pd.concat([num, cat], axis=1)
    spec = FeatureSpec(numeric=list(num.columns), categorical_columns=list(cat.columns))
    return X, spec


def align_to_spec(df: pd.DataFrame, spec: FeatureSpec) -> np.ndarray:
    """Encode a (possibly single-row) frame to the exact training column order."""
    num = df.reindex(columns=NUMERIC_FEATURES).astype(float)
    cat = pd.get_dummies(df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES)
    X = pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    X = X.reindex(columns=spec.columns, fill_value=0.0)
    return X.to_numpy(dtype="float32")
