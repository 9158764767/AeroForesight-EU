"""AeroForesight-EU — European aviation intelligence & foresight platform.

Sub-packages
------------
data      : synthetic flight-network generation + loaders (OpenSky/Eurocontrol stand-in)
features  : feature engineering for the DL / RL / forecasting models
models    : DL delay predictor, RL network optimiser, scenario forecaster, LLM briefings
mlops     : training pipeline, model registry, drift monitoring
serving   : FastAPI inference service
"""

from __future__ import annotations

__version__ = "0.1.0"
