"""FastAPI inference & foresight service.

Run: ``uvicorn aeroforesight.serving.api:app --reload``

Endpoints
---------
GET  /health              liveness + which model version is loaded
POST /predict/delay       delay probability for a single flight
GET  /network/kpis        current network KPIs
GET  /forecast            2040 scenario forecast
GET  /rl/recommend        RL buffer allocation for given hub congestion bands
GET  /briefing            latest executive foresight briefing (from last pipeline run)
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import artifact_dir, load_config
from ..data.loader import load_flights
from ..features.build import build_features, align_to_spec
from ..models import business, forecast
from ..models.dl_delay import DelayModel
from ..models.rl_network import HubEnv, recommend_allocation, train_agent
from ..mlops.registry import ModelRegistry

app = FastAPI(
    title="AeroForesight-EU API",
    version="0.1.0",
    description="European aviation intelligence & foresight — DL/RL/LLM/MLOps.",
)

# ---- lazily-initialised state ------------------------------------------------
_STATE: dict = {}


def _bootstrap():
    """Load config, data, feature spec and the registered DL model on first use."""
    if _STATE:
        return _STATE
    cfg = load_config()
    df = load_flights(cfg)
    _, spec = build_features(df)
    reg = ModelRegistry()
    entry = reg.latest("delay_model")
    model = DelayModel.load(entry["artifact_path"]) if entry else None
    _STATE.update(cfg=cfg, df=df, spec=spec, model=model, model_entry=entry)
    return _STATE


# ---- schemas -----------------------------------------------------------------
class FlightRequest(BaseModel):
    origin: str = Field(..., examples=["LHR"])
    dest: str = Field(..., examples=["FRA"])
    carrier: str = Field(..., examples=["LH"])
    distance_km: float = 650
    flight_time_min: float = 90
    congestion: float = 1.2
    weather_index: float = 2.0
    cost_index: float = 1.0
    dep_hour: int = 8
    dow: int = 2
    pax: int = 160


class HubBands(BaseModel):
    bands: list[int] = Field(..., examples=[[2, 1, 2, 0]], description="Congestion band 0/1/2 per hub")


# ---- endpoints ---------------------------------------------------------------
@app.get("/health")
def health():
    s = _bootstrap()
    entry = s.get("model_entry")
    return {
        "status": "ok",
        "model_loaded": s.get("model") is not None,
        "model_version": entry["version"] if entry else None,
        "model_stage": entry["stage"] if entry else None,
    }


@app.post("/predict/delay")
def predict_delay(req: FlightRequest):
    s = _bootstrap()
    if s["model"] is None:
        raise HTTPException(503, "No trained model registered. Run the pipeline first.")
    row = pd.DataFrame([req.model_dump()])
    X = align_to_spec(row, s["spec"])
    prob = float(s["model"].predict_proba(X)[0])
    return {
        "route": f"{req.origin}-{req.dest}",
        "carrier": req.carrier,
        "delay_probability": round(prob, 4),
        "risk": "high" if prob > 0.5 else ("medium" if prob > 0.25 else "low"),
    }


@app.get("/network/kpis")
def network_kpis():
    s = _bootstrap()
    return business.network_kpis(s["df"])


@app.get("/forecast")
def get_forecast():
    s = _bootstrap()
    fc = forecast.run_forecast(s["cfg"])
    return fc.to_dict("records")


@app.get("/rl/recommend")
def rl_recommend(bands: str = "2,1,2,0"):
    s = _bootstrap()
    try:
        band_list = [int(b) for b in bands.split(",")]
    except ValueError:
        raise HTTPException(400, "bands must be comma-separated integers, e.g. 2,1,2,0")
    env = HubEnv(n_hubs=len(band_list), seed=s["cfg"]["data"]["seed"])
    agent = train_agent(env, episodes=s["cfg"]["rl_agent"]["episodes"])
    return recommend_allocation(agent, band_list)


@app.get("/briefing")
def briefing():
    report_path = artifact_dir() / "foresight_report.json"
    if not report_path.exists():
        raise HTTPException(404, "No report yet. Run the pipeline to generate a briefing.")
    report = json.loads(report_path.read_text())
    return {"briefing": report.get("briefing", ""), "forecast_2040": report.get("forecast_2040", {})}
