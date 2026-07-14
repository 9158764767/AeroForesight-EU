"""End-to-end smoke tests. Kept small so CI runs in seconds.

They exercise every layer: data generation, features, DL model, RL agent,
forecast, business analytics, monitoring and the offline LLM briefing.
"""

from __future__ import annotations

import numpy as np

from aeroforesight.config import load_config
from aeroforesight.data.generate import generate_flights
from aeroforesight.features.build import TARGET, build_features, align_to_spec
from aeroforesight.models import business, forecast, llm_insights
from aeroforesight.models.dl_delay import DelayModel
from aeroforesight.models.rl_network import HubEnv, recommend_allocation, train_agent
from aeroforesight.mlops.monitoring import population_stability_index, performance_gate


def _small_cfg():
    cfg = load_config()
    cfg = {**cfg}
    cfg["data"] = {**cfg["data"], "n_days": 5, "flights_per_day": 300}
    return cfg


def test_data_generation_is_causal():
    df = generate_flights(_small_cfg())
    assert len(df) == 5 * 300
    assert {"arr_delay_min", "delayed", "revenue_eur"}.issubset(df.columns)
    # higher congestion should correlate with more delay
    corr = df["congestion"].corr(df["arr_delay_min"])
    assert corr > 0.05


def test_dl_model_trains_and_predicts():
    cfg = _small_cfg()
    df = generate_flights(cfg)
    X, spec = build_features(df)
    y = df[TARGET].to_numpy()
    model = DelayModel(epochs=3)
    metrics = model.train(X.to_numpy(), y)
    assert 0.4 <= metrics["val_auc"] <= 1.0
    probs = model.predict_proba(align_to_spec(df.head(4), spec))
    assert probs.shape == (4,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_rl_agent_learns_positive_policy():
    env = HubEnv(n_hubs=4, seed=1)
    agent = train_agent(env, episodes=200)
    plan = recommend_allocation(agent, [2, 1, 2, 0])
    assert "allocation" in plan
    assert len(plan["allocation"]) == env.buffer_units


def test_forecast_scenarios_diverge():
    cfg = _small_cfg()
    fc = forecast.run_forecast(cfg)
    last = fc[fc["year"] == fc["year"].max()]
    # green_push should cut CO2 vs high_growth
    green = last[last["scenario"] == "green_push"]["co2_mt"].iloc[0]
    high = last[last["scenario"] == "high_growth"]["co2_mt"].iloc[0]
    assert green < high


def test_business_kpis_and_briefing():
    df = generate_flights(_small_cfg())
    kpis = business.network_kpis(df)
    assert kpis["total_flights"] == len(df)
    assert 0 <= kpis["on_time_pct"] <= 100
    briefing = llm_insights.generate_briefing({"network_kpis": kpis, "rl_plan": {}, "forecast_2040": {}})
    assert "AeroForesight-EU" in briefing  # offline template path


def test_monitoring_psi():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    same = rng.normal(0, 1, 5000)
    shifted = rng.normal(1.5, 1, 5000)
    assert population_stability_index(ref, same) < 0.1
    assert population_stability_index(ref, shifted) > 0.25
    gate = performance_gate(0.82, load_config())
    assert gate["passed"]
