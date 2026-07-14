"""End-to-end training & foresight pipeline.

Run: ``python -m aeroforesight.mlops.pipeline``

Steps
-----
1. Generate / load the synthetic European flight network.
2. Engineer features and train the DL delay predictor.
3. Gate on validation AUC and register the model.
4. Train the RL buffer-allocation agent.
5. Run the 2040 sustainability-cost scenario forecast.
6. Compute business-entity KPIs.
7. Produce an executive foresight briefing (LLM or offline template).
8. Persist a consolidated ``artifacts/foresight_report.json`` the dashboard reads.
"""

from __future__ import annotations

import json

import numpy as np

from ..config import artifact_dir, load_config
from ..data.loader import load_flights
from ..features.build import TARGET, build_features
from ..models import business, forecast, llm_insights
from ..models.dl_delay import DelayModel
from ..models.rl_network import HubEnv, recommend_allocation, train_agent
from .monitoring import drift_report, performance_gate
from .registry import ModelRegistry


def run_pipeline(write_report: bool = True) -> dict:
    cfg = load_config()
    print("[1/8] Loading flight network ...")
    df = load_flights(cfg)

    print("[2/8] Building features + training DL delay model ...")
    X, spec = build_features(df)
    y = df[TARGET].to_numpy()
    dl = DelayModel(
        hidden_dim=cfg["dl_model"]["hidden_dim"],
        epochs=cfg["dl_model"]["epochs"],
        batch_size=cfg["dl_model"]["batch_size"],
        lr=cfg["dl_model"]["lr"],
    )
    metrics = dl.train(X.to_numpy(), y)
    print(f"      DL backend={metrics['backend']} val_auc={metrics['val_auc']}")

    print("[3/8] Performance gate + registry ...")
    gate = performance_gate(metrics["val_auc"], cfg)
    reg = ModelRegistry()
    artifact = artifact_dir() / "models" / "delay_model"
    dl.save(artifact)
    entry = reg.register(
        "delay_model",
        str(artifact),
        metrics,
        stage="production" if gate["passed"] else "staging",
    )
    print(f"      gate passed={gate['passed']} registered v{entry['version']} ({entry['stage']})")

    print("[4/8] Training RL buffer-allocation agent ...")
    rcfg = cfg["rl_agent"]
    env = HubEnv(n_hubs=4, seed=cfg["data"]["seed"])
    agent = train_agent(
        env,
        episodes=rcfg["episodes"],
        gamma=rcfg["gamma"],
        alpha=rcfg["alpha"],
        epsilon=rcfg["epsilon"],
    )
    # Use today's top-4 hub congestion bands as the live state.
    top_hubs = df["origin"].value_counts().head(4).index.tolist()
    bands = [
        int(np.clip(round(df[df["origin"] == h]["congestion"].mean()), 0, 2))
        for h in top_hubs
    ]
    rl_plan = recommend_allocation(agent, bands)
    rl_plan["hubs"] = top_hubs
    print(f"      hubs={top_hubs} bands={bands} net_benefit=EUR {rl_plan['expected_net_benefit_eur']:,}")

    print("[5/8] Running 2040 scenario forecast ...")
    fc = forecast.run_forecast(cfg)
    forecast_2040 = (
        fc[fc["year"] == fc["year"].max()]
        .set_index("scenario")[["green_cost_eur_m", "co2_mt", "saf_share", "ets_price_eur"]]
        .to_dict("index")
    )

    print("[6/8] Business-entity analytics ...")
    kpis = business.network_kpis(df)
    routes = business.route_summary(df)
    carriers = business.carrier_market_share(df)

    print("[7/8] Drift monitoring (self-check) ...")
    ref = {c: X[c].to_numpy() for c in ["congestion", "weather_index", "distance_km"]}
    live = {c: v + np.random.default_rng(1).normal(0, 0.05, size=len(v)) for c, v in ref.items()}
    drift = drift_report(ref, live, cfg)

    print("[8/8] Generating foresight briefing ...")
    briefing_payload = {
        "network_kpis": kpis,
        "rl_plan": rl_plan,
        "forecast_2040": forecast_2040,
    }
    briefing = llm_insights.generate_briefing(briefing_payload)

    report = {
        "dl_metrics": metrics,
        "performance_gate": gate,
        "network_kpis": kpis,
        "top_routes": routes.to_dict("records"),
        "carrier_market_share": carriers.to_dict("records"),
        "rl_plan": rl_plan,
        "forecast": fc.to_dict("records"),
        "forecast_2040": forecast_2040,
        "drift": drift,
        "briefing": briefing,
    }

    if write_report:
        out = artifact_dir() / "foresight_report.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote consolidated report -> {out}")

    return report


if __name__ == "__main__":
    run_pipeline()
