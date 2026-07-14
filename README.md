# ✈️ AeroForesight-EU

**European aviation intelligence & foresight platform** — predicting flight-delay
risk, optimising hub networks, mapping business dynamics, and forecasting the
2040 sustainability-cost future of European aviation.

Built as an end-to-end, deployable MLOps project that combines **Deep Learning**,
**Reinforcement Learning**, an **LLM foresight layer**, and a full **MLOps**
pipeline (feature store → training → gating → registry → drift monitoring →
serving → dashboard).

---

## The real-world problem

European carriers, airports and regulators need to anticipate:

| Question | Who cares | AeroForesight answers with |
|---|---|---|
| Which flights will arrive late, and why? | Airline OCC, airports | **DL** delay predictor |
| Where should we invest limited schedule buffer? | Network planning | **RL** buffer-allocation agent |
| How concentrated & connected is the network? | Regulators, strategy | **Business analytics** (HHI, connectivity, revenue-at-risk) |
| What does 2040 cost under EU-ETS + ReFuelEU SAF mandates? | CFO, sustainability | **Scenario forecaster** (baseline / green_push / high_growth) |
| What should leadership actually do? | Executives | **LLM** foresight briefing |

The demo runs on a **synthetic European network** (12 hubs — LHR, CDG, FRA, AMS,
MAD, …; 6 carriers) generated with realistic causal structure (congestion,
weather, distance, low-cost turnaround pressure), so it works fully offline.

---

## Architecture

```
                         ┌────────────────────────────────────────────┐
 Synthetic feeds  ─────▶ │ data.generate → features.build             │
 (OpenSky / Eurocontrol  └───────────────┬────────────────────────────┘
  stand-in)                              │
                     ┌───────────────────┼─────────────────────────────┐
                     ▼                   ▼                 ▼            ▼
             DL delay model      RL network agent   Scenario forecast  Business
             (PyTorch/​sklearn)   (Q-learning)       (2040, ETS+SAF)    analytics
                     │                   │                 │            │
                     └──────────┬────────┴────────┬────────┴────────────┘
                                ▼                 ▼
                     MLOps: gate → registry     LLM foresight briefing
                          → drift monitor         (Claude, opt-in)
                                │                        │
                                ▼                        ▼
                     FastAPI service  ◀──────▶  Streamlit + Plotly dashboard
```

---

## Quick start

```bash
# 1. install (core + optional extras)
pip install -e ".[dl,llm,dashboard,dev]"     # or: make install

# 2. run the full pipeline (generates data, trains DL+RL, forecasts, briefs)
python -m aeroforesight.mlops.pipeline        # or: make pipeline

# 3. serve predictions
uvicorn aeroforesight.serving.api:app --reload   # http://localhost:8000/docs

# 4. explore the dashboard
streamlit run dashboard/streamlit_app.py         # http://localhost:8501
```

Or with Docker:

```bash
docker compose up --build
# API       → http://localhost:8000/docs
# Dashboard → http://localhost:8501
```

> The platform runs **without any API key**. Set `ANTHROPIC_API_KEY` in `.env`
> (copy `.env.example`) to turn the templated briefing into a full Claude-written
> narrative. Model defaults to `claude-opus-4-8`.

---

## Components

### 🧠 Deep Learning — delay predictor (`models/dl_delay.py`)
Feed-forward net predicting P(arrival delay > 15 min) from network, weather and
congestion features. PyTorch backend with a transparent scikit-learn fallback,
so it trains anywhere. Reports validation AUC; gated in the pipeline.

### 🎯 Reinforcement Learning — network optimiser (`models/rl_network.py`)
Tabular Q-learning agent that allocates a limited pool of *schedule buffer*
across congested hubs to minimise cascading-delay cost net of utilisation cost.
`recommend_allocation()` returns a concrete plan for today's hub congestion.

### 🔭 Futuristic forecasting (`models/forecast.py`)
Deterministic 2040 scenario paths for traffic growth, SAF adoption (ReFuelEU),
EU-ETS carbon price, CO₂, and the resulting green cost (carbon bill + SAF premium).

### 🏢 Business analytics (`models/business.py`)
Route economics, carrier market share, hub concentration (**HHI**), network
connectivity, and delay **revenue-at-risk** (EU261-style exposure).

### 💬 LLM foresight briefing (`models/llm_insights.py`)
Feeds the structured outputs to Claude (`claude-opus-4-8`, adaptive thinking,
streamed) to produce an executive briefing. Falls back to a deterministic
template offline.

### ⚙️ MLOps (`mlops/`)
- `pipeline.py` — orchestrates the whole run and writes `artifacts/foresight_report.json`
- `registry.py` — file-based model registry (MLflow-swappable)
- `monitoring.py` — PSI drift detection + performance gate

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | liveness + loaded model version |
| POST | `/predict/delay` | delay probability for a flight |
| GET  | `/network/kpis` | current network KPIs |
| GET  | `/forecast` | 2040 scenario forecast |
| GET  | `/rl/recommend?bands=2,1,2,0` | RL buffer allocation |
| GET  | `/briefing` | latest executive foresight briefing |

Example:

```bash
curl -X POST localhost:8000/predict/delay -H 'content-type: application/json' \
  -d '{"origin":"LHR","dest":"FRA","carrier":"LH","congestion":1.8,"weather_index":3.2}'
```

---

## Tests

```bash
pytest -q        # or: make test
```

Smoke tests cover every layer (data → DL → RL → forecast → business → monitoring
→ briefing) and run in seconds using the scikit-learn fallback.

---

## Project layout

```
AeroForesight-EU/
├── config/config.yaml            # hubs, carriers, model + scenario params
├── src/aeroforesight/
│   ├── data/       generate.py, loader.py
│   ├── features/   build.py
│   ├── models/     dl_delay.py, rl_network.py, forecast.py, business.py, llm_insights.py
│   ├── mlops/      pipeline.py, registry.py, monitoring.py
│   └── serving/    api.py
├── dashboard/      streamlit_app.py
├── tests/          test_smoke.py
├── Dockerfile, docker-compose.yml, Makefile, .github/workflows/ci.yml
└── requirements.txt, pyproject.toml
```

---

## Notes & disclaimers

- Data is **synthetic** (no real airline/airport feeds); scenario numbers are
  illustrative, not investment advice.
- Swap `data/generate.py` for real OpenSky / Eurocontrol / OAG connectors to go
  from demo to production without touching the model or serving code.

MIT licensed.
