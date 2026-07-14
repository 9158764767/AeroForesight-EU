// Futuristic prediction layer — pure functions over live flight state.
//
// Three horizons of "foresight":
//   1. per-flight delay risk (next-hour, logistic model on live kinematics)
//   2. live carbon / EU-ETS exposure (now)
//   3. 2040 scenario forecast (traffic CAGR + SAF blend + ETS carbon price)
//
// No ML runtime needed — the DL model in the Python package produces the same
// shape of output; here the coefficients are distilled so the live layer stays
// dependency-free and streams in real time.

'use strict';

const CRUISE_ALT_M = 10500;   // typical cruise altitude
const CRUISE_SPD_MS = 235;    // ~ 850 km/h

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// ---- 1. Per-flight delay risk -------------------------------------------
// Probability that a currently-airborne flight arrives >15 min late.
// Features: descent phase into a congested hub, low speed, holding-pattern
// signature (low altitude + low vertical rate near a hub), and hub load.
function delayRisk(flight, hubLoad = 0) {
  if (flight.onGround) return 0.03;

  const alt = flight.altitude ?? CRUISE_ALT_M;
  const spd = flight.velocity ?? CRUISE_SPD_MS;
  const vrate = flight.verticalRate ?? 0;

  const descending = vrate < -1.5 ? 1 : 0;
  const lowAlt = alt < 4000 ? 1 : 0;
  const slow = spd < 170 ? 1 : 0;
  const holdingSignature = lowAlt && Math.abs(vrate) < 1.5 && spd < 200 ? 1 : 0;

  // Logistic score. hubLoad in [0,1] scales congestion pressure.
  const z =
    -2.4 +
    1.3 * hubLoad +
    0.9 * descending * hubLoad +
    0.8 * slow +
    1.6 * holdingSignature +
    0.5 * lowAlt;

  return clamp(sigmoid(z), 0.01, 0.98);
}

// ---- 2. Live carbon / EU-ETS exposure -----------------------------------
// Rough per-flight fuel-burn proxy (kg/min) from speed & altitude, then CO2
// and the current EU-ETS cost of that CO2 for the last minute of flight.
function carbonNow(flight, etsPriceEur = 95) {
  if (flight.onGround) return { co2Kg: 0, etsEur: 0, fuelKgMin: 0 };
  const spd = flight.velocity ?? CRUISE_SPD_MS;
  const alt = flight.altitude ?? CRUISE_ALT_M;
  // Higher & faster = more thrust; simple monotonic proxy.
  const fuelKgMin = clamp(30 + spd * 0.11 + (alt / 1000) * 2.2, 25, 120);
  const co2KgMin = fuelKgMin * 3.16; // jet-A1 CO2 factor
  const etsEur = (co2KgMin / 1000) * etsPriceEur; // price is EUR / tonne
  return { co2Kg: co2KgMin, etsEur, fuelKgMin };
}

// ---- 3. 2040 scenario forecast ------------------------------------------
// Given a live traffic base, project each scenario forward, blending SAF and
// pricing the residual fossil CO2 at the scenario's ETS price. This is the
// "attraction" — a live, futuristic what-if projected off the current sky.
const SCENARIOS = {
  baseline:    { label: 'Baseline',    trafficCagr: 0.028, safShare2040: 0.20, etsPrice2040: 140 },
  green_push:  { label: 'Green Push',  trafficCagr: 0.018, safShare2040: 0.42, etsPrice2040: 220 },
  high_growth: { label: 'High Growth', trafficCagr: 0.041, safShare2040: 0.14, etsPrice2040: 110 },
};

// `annualFlights` is an annualised base (see server.js: live airborne count is
// bridged to ~EU annual departures). avgCo2PerFlightTonnes ≈ intra-EU narrowbody.
function forecast2040(annualFlights, baseYear = 2035, targetYear = 2040, avgCo2PerFlightTonnes = 12) {
  const liveTraffic = annualFlights;
  const years = targetYear - baseYear;
  const out = {};
  for (const [key, s] of Object.entries(SCENARIOS)) {
    const series = [];
    for (let y = 0; y <= years; y++) {
      const year = baseYear + y;
      const traffic = liveTraffic * Math.pow(1 + s.trafficCagr, y);
      const safShare = (s.safShare2040 / years) * y; // linear ramp
      const fossilFraction = 1 - safShare;
      const co2Mt = (traffic * avgCo2PerFlightTonnes * fossilFraction) / 1e6; // megatonnes/yr equiv
      const etsPrice = 95 + ((s.etsPrice2040 - 95) / years) * y;
      const etsCostBnEur = (co2Mt * 1e6 * etsPrice) / 1e9; // €bn/yr
      series.push({
        year,
        traffic: Math.round(traffic),
        safShare: +(safShare * 100).toFixed(1),
        co2Mt: +co2Mt.toFixed(2),
        etsPrice: Math.round(etsPrice),
        etsCostBnEur: +etsCostBnEur.toFixed(2),
      });
    }
    out[key] = { label: s.label, series };
  }
  return out;
}

module.exports = { delayRisk, carbonNow, forecast2040, SCENARIOS };
