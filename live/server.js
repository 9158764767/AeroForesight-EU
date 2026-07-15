// AeroForesight-EU — live streaming server (zero external dependencies).
//
//   http://localhost:8787/            live dashboard
//   http://localhost:8787/stream      Server-Sent Events feed (aggregated snapshots)
//   http://localhost:8787/api/health  status
//   http://localhost:8787/api/snapshot latest aggregate (JSON)
//   http://localhost:8787/api/forecast 2040 scenario projection

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { LiveIngest } = require('./ingest');
const { AIRPORTS, nearestAirport } = require('./airports');
const { delayRisk, carbonNow, forecast2040 } = require('./predict');

const PORT = process.env.PORT || 8787;
const MAP_FLIGHT_CAP = 1500;   // cap positions pushed to the browser per tick
const ETS_PRICE_NOW = 95;      // EUR / tonne CO2 (live spot proxy)
const ANNUALIZE_FACTOR = 4200; // ~2,600 concurrently airborne ⇔ ~11M EU flights/yr

const clients = new Set();   // SSE response objects
let latest = null;           // most recent aggregate

// ---- Aggregation: turn a raw snapshot into a dashboard payload -----------
function aggregate(snapshot) {
  const airborne = snapshot.flights.filter((f) => !f.onGround);

  // 1st pass: hub occupancy
  const hubMap = new Map(AIRPORTS.map((a) => [a.iata, { ...a, count: 0, riskSum: 0 }]));
  for (const f of airborne) {
    const near = nearestAirport(f.lat, f.lon, 120);
    if (near) hubMap.get(near.airport.iata).count += 1;
    f._hub = near ? near.airport.iata : null;
  }
  const hubs = [...hubMap.values()].map((h) => {
    const load = Math.min(1.2, h.count / Math.max(1, h.capacity * 0.03));
    return { iata: h.iata, city: h.city, lat: h.lat, lon: h.lon, capacity: h.capacity, count: h.count, load: +load.toFixed(3) };
  });
  const hubLoad = new Map(hubs.map((h) => [h.iata, h.load]));

  // 2nd pass: per-flight prediction + carbon
  let riskSum = 0, highRisk = 0, co2 = 0, ets = 0;
  const flightsOut = [];
  const scored = [];
  for (const f of airborne) {
    const load = f._hub ? hubLoad.get(f._hub) : 0;
    const risk = delayRisk(f, load);
    const c = carbonNow(f, ETS_PRICE_NOW);
    riskSum += risk;
    if (risk >= 0.5) highRisk += 1;
    co2 += c.co2Kg;
    ets += c.etsEur;
    scored.push({ callsign: f.callsign, country: f.country, risk, altitude: f.altitude, velocity: f.velocity, hub: f._hub });
    if (flightsOut.length < MAP_FLIGHT_CAP) {
      flightsOut.push({
        id: f.id, callsign: f.callsign, lat: +f.lat.toFixed(3), lon: +f.lon.toFixed(3),
        heading: Math.round(f.heading), alt: Math.round(f.altitude), risk: +risk.toFixed(2),
      });
    }
  }

  const busiest = hubs.reduce((a, b) => (b.count > a.count ? b : a), hubs[0]);
  const topRisk = scored.sort((a, b) => b.risk - a.risk).slice(0, 10)
    .map((s) => ({ ...s, risk: +s.risk.toFixed(2), altitude: Math.round(s.altitude), velocity: Math.round(s.velocity) }));

  return {
    ts: snapshot.ts,
    source: snapshot.source,
    total: snapshot.count,
    kpis: {
      airborne: airborne.length,
      avgDelayRisk: airborne.length ? +(riskSum / airborne.length).toFixed(3) : 0,
      highRisk,
      co2PerMinT: +(co2 / 1000).toFixed(2),
      etsPerMinEur: Math.round(ets),
      busiestHub: busiest ? `${busiest.iata} (${busiest.count})` : '—',
    },
    hubs,
    flights: flightsOut,
    topRisk,
    // Bridge the instantaneous airborne count to an EU-scale annual departures
    // base (~11M/yr ⇔ ~2,600 concurrently airborne) so 2040 CO2/ETS are realistic.
    forecast: forecast2040(airborne.length * ANNUALIZE_FACTOR),
  };
}

// ---- SSE broadcast ------------------------------------------------------
function broadcast(agg) {
  const payload = `data: ${JSON.stringify(agg)}\n\n`;
  for (const res of clients) res.write(payload);
}

// ---- HTTP server --------------------------------------------------------
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8' };

function serveStatic(req, res) {
  const rel = req.url === '/' ? '/index.html' : req.url.split('?')[0];
  const file = path.join(__dirname, 'public', path.normalize(rel).replace(/^(\.\.[/\\])+/, ''));
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  const url = req.url.split('?')[0];

  if (url === '/stream') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*',
    });
    res.write('retry: 5000\n\n');
    clients.add(res);
    if (latest) res.write(`data: ${JSON.stringify(latest)}\n\n`);
    req.on('close', () => clients.delete(res));
    return;
  }

  if (url === '/api/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ ok: true, source: latest?.source ?? 'starting', clients: clients.size, ts: latest?.ts ?? null }));
  }
  if (url === '/api/snapshot') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(latest ?? { status: 'warming up' }));
  }
  if (url === '/api/forecast') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify(latest?.forecast ?? {}));
  }

  return serveStatic(req, res);
});

// ---- Wire the ingest engine to the broadcaster --------------------------
const ingest = new LiveIngest({ intervalMs: Number(process.env.POLL_MS) || 10000, capture: true });
ingest.on('snapshot', (snap) => {
  latest = aggregate(snap);
  broadcast(latest);
  const k = latest.kpis;
  console.log(`[${new Date(latest.ts).toISOString()}] ${latest.source} | airborne=${k.airborne} avgRisk=${k.avgDelayRisk} highRisk=${k.highRisk} busiest=${k.busiestHub} clients=${clients.size}`);
});
ingest.start();

server.listen(PORT, () => {
  console.log(`AeroForesight-EU live server → http://localhost:${PORT}`);
  console.log(`  dashboard  /            SSE  /stream            API  /api/snapshot`);
});
