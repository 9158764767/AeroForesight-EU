// Live data ingestion — the "data capturing" layer.
//
// Polls the OpenSky Network `/states/all` REST feed bounded to Europe, normalises
// each aircraft state vector, and emits a snapshot on every poll. If the network
// is unavailable it transparently switches to a physics-lite simulator so the
// pipeline never stalls. Every snapshot is appended to data/captures/*.jsonl.

'use strict';

const https = require('https');
const fs = require('fs');
const path = require('path');
const { EventEmitter } = require('events');
const { AIRPORTS, BBOX } = require('./airports');

const CAPTURE_DIR = path.join(__dirname, '..', 'data', 'captures');

// OpenSky state-vector column indices we use.
const IDX = {
  icao24: 0, callsign: 1, country: 2, lon: 5, lat: 6,
  baroAlt: 7, onGround: 8, velocity: 9, heading: 10, vrate: 11, geoAlt: 13,
};

function normalise(state) {
  const cs = (state[IDX.callsign] || '').trim();
  const lat = state[IDX.lat];
  const lon = state[IDX.lon];
  if (lat == null || lon == null) return null;
  return {
    id: state[IDX.icao24],
    callsign: cs || state[IDX.icao24],
    country: state[IDX.country] || '—',
    lat, lon,
    altitude: state[IDX.geoAlt] ?? state[IDX.baroAlt] ?? 0,
    velocity: state[IDX.velocity] ?? 0,
    heading: state[IDX.heading] ?? 0,
    verticalRate: state[IDX.vrate] ?? 0,
    onGround: !!state[IDX.onGround],
  };
}

class LiveIngest extends EventEmitter {
  constructor({ intervalMs = 10000, capture = true } = {}) {
    super();
    this.intervalMs = intervalMs;
    this.capture = capture;
    this.timer = null;
    this.source = 'starting';
    this.simSeed = this._seedSim();
    this.captureFile = path.join(
      CAPTURE_DIR,
      `capture-${new Date().toISOString().slice(0, 10)}.jsonl`
    );
    if (capture) fs.mkdirSync(CAPTURE_DIR, { recursive: true });
  }

  start() {
    this._poll();
    this.timer = setInterval(() => this._poll(), this.intervalMs);
    return this;
  }

  stop() {
    if (this.timer) clearInterval(this.timer);
  }

  _poll() {
    const url =
      `https://opensky-network.org/api/states/all` +
      `?lamin=${BBOX.lamin}&lomin=${BBOX.lomin}&lamax=${BBOX.lamax}&lomax=${BBOX.lomax}`;

    const req = https.get(url, { headers: { 'User-Agent': 'AeroForesight-EU/live' }, timeout: 9000 }, (res) => {
      if (res.statusCode !== 200) { res.resume(); return this._fallback(`http ${res.statusCode}`); }
      let body = '';
      res.on('data', (c) => (body += c));
      res.on('end', () => {
        try {
          const json = JSON.parse(body);
          const flights = (json.states || []).map(normalise).filter(Boolean);
          if (!flights.length) return this._fallback('empty feed');
          this.source = 'opensky-live';
          this._emit(flights);
        } catch (e) {
          this._fallback('parse error');
        }
      });
    });
    req.on('timeout', () => { req.destroy(); this._fallback('timeout'); });
    req.on('error', () => this._fallback('network error'));
  }

  _emit(flights) {
    const snapshot = { ts: Date.now(), source: this.source, count: flights.length, flights };
    if (this.capture) {
      // capture a compact record per snapshot (not every raw flight) to keep files small
      const rec = JSON.stringify({ ts: snapshot.ts, source: snapshot.source, count: snapshot.count });
      fs.appendFile(this.captureFile, rec + '\n', () => {});
    }
    this.emit('snapshot', snapshot);
  }

  // ---- Simulator fallback ------------------------------------------------
  _seedSim() {
    const rng = this._mulberry32(42);
    const fleet = [];
    const carriers = ['LH', 'AF', 'BA', 'FR', 'U2', 'KL'];
    for (let i = 0; i < 900; i++) {
      const a = AIRPORTS[Math.floor(rng() * AIRPORTS.length)];
      const b = AIRPORTS[Math.floor(rng() * AIRPORTS.length)];
      fleet.push({
        id: 'SIM' + i.toString(36),
        callsign: carriers[Math.floor(rng() * carriers.length)] + (100 + Math.floor(rng() * 8900)),
        country: a.country,
        lat: a.lat, lon: a.lon,
        tgtLat: b.lat, tgtLon: b.lon,
        alt: 9000 + rng() * 3000,
        vel: 200 + rng() * 60,
        prog: rng(),
        speedProg: 0.002 + rng() * 0.004,
      });
    }
    return { rng, fleet };
  }

  _mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  _fallback(reason) {
    this.source = `simulator (${reason})`;
    const { rng, fleet } = this.simSeed;
    const flights = fleet.map((f) => {
      f.prog += f.speedProg;
      if (f.prog >= 1) { // arrived — assign a new leg
        f.prog = 0;
        f.lat = f.tgtLat; f.lon = f.tgtLon;
        const b = AIRPORTS[Math.floor(rng() * AIRPORTS.length)];
        f.tgtLat = b.lat; f.tgtLon = b.lon;
      }
      const lat = f.lat + (f.tgtLat - f.lat) * f.prog;
      const lon = f.lon + (f.tgtLon - f.lon) * f.prog;
      const climbing = f.prog < 0.15;
      const descending = f.prog > 0.85;
      return {
        id: f.id,
        callsign: f.callsign,
        country: f.country,
        lat, lon,
        altitude: descending ? f.alt * (1 - (f.prog - 0.85) / 0.15) : climbing ? f.alt * (f.prog / 0.15) : f.alt,
        velocity: descending ? 150 + rng() * 40 : f.vel,
        heading: (Math.atan2(f.tgtLon - lon, f.tgtLat - lat) * 180) / Math.PI,
        verticalRate: descending ? -6 - rng() * 4 : climbing ? 6 + rng() * 4 : (rng() - 0.5),
        onGround: false,
      };
    });
    this._emit(flights);
  }
}

module.exports = { LiveIngest };
