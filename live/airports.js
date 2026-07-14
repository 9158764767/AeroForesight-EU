// European hub reference data (mirrors config/config.yaml).
// Used for congestion attribution, map rendering and business KPIs.

'use strict';

const AIRPORTS = [
  { iata: 'LHR', city: 'London',     country: 'UK', lat: 51.47, lon: -0.45, capacity: 1350 },
  { iata: 'CDG', city: 'Paris',      country: 'FR', lat: 49.01, lon:  2.55, capacity: 1400 },
  { iata: 'FRA', city: 'Frankfurt',  country: 'DE', lat: 50.04, lon:  8.56, capacity: 1500 },
  { iata: 'AMS', city: 'Amsterdam',  country: 'NL', lat: 52.31, lon:  4.76, capacity: 1300 },
  { iata: 'MAD', city: 'Madrid',     country: 'ES', lat: 40.47, lon: -3.56, capacity: 1150 },
  { iata: 'BCN', city: 'Barcelona',  country: 'ES', lat: 41.30, lon:  2.08, capacity:  950 },
  { iata: 'FCO', city: 'Rome',       country: 'IT', lat: 41.80, lon: 12.25, capacity:  900 },
  { iata: 'MUC', city: 'Munich',     country: 'DE', lat: 48.35, lon: 11.79, capacity: 1000 },
  { iata: 'DUB', city: 'Dublin',     country: 'IE', lat: 53.42, lon: -6.27, capacity:  700 },
  { iata: 'VIE', city: 'Vienna',     country: 'AT', lat: 48.11, lon: 16.57, capacity:  740 },
  { iata: 'CPH', city: 'Copenhagen', country: 'DK', lat: 55.62, lon: 12.65, capacity:  680 },
  { iata: 'LIS', city: 'Lisbon',     country: 'PT', lat: 38.77, lon: -9.13, capacity:  620 },
];

// Europe bounding box used for the live OpenSky query and map projection.
const BBOX = { lamin: 35.0, lomin: -15.0, lamax: 60.0, lomax: 30.0 };

const R_EARTH_KM = 6371;

function haversineKm(lat1, lon1, lat2, lon2) {
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R_EARTH_KM * Math.asin(Math.sqrt(a));
}

// Nearest hub within `radiusKm`, else null. Used to attribute congestion.
function nearestAirport(lat, lon, radiusKm = 120) {
  let best = null;
  let bestD = Infinity;
  for (const a of AIRPORTS) {
    const d = haversineKm(lat, lon, a.lat, a.lon);
    if (d < bestD) {
      bestD = d;
      best = a;
    }
  }
  return bestD <= radiusKm ? { airport: best, distanceKm: bestD } : null;
}

module.exports = { AIRPORTS, BBOX, R_EARTH_KM, haversineKm, nearestAirport };
