#!/usr/bin/env python3
"""
Previous Flight Finder
─────────────────────────────────────────────────────────────
Combines live ADS-B tracking (via sbs_raw_connector) with
FlightRadar24 API (via fr24_client) to find and display the
previous flight of a given flight number among currently
tracked aircraft.

Usage:
    # Use defaults (flight=AAL123, SBS feed from adsbhub)
    python previous_flight_finder.py

    # Specify flight number
    python previous_flight_finder.py --flight KE937

    # Custom ADS-B source
    python previous_flight_finder.py --flight UAL456 --host 10.0.0.15 --mode raw

    # With FR24 API token
    python previous_flight_finder.py --flight KE937 --fr24-token YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Import from sbs_raw_connector (same directory) ──────────────────────────
from sbs_raw_connector import (
    AircraftTracker,
    AircraftState,
    TrackerFeedWorker,
    TrackerFeedConfig,
)

# ── Import from fr24_client (in speech/ directory) ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "speech"))
from fr24_client import get_previous_flight

# ── Load .env from departure_prediction/.env ─────────────────────────────────
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / "departure_prediction" / ".env"


def _load_env(env_path: Path = _ENV_PATH) -> dict[str, str]:
    """Parse a .env file into a dict (no external dependencies)."""
    env = {}
    if not env_path.is_file():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and (
            (value[0] == value[-1] == '"') or
            (value[0] == value[-1] == "'")
        ):
            value = value[1:-1]
        env[key] = value
    return env


_dotenv = _load_env()
FR24_API_TOKEN = _dotenv.get("FR24_API_TOKEN", "")
GOOGLE_MAPS_API_KEY = _dotenv.get("GOOGLE_MAPS_API_KEY", "")


# ═════════════════════════════════════════════════════════════════════════════
#  Airport Coordinates DB  (ICAO code → lat, lon)
# ═════════════════════════════════════════════════════════════════════════════

AIRPORT_COORDS: dict[str, tuple[float, float]] = {
    # ── North America ────────────────────────────────────────────────────────
    "KJFK": (40.6413, -73.7781),   # New York JFK
    "KLAX": (33.9416, -118.4085),  # Los Angeles
    "KORD": (41.9742, -87.9073),   # Chicago O'Hare
    "KATL": (33.6407, -84.4277),   # Atlanta
    "KDFW": (32.8998, -97.0403),   # Dallas/Fort Worth
    "KSFO": (37.6213, -122.3790),  # San Francisco
    "KMIA": (25.7959, -80.2870),   # Miami
    "KDEN": (39.8561, -104.6737),  # Denver
    "KBOS": (42.3656, -71.0096),   # Boston
    "KLAS": (36.0840, -115.1537),  # Las Vegas
    "KSEA": (47.4502, -122.3088),  # Seattle
    "KMCO": (28.4312, -81.3081),   # Orlando
    "KEWR": (40.6895, -74.1745),   # Newark
    "KPHL": (39.8744, -75.2424),   # Philadelphia
    "KIAD": (38.9531, -77.4565),   # Washington Dulles
    "KDCA": (38.8512, -77.0402),   # Washington Reagan
    "KBUF": (42.9405, -78.7322),   # Buffalo
    "CYYZ": (43.6777, -79.6248),   # Toronto Pearson
    "CYVR": (49.1967, -123.1815),  # Vancouver
    "CYUL": (45.4706, -73.7408),   # Montreal
    "MMMX": (19.4363, -99.0721),   # Mexico City

    # ── Europe ───────────────────────────────────────────────────────────────
    "EGLL": (51.4700, -0.4543),    # London Heathrow
    "LFPG": (49.0097, 2.5479),     # Paris CDG
    "EDDF": (50.0379, 8.5622),     # Frankfurt
    "EHAM": (52.3105, 4.7683),     # Amsterdam Schiphol
    "LEMD": (40.4983, -3.5676),    # Madrid Barajas
    "LIRF": (41.8003, 12.2389),    # Rome Fiumicino
    "LOWW": (48.1103, 16.5697),    # Vienna
    "LSZH": (47.4647, 8.5492),     # Zurich
    "EKCH": (55.6180, 12.6508),    # Copenhagen
    "ENGM": (60.1939, 11.1004),    # Oslo
    "EFHK": (60.3172, 24.9633),    # Helsinki
    "LEBL": (41.2971, 2.0785),     # Barcelona
    "EDDM": (48.3538, 11.7861),    # Munich
    "EGKK": (51.1537, -0.1821),    # London Gatwick
    "LPPT": (38.7813, -9.1359),    # Lisbon
    "UUEE": (55.9726, 37.4146),    # Moscow Sheremetyevo
    "LTFM": (41.2753, 28.7519),    # Istanbul

    # ── Asia ─────────────────────────────────────────────────────────────────
    "RKSI": (37.4602, 126.4407),   # Seoul Incheon
    "RKSS": (37.5586, 126.7906),   # Seoul Gimpo
    "RJTT": (35.5533, 139.7811),   # Tokyo Haneda
    "RJAA": (35.7647, 140.3864),   # Tokyo Narita
    "VHHH": (22.3080, 113.9185),   # Hong Kong
    "WSSS": (1.3502, 103.9944),    # Singapore Changi
    "VTBS": (13.6900, 100.7501),   # Bangkok Suvarnabhumi
    "RPLL": (14.5086, 121.0198),   # Manila
    "WMKK": (2.7456, 101.7099),    # Kuala Lumpur
    "ZBAD": (39.5098, 116.4105),   # Beijing Daxing
    "ZSPD": (31.1443, 121.8083),   # Shanghai Pudong
    "VABB": (19.0896, 72.8656),    # Mumbai
    "VIDP": (28.5562, 77.1000),    # Delhi
    "OMDB": (25.2528, 55.3644),    # Dubai
    "OEJN": (21.6796, 39.1565),    # Jeddah
    "LLBG": (32.0114, 34.8867),    # Tel Aviv Ben Gurion

    # ── Oceania ──────────────────────────────────────────────────────────────
    "NZAA": (-36.8087, 174.7924),  # Auckland
    "NZCH": (-43.4894, 172.5322),  # Christchurch
    "YSSY": (-33.9461, 151.1772),  # Sydney
    "YMML": (-37.6733, 144.8433),  # Melbourne

    # ── South America ────────────────────────────────────────────────────────
    "SBGR": (-23.4356, -46.4731),  # Sao Paulo Guarulhos
    "SCEL": (-33.3930, -70.7858),  # Santiago
    "SAEZ": (-34.8222, -58.5358),  # Buenos Aires Ezeiza
    "SKBO": (4.7016, -74.1469),    # Bogota

    # ── Africa ───────────────────────────────────────────────────────────────
    "FAOR": (-26.1392, 28.2460),   # Johannesburg
    "HECA": (30.1219, 31.4056),    # Cairo
    "DNMM": (6.5774, 3.3212),     # Lagos
    "GMMN": (33.3675, -7.5900),    # Casablanca
}


_google_geocode_cache: dict[str, Optional[tuple[float, float]]] = {}


def _geocode_airport_google(icao: str, api_key: str) -> Optional[tuple[float, float]]:
    """Resolve airport coordinates via Google Geocoding API.

    Caches results in-process so the same ICAO is only looked up once.
    """
    icao = icao.upper().strip()
    if icao in _google_geocode_cache:
        return _google_geocode_cache[icao]

    try:
        import requests  # already a dependency via fr24_client
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": f"{icao} airport", "key": api_key}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            if results:
                loc = results[0]["geometry"]["location"]
                coords = (loc["lat"], loc["lng"])
                _google_geocode_cache[icao] = coords
                return coords
    except Exception as exc:
        print(f"    (Google Geocoding failed for {icao}: {exc})")

    _google_geocode_cache[icao] = None
    return None


# Airport lookup mode — set at runtime via CLI --airport-lookup
_airport_lookup_mode: str = "auto"  # "db", "google", "auto"


def get_airport_coords(icao: str) -> Optional[tuple[float, float]]:
    """Look up airport coordinates by ICAO code.

    Lookup strategy depends on ``_airport_lookup_mode``:
      - "db"     : hardcoded DB only
      - "google" : Google Geocoding API only (requires GOOGLE_MAPS_API_KEY)
      - "auto"   : try DB first, fall back to Google API if not found
    """
    if not icao:
        return None
    icao_upper = icao.upper().strip()

    if _airport_lookup_mode == "google":
        if GOOGLE_MAPS_API_KEY:
            return _geocode_airport_google(icao_upper, GOOGLE_MAPS_API_KEY)
        print(f"    (Google API requested but GOOGLE_MAPS_API_KEY not set — falling back to DB)")

    # "db" or "auto" — try hardcoded first
    result = AIRPORT_COORDS.get(icao_upper)
    if result is not None:
        return result

    # "auto" — DB miss, try Google
    if _airport_lookup_mode == "auto" and GOOGLE_MAPS_API_KEY:
        coords = _geocode_airport_google(icao_upper, GOOGLE_MAPS_API_KEY)
        if coords is not None:
            print(f"    (Resolved {icao_upper} via Google Geocoding: {coords[0]:.4f}, {coords[1]:.4f})")
            return coords

    return None


# ═════════════════════════════════════════════════════════════════════════════
#  Haversine Distance
# ═════════════════════════════════════════════════════════════════════════════

_EARTH_RADIUS_KM = 6371.0
_KM_PER_NM = 1.852  # 1 nautical mile = 1.852 km


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres (Haversine)."""
    lat1, lon1, lat2, lon2 = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


# ═════════════════════════════════════════════════════════════════════════════
#  Delay Estimation
# ═════════════════════════════════════════════════════════════════════════════

def _parse_takeoff_utc(raw: str) -> Optional[datetime]:
    """Parse a datetime string from FR24 into a timezone-aware UTC datetime."""
    if not raw:
        return None
    try:
        s = raw.strip()
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _fmt_duration(minutes: float) -> str:
    """Format minutes into 'Xh Ym' string."""
    h = int(abs(minutes)) // 60
    m = int(abs(minutes)) % 60
    sign = "-" if minutes < 0 else ""
    if h > 0:
        return f"{sign}{h}h {m:02d}m"
    return f"{sign}{m}m"


def estimate_delay(
    origin_coords: tuple[float, float],
    dest_coords: tuple[float, float],
    current_coords: tuple[float, float],
    takeoff_utc: datetime,
    avg_speed_kt: float = 450.0,
) -> dict:
    """Estimate delay of an in-flight aircraft.

    Uses Haversine distances and a fixed average cruise speed.

    Args:
        origin_coords: (lat, lon) of departure airport
        dest_coords: (lat, lon) of destination airport
        current_coords: (lat, lon) from ADS-B
        takeoff_utc: actual takeoff time (UTC)
        avg_speed_kt: average cruise speed in knots (default 450)

    Returns:
        dict with keys:
            total_dist_km, remaining_km, covered_km, progress_pct,
            elapsed_min, expected_elapsed_min, delay_minutes,
            delay_str, warning
    """
    avg_speed_kmh = avg_speed_kt * _KM_PER_NM

    total_km = haversine_km(*origin_coords, *dest_coords)
    remaining_km = haversine_km(*current_coords, *dest_coords)
    covered_km = total_km - remaining_km
    progress_pct = (covered_km / total_km * 100.0) if total_km > 0 else 0.0

    now = datetime.now(timezone.utc)
    elapsed_sec = (now - takeoff_utc).total_seconds()
    elapsed_min = elapsed_sec / 60.0

    # How long *should* it take to cover covered_km at avg speed?
    expected_elapsed_min = (covered_km / avg_speed_kmh * 60.0) if avg_speed_kmh > 0 else 0.0

    delay_minutes = elapsed_min - expected_elapsed_min

    # Classify
    if delay_minutes > 30:
        warning = "SIGNIFICANT DELAY — expect late arrival"
    elif delay_minutes > 15:
        warning = "MODERATE DELAY — may arrive late"
    elif delay_minutes > 5:
        warning = "MINOR DELAY"
    else:
        warning = None  # on time or ahead

    return {
        "total_dist_km": total_km,
        "remaining_km": remaining_km,
        "covered_km": covered_km,
        "progress_pct": progress_pct,
        "elapsed_min": elapsed_min,
        "expected_elapsed_min": expected_elapsed_min,
        "delay_minutes": delay_minutes,
        "delay_str": _fmt_duration(delay_minutes),
        "warning": warning,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  Callsign Search (equivalent to C++ HookByCallsign)
# ═════════════════════════════════════════════════════════════════════════════

def find_by_callsign(tracker: AircraftTracker, callsign: str) -> Optional[AircraftState]:
    """Search the tracker for an aircraft matching the given callsign."""
    callsign_upper = callsign.upper().strip()
    with tracker.lock:
        for ac in tracker.aircraft.values():
            if ac.callsign.upper().strip() == callsign_upper:
                return ac
    return None


def find_by_icao(tracker: AircraftTracker, icao_hex: str) -> Optional[AircraftState]:
    """Search the tracker for an aircraft matching the given ICAO hex address."""
    icao_lower = icao_hex.lower().strip()
    with tracker.lock:
        return tracker.aircraft.get(icao_lower)


def list_tracked_aircraft(tracker: AircraftTracker) -> list[AircraftState]:
    """Return a snapshot of all currently tracked aircraft."""
    now = time.time()
    with tracker.lock:
        return [
            ac for ac in tracker.aircraft.values()
            if now - ac.last_seen < tracker.timeout
        ]


# ═════════════════════════════════════════════════════════════════════════════
#  Display Helpers
# ═════════════════════════════════════════════════════════════════════════════

def format_aircraft(ac: AircraftState) -> str:
    """Format an AircraftState into a human-readable single line."""
    parts = [f"ICAO={ac.icao}"]
    if ac.callsign:
        parts.append(f"CS={ac.callsign}")
    if ac.altitude is not None:
        parts.append(f"Alt={ac.altitude:,}ft")
    if ac.ground_speed is not None:
        parts.append(f"GS={ac.ground_speed:.0f}kt")
    if ac.lat is not None and ac.lon is not None:
        parts.append(f"Pos=({ac.lat:.4f},{ac.lon:.4f})")
    if ac.track is not None:
        parts.append(f"Trk={ac.track:.0f}deg")
    if ac.squawk:
        parts.append(f"Sqk={ac.squawk}")
    parts.append(f"Msgs={ac.messages}")
    return " | ".join(parts)


def print_divider(char="─", width=70):
    print(char * width)


def _show_similar(tracked: list[AircraftState], callsign: str):
    """Show aircraft with a similar callsign prefix (e.g. KAL for KAL412)."""
    prefix = callsign[:3].upper()
    partial = [ac for ac in tracked if ac.callsign.upper().startswith(prefix)]
    if partial:
        print(f"\n    Similar callsigns (prefix '{prefix}'):")
        for ac in partial:
            print(f"      {format_aircraft(ac)}")


def estimate_previous_leg_delay_minutes(
    flight_no: str,
    *,
    fr24_token: Optional[str] = None,
    host: str = "data.adsbhub.org",
    mode: str = "sbs",
    raw_port: int = 30002,
    sbs_port: int = 5002,
    timeout: float = 60.0,
    reconnect_delay: float = 3.0,
    collect_time: int = 5,
    avg_speed: float = 450.0,
    expected_origin: Optional[str] = None,
    expected_dest: Optional[str] = None,
    expected_date: Optional[datetime] = None,
) -> dict:
    """Return a numeric delay estimate (minutes) for previous-leg turnaround risk.

    This is the non-CLI adapter for hybrid predictor integration.
    """
    token = (fr24_token or FR24_API_TOKEN or "").strip()
    result = {
        "delay_minutes": 0,
        "found": False,
        "in_air": False,
        "source": "none",
        "reason": "not_started",
        "adsb_tracked": False,
        "validation_mismatch": False,
        "validation_notes": [],
        "current_flight": {},
        "previous_flight": {},
    }

    if not token:
        result["source"] = "error"
        result["reason"] = "missing_fr24_token"
        return result

    try:
        fr24_result = get_previous_flight(token, flight_no)
    except Exception as exc:
        result["source"] = "error"
        result["reason"] = f"fr24_error:{exc}"
        return result

    current = fr24_result.get("current_flight", {}) or {}
    prev = fr24_result.get("previous_flight") or {}
    result["current_flight"] = current
    result["previous_flight"] = prev

    validation_notes: list[str] = []
    if expected_origin and current.get("origin"):
        if expected_origin.upper().strip() != str(current.get("origin")).upper().strip():
            validation_notes.append(
                f"origin_mismatch expected={expected_origin} fr24={current.get('origin')}"
            )
    if expected_dest and current.get("destination"):
        if expected_dest.upper().strip() != str(current.get("destination")).upper().strip():
            validation_notes.append(
                f"dest_mismatch expected={expected_dest} fr24={current.get('destination')}"
            )
    if expected_date and prev.get("datetime_takeoff"):
        takeoff_dt = _parse_takeoff_utc(prev.get("datetime_takeoff"))
        if takeoff_dt and takeoff_dt.date() != expected_date.date():
            validation_notes.append(
                f"date_mismatch expected={expected_date.date()} fr24_prev_takeoff={takeoff_dt.date()}"
            )
    result["validation_notes"] = validation_notes
    result["validation_mismatch"] = bool(validation_notes)

    if not prev:
        result["source"] = "fr24"
        result["reason"] = "previous_not_found"
        return result

    result["found"] = True
    prev_in_air = bool(prev.get("in_air"))
    result["in_air"] = prev_in_air

    if not prev_in_air:
        result["source"] = "fr24"
        result["reason"] = "previous_landed"
        result["delay_minutes"] = 0
        return result

    prev_callsign = (prev.get("callsign") or "").strip()
    if not prev_callsign:
        result["source"] = "fr24"
        result["reason"] = "missing_previous_callsign"
        return result

    orig_icao = prev.get("orig_icao")
    dest_icao = prev.get("dest_icao")
    origin_coords = get_airport_coords(orig_icao) if orig_icao else None
    dest_coords = get_airport_coords(dest_icao) if dest_icao else None
    takeoff_dt = _parse_takeoff_utc(prev.get("datetime_takeoff"))

    stop_event = threading.Event()
    tracker = AircraftTracker(timeout=timeout)
    workers: list[TrackerFeedWorker] = []

    if mode in ("raw", "both"):
        workers.append(TrackerFeedWorker(
            TrackerFeedConfig("RAW", host, raw_port, None, tracker.update_raw),
            stop_event=stop_event,
            reconnect_delay=reconnect_delay,
        ))
    if mode in ("sbs", "both"):
        workers.append(TrackerFeedWorker(
            TrackerFeedConfig("SBS", host, sbs_port, None, tracker.update_sbs),
            stop_event=stop_event,
            reconnect_delay=reconnect_delay,
        ))

    try:
        for w in workers:
            w.start()

        for _ in range(max(0, int(collect_time))):
            if stop_event.is_set():
                break
            time.sleep(1)

        found = find_by_callsign(tracker, prev_callsign)
        if found:
            result["adsb_tracked"] = True
            can_estimate = (
                origin_coords is not None
                and dest_coords is not None
                and takeoff_dt is not None
                and found.lat is not None
                and found.lon is not None
            )
            if can_estimate:
                delay_obj = estimate_delay(
                    origin_coords=origin_coords,
                    dest_coords=dest_coords,
                    current_coords=(found.lat, found.lon),
                    takeoff_utc=takeoff_dt,
                    avg_speed_kt=avg_speed,
                )
                result["delay_minutes"] = max(0, int(round(delay_obj["delay_minutes"])))
                result["source"] = "adsb_live"
                result["reason"] = "in_air_adsb_tracked"
                return result

            result["source"] = "adsb_live"
            result["reason"] = "in_air_adsb_tracked_missing_meta"
            return result

        if origin_coords and dest_coords and takeoff_dt:
            total_km = haversine_km(*origin_coords, *dest_coords)
            avg_speed_kmh = avg_speed * _KM_PER_NM
            expected_total_min = (total_km / avg_speed_kmh * 60.0) if avg_speed_kmh > 0 else 0.0
            elapsed_min = (datetime.now(timezone.utc) - takeoff_dt).total_seconds() / 60.0
            fallback_delay = max(0.0, elapsed_min - expected_total_min)
            result["delay_minutes"] = int(round(fallback_delay))
            result["source"] = "fr24_fallback"
            result["reason"] = "in_air_adsb_not_found"
            return result

        result["source"] = "fr24_fallback"
        result["reason"] = "in_air_adsb_not_found_missing_meta"
        return result
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=2)


# ═════════════════════════════════════════════════════════════════════════════
#  Main Logic
# ═════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> int:
    stop_event = threading.Event()

    def handle_signal(_sig, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("=" * 70)
    print("  Previous Flight Finder")
    print("=" * 70)
    print(f"  Flight      : {args.flight}")
    print(f"  FR24 Token  : {'(set)' if args.fr24_token else '(not set — check departure_prediction/.env)'}")
    print(f"  ADS-B Host  : {args.host} ({args.mode.upper()})")
    print("=" * 70)

    # ── 1. Query FR24 for previous flight (first — decides if ADS-B needed) ──
    print(f"\n[1/2] Querying FR24 API for previous flight of {args.flight}...")

    prev_callsign = None
    prev_in_air = False
    prev = None
    fr24_result = None

    try:
        fr24_result = get_previous_flight(args.fr24_token, args.flight)
        current = fr24_result.get("current_flight", {})
        prev = fr24_result.get("previous_flight")
        route_history = fr24_result.get("route_history", "")

        print(f"      Current Flight : {current.get('flight_no')}")
        print(f"        Registration : {current.get('registration')}")
        print(f"        Route        : {current.get('origin')} -> {current.get('destination')}")

        if prev:
            prev_callsign = prev.get("callsign")
            prev_in_air = prev.get("in_air", False)
            status_str = "IN FLIGHT" if prev_in_air else "LANDED"

            print(f"      Previous Flight: {prev.get('flight_no')}")
            print(f"        Callsign     : {prev_callsign}")
            print(f"        Route        : {prev.get('origin')} -> {prev.get('destination')}")
            print(f"        Status       : {status_str}")
            if prev.get("datetime_takeoff"):
                print(f"        Takeoff      : {prev['datetime_takeoff']}")
            if prev.get("datetime_landed"):
                print(f"        Landed       : {prev['datetime_landed']}")
            if route_history:
                print(f"        Route History: {route_history}")
        else:
            print("      Previous Flight: NOT FOUND")

    except Exception as e:
        print(f"      ERROR: FR24 API call failed — {e}")
        print("      (Continuing with manual callsign search...)")

    # Use fallback callsign if FR24 failed or didn't return one
    if not prev_callsign and args.prev_callsign:
        prev_callsign = args.prev_callsign
        prev_in_air = True  # assume in-flight if manually specified
        print(f"      Using provided fallback callsign: {prev_callsign}")

    # ── 2. Branch: IN FLIGHT vs LANDED ───────────────────────────────────────
    if not prev_callsign:
        print("\n[2/2] No previous callsign found — done.")

    elif prev_in_air:
        # ── Branch A: Previous flight is still IN THE AIR ────────────────
        print_divider("═")
        print(f"  BRANCH: PREVIOUS FLIGHT IS IN THE AIR")
        print_divider("═")
        print(f"  The previous flight '{prev_callsign}' has NOT landed yet.")
        print(f"  → The aircraft is still en route and cannot begin the current flight on time.")
        print(f"  → Expect a DELAY for {args.flight}.")

        # Resolve airport coordinates for delay estimation
        orig_icao = prev.get("orig_icao") if prev else None
        dest_icao = prev.get("dest_icao") if prev else None
        origin_coords = get_airport_coords(orig_icao) if orig_icao else None
        dest_coords = get_airport_coords(dest_icao) if dest_icao else None
        takeoff_dt = _parse_takeoff_utc(prev.get("datetime_takeoff")) if prev else None

        # ── Start ADS-B collection (only for IN FLIGHT) ─────────────────
        tracker = AircraftTracker(timeout=args.timeout)
        workers: list[TrackerFeedWorker] = []

        if args.mode in ("raw", "both"):
            workers.append(TrackerFeedWorker(
                TrackerFeedConfig("RAW", args.host, args.raw_port, None, tracker.update_raw),
                stop_event=stop_event,
                reconnect_delay=args.reconnect_delay,
            ))
        if args.mode in ("sbs", "both"):
            workers.append(TrackerFeedWorker(
                TrackerFeedConfig("SBS", args.host, args.sbs_port, None, tracker.update_sbs),
                stop_event=stop_event,
                reconnect_delay=args.reconnect_delay,
            ))

        for w in workers:
            w.start()

        collect_sec = args.collect_time
        print(f"\n  [2/2] Collecting ADS-B data for {collect_sec}s...")

        for i in range(collect_sec):
            if stop_event.is_set():
                break
            time.sleep(1)
            count = len(list_tracked_aircraft(tracker))
            sys.stdout.write(f"\r        {i+1}/{collect_sec}s — tracking {count} aircraft")
            sys.stdout.flush()
        print()

        tracked = list_tracked_aircraft(tracker)
        print(f"        => {len(tracked)} aircraft tracked")

        # Search for the previous flight
        print(f"  Searching for callsign '{prev_callsign}'...")
        found = find_by_callsign(tracker, prev_callsign)

        if found:
            print_divider()
            print(f"    TRACKED — {format_aircraft(found)}")
            print_divider()

            # ── Delay estimation ─────────────────────────────────────────
            can_estimate = (
                origin_coords is not None
                and dest_coords is not None
                and takeoff_dt is not None
                and found.lat is not None
                and found.lon is not None
            )

            if can_estimate:
                result = estimate_delay(
                    origin_coords=origin_coords,
                    dest_coords=dest_coords,
                    current_coords=(found.lat, found.lon),
                    takeoff_utc=takeoff_dt,
                    avg_speed_kt=args.avg_speed,
                )

                orig_label = prev.get("origin") or orig_icao
                dest_label = prev.get("destination") or dest_icao

                print()
                print(f"  Delay Estimation:")
                print(f"    Route        : {orig_label} -> {dest_label} ({result['total_dist_km']:,.0f} km)")
                print(f"    Progress     : {result['covered_km']:,.0f} km covered ({result['progress_pct']:.1f}%)")
                print(f"    Elapsed      : {_fmt_duration(result['elapsed_min'])} since takeoff")
                print(f"    Expected     : {_fmt_duration(result['expected_elapsed_min'])} for this distance at {args.avg_speed:.0f}kt")

                dm = result["delay_minutes"]
                if dm > 0:
                    print(f"    Delay        : +{result['delay_str']} behind schedule")
                else:
                    print(f"    Delay        : {result['delay_str']} (on time / ahead)")

                if result["warning"]:
                    print(f"    >>> WARNING: {result['warning']}")
            else:
                missing = []
                if origin_coords is None:
                    missing.append(f"origin airport coords ({orig_icao})")
                if dest_coords is None:
                    missing.append(f"dest airport coords ({dest_icao})")
                if takeoff_dt is None:
                    missing.append("takeoff time")
                if found.lat is None or found.lon is None:
                    missing.append("ADS-B position")
                print(f"\n    Cannot estimate delay — missing: {', '.join(missing)}")
        else:
            print(f"    Aircraft '{prev_callsign}' not in ADS-B range.")
            _show_similar(tracked, prev_callsign)

            # Even without ADS-B, show basic route info if available
            if origin_coords and dest_coords and takeoff_dt:
                total_km = haversine_km(*origin_coords, *dest_coords)
                avg_speed_kmh = args.avg_speed * _KM_PER_NM
                expected_total_min = (total_km / avg_speed_kmh * 60.0) if avg_speed_kmh > 0 else 0
                elapsed_min = (datetime.now(timezone.utc) - takeoff_dt).total_seconds() / 60.0
                orig_label = prev.get("origin") or orig_icao
                dest_label = prev.get("destination") or dest_icao
                print(f"\n    Route info   : {orig_label} -> {dest_label} ({total_km:,.0f} km)")
                print(f"    Elapsed      : {_fmt_duration(elapsed_min)} since takeoff")
                print(f"    Est. total   : {_fmt_duration(expected_total_min)} at {args.avg_speed:.0f}kt")
                remaining_min = expected_total_min - elapsed_min
                if remaining_min > 0:
                    print(f"    Est. remaining: ~{_fmt_duration(remaining_min)}")
                else:
                    print(f"    Est. remaining: should have arrived {_fmt_duration(-remaining_min)} ago — likely delayed")

        # Stop ADS-B workers
        stop_event.set()
        for w in workers:
            w.join(timeout=2)

    else:
        # ── Branch B: Previous flight has LANDED ─────────────────────────
        print_divider("═")
        print(f"  BRANCH: PREVIOUS FLIGHT HAS LANDED")
        print_divider("═")
        print(f"  The previous flight '{prev_callsign}' has landed.")
        print(f"  → Aircraft should be available for turnaround.")
        print(f"  [2/2] ADS-B lookup skipped (aircraft already on the ground).")

    print("\nDone.")
    return 0


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find previous flight using ADS-B + FR24 API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Flight input
    flight = parser.add_argument_group("flight query")
    flight.add_argument("--flight", default="AAL123",
                        help="Flight number to look up (default: %(default)s)")
    flight.add_argument("--prev-callsign", default=None,
                        help="Fallback: manually specify previous flight callsign "
                             "(skips FR24 API if set alone)")

    # ADS-B connection
    adsb = parser.add_argument_group("ADS-B connection")
    adsb.add_argument("--host", default="data.adsbhub.org",
                      help="ADS-B feed host (default: %(default)s)")
    adsb.add_argument("--mode", choices=["raw", "sbs", "both"], default="sbs",
                      help="Feed mode (default: %(default)s)")
    adsb.add_argument("--raw-port", type=int, default=30002,
                      help="Raw feed port (default: %(default)s)")
    adsb.add_argument("--sbs-port", type=int, default=5002,
                      help="SBS feed port (default: %(default)s)")
    adsb.add_argument("--timeout", type=float, default=60.0,
                      help="Aircraft timeout in seconds (default: %(default)s)")
    adsb.add_argument("--reconnect-delay", type=float, default=3.0,
                      help="Reconnect delay (default: %(default)s)")
    adsb.add_argument("--collect-time", type=int, default=5,
                      help="Seconds to collect ADS-B data (default: %(default)s)")

    # FR24 API
    fr24 = parser.add_argument_group("FR24 API")
    fr24.add_argument("--fr24-token", default=None,
                      help="FlightRadar24 API bearer token "
                           "(default: loaded from departure_prediction/.env)")

    # Delay estimation
    delay = parser.add_argument_group("delay estimation")
    delay.add_argument("--avg-speed", type=float, default=450.0,
                       help="Average cruise speed in knots for delay estimation "
                            "(default: %(default)s kt)")
    delay.add_argument("--airport-lookup", choices=["db", "google", "auto"], default="auto",
                       help="Airport coordinate source: "
                            "db = hardcoded only, "
                            "google = Google Geocoding API only, "
                            "auto = DB first then Google fallback "
                            "(default: %(default)s)")

    return parser.parse_args()


def main() -> int:
    global _airport_lookup_mode
    args = parse_args()
    # Resolve FR24 token: CLI flag > .env > empty
    if args.fr24_token is None:
        args.fr24_token = FR24_API_TOKEN
    # Set airport lookup mode
    _airport_lookup_mode = args.airport_lookup
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
