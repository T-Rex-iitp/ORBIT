#!/usr/bin/env python3
"""
Build a structured departure brief for "when should I leave?" queries.

Output format (machine-readable):
--- DEPARTURE_BRIEF ---
KEY=VALUE
...
--- END_DEPARTURE_BRIEF ---
"""

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote_plus

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

try:
    import requests
except ImportError:
    requests = None

DEFAULT_CONFIG_PATH = Path(__file__).parent / "departure_config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "timezone": "America/New_York",
    "default_origin": "Times Square, Manhattan, NY",
    "default_airport": "JFK",
    "default_flight_lead_minutes": 180,
    "default_security_wait_minutes": 30,
    "default_checkin_minutes": 45,
    "default_walk_to_gate_minutes": 15,
    "default_buffer_minutes": 10,
    "fallback_drive_minutes": 55,
    "google_maps_api_key": "",
    "aviationstack_api_key": "",
    "traffic_request_timeout_seconds": 6,
    "aviationstack_timeout_seconds": 6,
    "open_map_automatically": True,
    "airport_code_aliases": {
        "JFKJ": "JFK",
        "KJFK": "JFK",
        "KLGA": "LGA",
        "KEWR": "EWR",
        "KSFO": "SFO",
        "KLAX": "LAX",
        "KSEA": "SEA",
        "RKSI": "ICN",
        "RKSS": "GMP",
    },
    "security_wait_minutes_by_airport": {
        "JFK": 30,
        "LGA": 24,
        "EWR": 28,
        "SFO": 22,
        "LAX": 28,
        "SEA": 20,
        "ICN": 25,
    },
    "airport_query_by_code": {
        "JFK": "John F. Kennedy International Airport",
        "LGA": "LaGuardia Airport",
        "EWR": "Newark Liberty International Airport",
        "SFO": "San Francisco International Airport",
        "LAX": "Los Angeles International Airport",
        "SEA": "Seattle-Tacoma International Airport",
        "ICN": "Incheon International Airport",
        "GMP": "Gimpo International Airport",
    },
}

ENGLISH_INTENT_KEYWORDS = [
    "when should i leave",
    "when do i leave",
    "when should i depart",
    "what time should i leave",
    "leave for airport",
    "departure time",
    "when to leave",
]

KOREAN_INTENT_KEYWORDS = [
    "\uc5b8\uc81c\ucd9c\ubc1c\ud574\uc57c\ud574",  # "when should I leave"
    "\uc5b8\uc81c \ucd9c\ubc1c\ud574\uc57c\ud574",
    "\uc5b8\uc81c\ucd9c\ubc1c",
    "\ucd9c\ubc1c\ud574\uc57c\ud574",
    "\uacf5\ud56d \uc5b8\uc81c",
]

KOREAN_AIRPORT_ALIASES = {
    "\uc778\ucc9c": "ICN",
    "\uae40\ud3ec": "GMP",
    "\uae40\ud3ec\uacf5\ud56d": "GMP",
    "\uc778\ucc9c\uacf5\ud56d": "ICN",
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        config = _deep_merge(config, user)
    return config


def detect_departure_intent(query: str) -> bool:
    if not query:
        return False

    lower = query.lower().strip()
    compact = re.sub(r"\s+", "", lower)

    for keyword in ENGLISH_INTENT_KEYWORDS:
        if keyword in lower:
            return True

    for keyword in KOREAN_INTENT_KEYWORDS:
        if keyword in lower or keyword in compact:
            return True

    return False


def _get_now_local(tz_name: str) -> datetime:
    if ZoneInfo is None:
        return datetime.now()
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:
        return datetime.now()


def _parse_departure_time(value: str, now_local: datetime) -> Optional[datetime]:
    if not value:
        return None

    text = value.strip()
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y/%m/%d %H:%M",
        "%H:%M",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%H:%M":
                parsed = now_local.replace(
                    hour=parsed.hour,
                    minute=parsed.minute,
                    second=0,
                    microsecond=0,
                )
                if parsed < now_local:
                    parsed += timedelta(days=1)
            else:
                if now_local.tzinfo is not None:
                    parsed = parsed.replace(tzinfo=now_local.tzinfo)
            return parsed
        except ValueError:
            continue

    # No-year ticket formats -> assume current year
    current_year = now_local.year
    no_year_candidates = [
        (f"{current_year}-{text}", "%Y-%m-%d %H:%M"),
        (f"{current_year}-{text}", "%Y-%m-%dT%H:%M"),
        (f"{current_year}/{text}", "%Y/%m/%d %H:%M"),
        (f"{current_year}/{text}", "%Y/%m/%dT%H:%M"),
        (f"{current_year} {text}", "%Y %b %d %H:%M"),
        (f"{current_year} {text}", "%Y %B %d %H:%M"),
    ]

    for candidate, fmt in no_year_candidates:
        try:
            parsed = datetime.strptime(candidate, fmt)
            if now_local.tzinfo is not None:
                parsed = parsed.replace(tzinfo=now_local.tzinfo)
            return parsed
        except ValueError:
            continue

    return None


def _extract_time_from_query(query: str, now_local: datetime) -> Optional[datetime]:
    # English style: 07:30 or 7:30
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", query)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            dt = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if dt < now_local:
                dt += timedelta(days=1)
            return dt

    # Korean style examples with hour/minute words
    k_match = re.search(r"\b(\d{1,2})\s*(?:\uC2DC)\s*(\d{0,2})\s*(?:\uBD84)?", query)
    if k_match:
        hour = int(k_match.group(1))
        minute_str = k_match.group(2).strip()
        minute = int(minute_str) if minute_str else 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            dt = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if dt < now_local:
                dt += timedelta(days=1)
            return dt

    return None


def _extract_airport_from_route(route: str) -> Optional[str]:
    if not route:
        return None
    codes = re.findall(r"\b([A-Z]{3,4})\b", route.upper())
    if not codes:
        return None
    return codes[0]


def _extract_airport_from_query(query: str, known_codes: Optional[set] = None) -> Optional[str]:
    if not query:
        return None

    stopwords = {
        "WHEN",
        "WHAT",
        "TIME",
        "LEAVE",
        "DEPART",
        "WITH",
        "FROM",
        "FOR",
        "AIRPORT",
        "SHOULD",
        "NEED",
        "GO",
        "NOW",
        "TODAY",
        "TOMORROW",
    }

    upper_codes = re.findall(r"\b([A-Z]{3,4})\b", query.upper())
    if upper_codes:
        for code in upper_codes:
            if code in stopwords:
                continue
            if known_codes and code in known_codes:
                return code

    tagged = re.search(r"\b([A-Z]{3,4})\s*(?:AIRPORT|APT)\b", query.upper())
    if tagged:
        code = tagged.group(1)
        if code not in stopwords:
            return code

    lower = query.lower()
    for korean_name, code in KOREAN_AIRPORT_ALIASES.items():
        if korean_name in lower:
            return code

    return None


def _normalize_airport_code(raw_code: str, config: Dict[str, Any], known_codes: set) -> str:
    if not raw_code:
        return ""

    code = raw_code.strip().upper()
    aliases = config.get("airport_code_aliases", {}) or {}
    if code in aliases:
        code = str(aliases[code]).upper()

    # Common ICAO->IATA conversion for US/Canada when we already know the 3-letter code.
    if len(code) == 4 and code.startswith("K") and code[1:] in known_codes:
        code = code[1:]
    elif len(code) == 4 and code.startswith("C") and code[1:] in known_codes:
        code = code[1:]

    return code


def _extract_flight_number_from_query(query: str) -> Optional[str]:
    if not query:
        return None

    upper = query.upper()
    patterns = [
        r"\b([A-Z]{2,3})\s*[-]?\s*(\d{1,4})\b",
        r"\b([A-Z]{2}\d{1,4})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            if len(match.groups()) == 2:
                return f"{match.group(1)}{match.group(2)}"
            return match.group(1)
    return None


def fetch_aviationstack_position(
    flight_number: str,
    api_key: str,
    timeout_seconds: int,
) -> Optional[Dict[str, Any]]:
    if not flight_number or not api_key or requests is None:
        return None

    normalized = re.sub(r"[^A-Z0-9]", "", flight_number.upper())
    if not normalized:
        return None

    urls = [
        "https://api.aviationstack.com/v1/flights",
        "http://api.aviationstack.com/v1/flights",
    ]
    params = {
        "access_key": api_key,
        "flight_iata": normalized,
        "limit": 10,
    }

    for url in urls:
        try:
            response = requests.get(url, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("data", [])
            if not rows:
                continue

            for row in rows:
                live = row.get("live") or {}
                lat = live.get("latitude")
                lon = live.get("longitude")
                if lat is None or lon is None:
                    continue

                altitude = live.get("altitude")
                speed = live.get("speed_horizontal")
                if speed is None:
                    speed = live.get("speed")

                flight_obj = row.get("flight") or {}
                airline_obj = row.get("airline") or {}
                status = row.get("flight_status") or "unknown"

                return {
                    "flight_iata": flight_obj.get("iata") or normalized,
                    "airline_name": airline_obj.get("name") or "",
                    "status": status,
                    "lat": float(lat),
                    "lon": float(lon),
                    "altitude": float(altitude) if altitude is not None else None,
                    "speed": float(speed) if speed is not None else None,
                }
        except Exception:
            continue

    return None


def _traffic_level(duration_min: int, free_flow_min: int) -> str:
    if duration_min <= 0 or free_flow_min <= 0:
        return "unknown"
    ratio = float(duration_min) / float(free_flow_min)
    if ratio >= 1.35:
        return "heavy"
    if ratio >= 1.15:
        return "moderate"
    return "light"


def _parse_duration_seconds(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.endswith("s"):
        try:
            return int(float(value[:-1]))
        except ValueError:
            return None
    return None


def _normalize_travel_mode(travel_mode: str) -> str:
    mode = (travel_mode or "DRIVE").strip().upper()
    if mode not in {"DRIVE", "TRANSIT", "WALK", "BICYCLE"}:
        return "DRIVE"
    return mode


def _backfill_traffic_with_drive_mode(
    mode: str,
    traffic: str,
    origin: str,
    destination: str,
    api_key: str,
    timeout_seconds: int,
) -> str:
    """
    For non-drive ETA modes, estimate traffic level using the same DRIVE logic.
    Keeps ETA mode-specific while avoiding "unknown" traffic badges.
    """
    if mode == "DRIVE" or traffic != "unknown":
        return traffic

    drive_minutes, drive_traffic, _ = fetch_drive_info(
        origin=origin,
        destination=destination,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        travel_mode="DRIVE",
    )
    if drive_minutes > 0 and drive_traffic != "unknown":
        return drive_traffic
    return traffic


def fetch_drive_info_routes_api(
    origin: str,
    destination: str,
    api_key: str,
    timeout_seconds: int,
    travel_mode: str = "DRIVE",
) -> Tuple[int, str, str]:
    if not api_key or requests is None:
        return -1, "unknown", "fallback"

    mode = _normalize_travel_mode(travel_mode)
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "routes.duration,"
            "routes.staticDuration,"
            "routes.legs.duration,"
            "routes.legs.staticDuration"
        ),
        "Content-Type": "application/json",
    }
    payload = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": mode,
        "computeAlternativeRoutes": False,
    }
    if mode == "DRIVE":
        # Routes API allows routingPreference only for DRIVE/TWO_WHEELER.
        payload["routingPreference"] = "TRAFFIC_AWARE"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        routes = data.get("routes", [])
        if not routes:
            return -1, "unknown", "fallback"

        route = routes[0]
        duration_sec = _parse_duration_seconds(route.get("duration"))
        static_sec = _parse_duration_seconds(route.get("staticDuration"))

        if duration_sec is None:
            legs = route.get("legs", [])
            if legs:
                duration_sec = _parse_duration_seconds(legs[0].get("duration"))
                static_sec = _parse_duration_seconds(legs[0].get("staticDuration"))

        if duration_sec is None:
            return -1, "unknown", "fallback"

        drive_minutes = max(1, int(round(duration_sec / 60.0)))
        free_flow_minutes = max(1, int(round((static_sec or duration_sec) / 60.0)))
        traffic = _traffic_level(drive_minutes, free_flow_minutes) if mode == "DRIVE" else "unknown"
        return drive_minutes, traffic, "google_routes_api"
    except Exception:
        return -1, "unknown", "fallback"


def fetch_drive_info(
    origin: str,
    destination: str,
    api_key: str,
    timeout_seconds: int,
    travel_mode: str = "DRIVE",
) -> Tuple[int, str, str]:
    """
    Returns (drive_minutes, traffic_level, source)
    source in {"google_routes_api", "google_directions", "fallback"}
    """
    if not api_key or requests is None:
        return -1, "unknown", "fallback"

    mode = _normalize_travel_mode(travel_mode)

    # Prefer Routes API (new). Fall back to Directions API (legacy) for compatibility.
    drive_minutes, traffic, source = fetch_drive_info_routes_api(
        origin=origin,
        destination=destination,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        travel_mode=mode,
    )
    if drive_minutes > 0:
        traffic = _backfill_traffic_with_drive_mode(
            mode=mode,
            traffic=traffic,
            origin=origin,
            destination=destination,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return drive_minutes, traffic, source

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "mode": _GMAPS_TRAVEL_MODE.get(mode, "driving"),
        "key": api_key,
    }
    if mode in {"DRIVE", "TRANSIT"}:
        params["departure_time"] = "now"
    if mode == "DRIVE":
        params["traffic_model"] = "best_guess"

    try:
        response = requests.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "OK":
            return -1, "unknown", "fallback"

        routes = payload.get("routes", [])
        if not routes:
            return -1, "unknown", "fallback"

        legs = routes[0].get("legs", [])
        if not legs:
            return -1, "unknown", "fallback"

        leg = legs[0]
        duration = leg.get("duration", {}).get("value")
        effective_duration = duration
        if mode == "DRIVE":
            effective_duration = leg.get("duration_in_traffic", {}).get("value", duration)
        if not effective_duration:
            return -1, "unknown", "fallback"

        drive_minutes = max(1, int(round(effective_duration / 60.0)))
        free_flow_minutes = max(1, int(round((duration or effective_duration) / 60.0)))
        traffic = _traffic_level(drive_minutes, free_flow_minutes) if mode == "DRIVE" else "unknown"
        traffic = _backfill_traffic_with_drive_mode(
            mode=mode,
            traffic=traffic,
            origin=origin,
            destination=destination,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        return drive_minutes, traffic, "google_directions"
    except Exception:
        return -1, "unknown", "fallback"


_GMAPS_TRAVEL_MODE: dict = {
    "DRIVE": "driving",
    "TRANSIT": "transit",
    "WALK": "walking",
    "BICYCLE": "bicycling",
}


def build_maps_url(origin: str, destination: str, travel_mode: str = "DRIVE") -> str:
    gmode = _GMAPS_TRAVEL_MODE.get(travel_mode, "driving")
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={quote_plus(origin)}"
        f"&destination={quote_plus(destination)}"
        f"&travelmode={gmode}"
        "&hl=en"
        "&gl=us"
    )


def format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def emit(key: str, value: Any) -> None:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    print(f"{key}={text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a departure brief")
    parser.add_argument("--query", type=str, default="", help="User query text")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--origin", type=str, default=None, help="Origin address")
    parser.add_argument("--airport-code", type=str, default=None, help="Airport IATA/ICAO code")
    parser.add_argument("--flight-departure-local", type=str, default=None, help="Flight departure local time")
    parser.add_argument("--route", type=str, default=None, help="Route hint")
    parser.add_argument("--flight-number", type=str, default=None, help="Hooked flight number")
    parser.add_argument("--flight-lat", type=float, default=None, help="Hooked flight latitude")
    parser.add_argument("--flight-lon", type=float, default=None, help="Hooked flight longitude")
    parser.add_argument("--flight-alt", type=float, default=None, help="Hooked flight altitude")
    parser.add_argument("--flight-speed", type=float, default=None, help="Hooked flight speed")
    parser.add_argument("--force-intent", action="store_true", help="Skip intent detection")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    print("--- DEPARTURE_BRIEF ---")

    query = (args.query or "").strip()
    intent = args.force_intent or detect_departure_intent(query)
    emit("INTENT", 1 if intent else 0)

    if not intent:
        emit("NOTES", "Not a departure-intent query")
        print("--- END_DEPARTURE_BRIEF ---")
        return 0

    tz_name = config.get("timezone", "America/New_York")
    now_local = _get_now_local(tz_name)

    origin = (args.origin or config.get("default_origin", "")).strip()
    airport_query_map = config.get("airport_query_by_code", {})
    known_codes = set(airport_query_map.keys()) | set(config.get("security_wait_minutes_by_airport", {}).keys())

    airport_code = ""
    if args.airport_code:
        airport_code = args.airport_code.strip().upper()
    elif args.route:
        airport_code = (_extract_airport_from_route(args.route) or "").upper()
    elif query:
        airport_code = (_extract_airport_from_query(query, known_codes=known_codes) or "").upper()
    if not airport_code:
        airport_code = str(config.get("default_airport", "JFK")).upper()
    airport_code = _normalize_airport_code(airport_code, config, known_codes)

    destination = airport_query_map.get(airport_code, f"{airport_code} airport")

    departure_dt = None
    source = "default_lead"

    if args.flight_departure_local:
        departure_dt = _parse_departure_time(args.flight_departure_local, now_local)
        if departure_dt:
            source = "arg_departure_time"

    if departure_dt is None and query:
        departure_dt = _extract_time_from_query(query, now_local)
        if departure_dt:
            source = "query_time"

    if departure_dt is None:
        lead = int(config.get("default_flight_lead_minutes", 180))
        departure_dt = now_local + timedelta(minutes=lead)

    security_by_airport = config.get("security_wait_minutes_by_airport", {})
    security_minutes = int(security_by_airport.get(airport_code, config.get("default_security_wait_minutes", 30)))
    checkin_minutes = int(config.get("default_checkin_minutes", 45))
    walk_minutes = int(config.get("default_walk_to_gate_minutes", 15))
    buffer_minutes = int(config.get("default_buffer_minutes", 10))

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        api_key = str(config.get("google_maps_api_key", "")).strip()

    timeout_seconds = int(config.get("traffic_request_timeout_seconds", 6))
    drive_minutes, traffic_level, drive_source = fetch_drive_info(
        origin=origin,
        destination=destination,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    if drive_minutes <= 0:
        drive_minutes = int(config.get("fallback_drive_minutes", 55))

    airport_arrival_dt = departure_dt - timedelta(
        minutes=(security_minutes + checkin_minutes + walk_minutes + buffer_minutes)
    )
    leave_dt = airport_arrival_dt - timedelta(minutes=drive_minutes)

    flight_number = (args.flight_number or "").strip().upper()
    if not flight_number:
        extracted = _extract_flight_number_from_query(query)
        if extracted:
            flight_number = extracted

    aviationstack_key = os.environ.get("AVIATIONSTACK_API_KEY", "").strip()
    if not aviationstack_key:
        aviationstack_key = str(config.get("aviationstack_api_key", "")).strip()
    aviation_timeout = int(config.get("aviationstack_timeout_seconds", 6))

    flight_position = "N/A"
    flight_position_source = "none"
    if flight_number and args.flight_lat is not None and args.flight_lon is not None:
        alt = f" alt {args.flight_alt:.0f}ft" if args.flight_alt is not None else ""
        speed = f" speed {args.flight_speed:.0f}kt" if args.flight_speed is not None else ""
        flight_position = f"{flight_number} @ {args.flight_lat:.4f},{args.flight_lon:.4f}{alt}{speed}"
        flight_position_source = "hooked_adsb"
    elif flight_number:
        live = fetch_aviationstack_position(
            flight_number=flight_number,
            api_key=aviationstack_key,
            timeout_seconds=aviation_timeout,
        )
        if live:
            alt = f" alt {live['altitude']:.0f}ft" if live.get("altitude") is not None else ""
            speed = f" speed {live['speed']:.0f}kt" if live.get("speed") is not None else ""
            status = live.get("status", "unknown")
            flight_position = f"{live['flight_iata']} @ {live['lat']:.4f},{live['lon']:.4f}{alt}{speed} [{status}]"
            flight_position_source = "aviationstack"

    map_url = build_maps_url(origin, destination)

    notes = []
    if drive_source == "fallback":
        notes.append("traffic_eta_fallback")
    if not api_key:
        notes.append("google_maps_api_key_missing")
    if flight_number and flight_position_source == "none":
        notes.append("flight_position_unavailable")
        if not aviationstack_key:
            notes.append("aviationstack_api_key_missing")

    emit("TIMEZONE", tz_name)
    emit("INPUT_QUERY", query)
    emit("AIRPORT_CODE", airport_code)
    emit("ORIGIN", origin)
    emit("DESTINATION", destination)
    emit("DEPARTURE_TIME_SOURCE", source)
    emit("FLIGHT_DEPARTURE_LOCAL", format_dt(departure_dt))
    emit("AIRPORT_ARRIVAL_LOCAL", format_dt(airport_arrival_dt))
    emit("RECOMMENDED_LEAVE_LOCAL", format_dt(leave_dt))
    emit("DRIVE_MINUTES", drive_minutes)
    emit("TRAFFIC_LEVEL", traffic_level)
    emit("SECURITY_MINUTES", security_minutes)
    emit("CHECKIN_MINUTES", checkin_minutes)
    emit("WALK_TO_GATE_MINUTES", walk_minutes)
    emit("BUFFER_MINUTES", buffer_minutes)
    emit("FLIGHT_NUMBER", flight_number or "N/A")
    emit("FLIGHT_POSITION", flight_position)
    emit("FLIGHT_POSITION_SOURCE", flight_position_source)
    emit("MAP_URL", map_url)
    emit("AUTO_OPEN_MAP", 1 if bool(config.get("open_map_automatically", True)) else 0)
    emit("NOTES", "|".join(notes) if notes else "ok")

    print("--- END_DEPARTURE_BRIEF ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
