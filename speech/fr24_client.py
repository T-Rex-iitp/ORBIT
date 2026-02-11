#!/usr/bin/env python3
"""
FlightRadar24 API Client
Gets previous flight information for a given flight number
"""

import sys
import json
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

BASE = "https://fr24api.flightradar24.com/api"

# API Token - Replace with your own or load from config
DEFAULT_API_TOKEN = "your_api_token_here"


def _parse_utc(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fr24_get(api_token: str, path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    headers = {
        "Accept": "application/json",
        "Accept-Version": "v1",
        "Authorization": f"Bearer {api_token}",
    }
    r = requests.get(f"{BASE}{path}", headers=headers, params=params, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    j = r.json()
    data = j.get("data", j)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response: {j}")
    return data


def get_latest_leg_by_flightno_now(api_token: str, flight_no: str, window_hours: int = 36) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    rows = fr24_get(api_token, "/flight-summary/light", {
        "flight_datetime_from": _iso_z(now - timedelta(hours=window_hours)),
        "flight_datetime_to": _iso_z(now),
        "flights": flight_no,
        "sort": "desc",
    })
    if not rows:
        raise RuntimeError(f"No flight found for {flight_no} in the last {window_hours} hours")
    return rows[0]


def previous_leg_from_reg(api_token: str, reg: str, cutoff_utc: str, lookback_days: int = 10) -> Optional[Dict[str, Any]]:
    cutoff = _parse_utc(cutoff_utc)
    rows = fr24_get(api_token, "/flight-summary/light", {
        "flight_datetime_from": _iso_z(cutoff - timedelta(days=lookback_days)),
        "flight_datetime_to": _iso_z(cutoff - timedelta(seconds=1)),
        "registrations": reg,
        "sort": "desc",
    })
    for r in rows:
        ended = (r.get("datetime_landed") is not None) or (str(r.get("flight_ended")).lower() == "true")
        if ended:
            return r
    return rows[0] if rows else None


def get_flight_history(api_token: str, reg: str, cutoff_utc: str, max_legs: int = 5, lookback_days: int = 3) -> List[Dict[str, Any]]:
    """
    Get multiple previous flights for a registration (all legs, no status filter).
    
    Args:
        api_token: FlightRadar24 API token
        reg: Aircraft registration
        cutoff_utc: Cutoff time (get flights before this time)
        max_legs: Maximum number of previous legs to retrieve
        lookback_days: How many days to look back
    
    Returns:
        List of flight dictionaries, most recent first (includes in-flight legs)
    """
    cutoff = _parse_utc(cutoff_utc)
    rows = fr24_get(api_token, "/flight-summary/light", {
        "flight_datetime_from": _iso_z(cutoff - timedelta(days=lookback_days)),
        "flight_datetime_to": _iso_z(cutoff - timedelta(seconds=1)),
        "registrations": reg,
        "sort": "desc",
    })
    
    return rows[:max_legs]


def is_flight_in_air(flight: Dict[str, Any]) -> bool:
    """Determine whether a flight leg is currently in the air.
    
    Returns True  if the aircraft is still airborne (no landing time, not ended).
    Returns False if the aircraft has landed or the flight has ended.
    """
    if flight.get("datetime_landed") is not None:
        return False
    if str(flight.get("flight_ended", "")).lower() == "true":
        return False
    # Has a takeoff time but no landing → in the air
    if flight.get("datetime_takeoff") is not None or flight.get("first_seen") is not None:
        return True
    return False


def build_route_chain(flights: List[Dict[str, Any]]) -> str:
    """
    Build a route chain string showing only the last round-trip.
    E.g., "JFK -> BUF -> JFK" for a round trip.
    
    Args:
        flights: List of flight dictionaries (most recent first)
    
    Returns:
        Route chain string (last round-trip only)
    """
    if not flights:
        return ""
    
    # Get the most recent flight (previous flight)
    most_recent = flights[0]
    recent_origin = icao_to_iata(most_recent.get("orig_icao")) or "?"
    recent_dest = icao_to_iata(most_recent.get("dest_icao")) or "?"
    
    # Check if it's a round-trip: look for a flight that started from recent_dest
    # This means: A -> B (older) then B -> A (recent) = round trip A -> B -> A
    for flight in flights[1:]:
        origin = icao_to_iata(flight.get("orig_icao")) or "?"
        dest = icao_to_iata(flight.get("dest_icao")) or "?"
        
        # Check if this flight's destination matches recent flight's origin
        # AND this flight's origin matches recent flight's destination
        # That means: origin -> dest (this flight) -> recent_dest (round trip back)
        if dest == recent_origin and origin == recent_dest:
            # Found round-trip: recent_dest -> recent_origin -> recent_dest
            return f"{recent_dest} -> {recent_origin} -> {recent_dest}"
    
    # No round-trip found, just show the most recent leg
    return f"{recent_origin} -> {recent_dest}"


def icao_to_iata(icao_code: str) -> str:
    """
    Convert ICAO airport code to IATA format.
    For US airports, removes the leading 'K' (e.g., KJFK -> JFK).
    For other airports, returns as-is or handles common cases.
    """
    if not icao_code:
        return icao_code
    
    icao_code = icao_code.upper().strip()
    
    # US airports: 4 letters starting with K -> remove K
    if len(icao_code) == 4 and icao_code.startswith('K'):
        return icao_code[1:]  # KJFK -> JFK
    
    # Canadian airports: start with C (CYYZ -> YYZ)
    if len(icao_code) == 4 and icao_code.startswith('C') and icao_code[1] == 'Y':
        return icao_code[1:]  # CYYZ -> YYZ
    
    # For other 4-letter ICAO codes, return as-is
    # (e.g., EGLL for London Heathrow, RKSI for Incheon)
    return icao_code


def get_previous_flight(api_token: str, flight_no: str, max_history: int = 5) -> Dict[str, Any]:
    """
    Get the previous flight for a given flight number, including route history
    and flight status (in-air vs landed).
    
    Args:
        api_token: FlightRadar24 API token
        flight_no: Flight number (e.g., "KE937", "AAL123")
        max_history: Maximum number of previous legs to include in history
    
    Returns:
        Dictionary with current flight, previous flight info (with status), and route history
    """
    mine = get_latest_leg_by_flightno_now(api_token, flight_no)
    
    cutoff = mine.get("first_seen") or mine.get("datetime_takeoff")
    if not cutoff:
        raise RuntimeError("Could not get first_seen/datetime_takeoff from current flight")
    
    # Get flight history (all legs, no ended filter)
    history = get_flight_history(api_token, reg=mine["reg"], cutoff_utc=cutoff, max_legs=max_history)
    
    # Get the immediate previous flight
    prev = history[0] if history else None
    
    # Determine previous flight status
    prev_in_air = is_flight_in_air(prev) if prev else False
    
    # Build route chain from history
    route_chain = build_route_chain(history) if history else ""
    
    return {
        "current_flight": {
            "flight_no": mine.get("flight"),
            "callsign": mine.get("callsign"),
            "registration": mine.get("reg"),
            "origin": icao_to_iata(mine.get("orig_icao")),
            "destination": icao_to_iata(mine.get("dest_icao")),
        },
        "previous_flight": {
            "flight_no": prev.get("flight") if prev else None,
            "callsign": prev.get("callsign") if prev else None,
            "registration": prev.get("reg") if prev else None,
            "origin": icao_to_iata(prev.get("orig_icao")) if prev else None,
            "destination": icao_to_iata(prev.get("dest_icao")) if prev else None,
            "orig_icao": prev.get("orig_icao") if prev else None,
            "dest_icao": prev.get("dest_icao") if prev else None,
            "in_air": prev_in_air,
            "datetime_takeoff": prev.get("datetime_takeoff") if prev else None,
            "datetime_landed": prev.get("datetime_landed") if prev else None,
        } if prev else None,
        "route_history": route_chain,
        "history_count": len(history)
    }


def main():
    """CLI interface for getting previous flight information"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Get previous flight information from FlightRadar24 API"
    )
    parser.add_argument(
        "flight_no",
        type=str,
        help="Flight number (e.g., KE937, AAL123, UAL456)"
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=DEFAULT_API_TOKEN,
        help="FlightRadar24 API token"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        result = get_previous_flight(args.api_token, args.flight_no)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            current = result["current_flight"]
            prev = result["previous_flight"]
            route_history = result.get("route_history", "")
            
            print(f"Current Flight: {current['flight_no']}")
            print(f"  Route: {current['origin']} -> {current['destination']}")
            
            if prev:
                print(f"\nPrevious Flight: {prev['flight_no']}")
                
                # Show route history if available (round-trip display)
                if route_history:
                    print(f"  Route History: {route_history}")
                else:
                    print(f"  Route: {prev['origin']} -> {prev['destination']}")
                
                # Output the callsign for hooking (this is what the GUI will use)
                print(f"\n--- HOOK_CALLSIGN ---")
                print(prev['callsign'])
            else:
                print("\nNo previous flight found")
                print(f"\n--- HOOK_CALLSIGN ---")
                print("NOT_FOUND")
                
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

