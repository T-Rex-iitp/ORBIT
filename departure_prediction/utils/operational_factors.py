"""
Operational-context adjustment factors.

Use cases:
1) Estimate congestion within a 50-mile radius of JFK
2) Reflect delay propagation from the previous leg of the same aircraft (or flight number)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import requests


AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"

# Major commercial/operational airports within 50 miles of JFK
JFK_50_MILE_AIRPORTS: List[str] = [
    "JFK",  # John F. Kennedy
    "LGA",  # LaGuardia
    "EWR",  # Newark Liberty
    "HPN",  # Westchester County
    "ISP",  # Long Island MacArthur
    "SWF",  # Stewart
    "TEB",  # Teterboro
]


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an AviationStack ISO datetime string (returns None on failure)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_delay_minutes(raw_delay: Optional[object]) -> int:
    """Normalize delay value to integer minutes."""
    if raw_delay is None:
        return 0
    try:
        return int(raw_delay)
    except Exception:
        return 0


class OperationalFactorsAnalyzer:
    """Operational context analyzer (congestion + previous-leg delay)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("AVIATIONSTACK_API_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _fetch_flights(self, **params) -> List[Dict]:
        if not self.api_key:
            return []

        query = {"access_key": self.api_key, **params}
        resp = requests.get(AVIATIONSTACK_URL, params=query, timeout=12)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", []) if isinstance(data, dict) else []

    def get_jfk_area_congestion(self, reference_time: datetime) -> Dict:
        """
        Compute congestion score using delay ratio and average delay
        across airports within 50 miles of JFK.

        Returns:
            {
                'score': float(0~1),
                'level': str(low|medium|high),
                'recommended_extra_delay': int,
                'sample_size': int,
                'delayed_ratio': float,
                'avg_delay_minutes': float,
                'airports': list,
            }
        """
        if not self.api_key:
            return {
                "score": 0.0,
                "level": "unknown",
                "recommended_extra_delay": 0,
                "sample_size": 0,
                "delayed_ratio": 0.0,
                "avg_delay_minutes": 0.0,
                "airports": JFK_50_MILE_AIRPORTS,
                "fallback_used": True,
            }

        delays: List[int] = []
        delayed_count = 0

        # Consider free-tier limits: sample 10 records per airport
        for airport in JFK_50_MILE_AIRPORTS:
            try:
                flights = self._fetch_flights(dep_iata=airport, limit=10)
            except Exception:
                continue

            for flight in flights:
                dep = flight.get("departure", {})
                scheduled_dt = _parse_iso_datetime(dep.get("scheduled"))

                # Include only departures within +/-3 hours of the reference time
                if scheduled_dt:
                    gap_hours = abs((scheduled_dt - reference_time).total_seconds()) / 3600
                    if gap_hours > 3:
                        continue

                delay = _safe_delay_minutes(dep.get("delay"))
                delays.append(delay)
                if delay >= 15:
                    delayed_count += 1

        if not delays:
            return {
                "score": 0.0,
                "level": "unknown",
                "recommended_extra_delay": 0,
                "sample_size": 0,
                "delayed_ratio": 0.0,
                "avg_delay_minutes": 0.0,
                "airports": JFK_50_MILE_AIRPORTS,
                "fallback_used": True,
            }

        sample_size = len(delays)
        avg_delay = sum(delays) / sample_size
        delayed_ratio = delayed_count / sample_size

        # Congestion score: 70% delay rate + 30% average delay (capped at 30 min)
        normalized_avg = min(avg_delay / 30.0, 1.0)
        score = 0.7 * delayed_ratio + 0.3 * normalized_avg

        if score >= 0.65:
            level = "high"
            extra = 20
        elif score >= 0.35:
            level = "medium"
            extra = 10
        else:
            level = "low"
            extra = 0

        return {
            "score": round(score, 3),
            "level": level,
            "recommended_extra_delay": extra,
            "sample_size": sample_size,
            "delayed_ratio": round(delayed_ratio, 3),
            "avg_delay_minutes": round(avg_delay, 1),
            "airports": JFK_50_MILE_AIRPORTS,
        }

    def get_previous_leg_delay(self, flight_number: str, scheduled_time: datetime) -> Dict:
        """
        Estimate previous-leg delay (turnaround risk) for the target flight.
        Priority:
          1) Previous leg by same aircraft registration (aircraft.registration)
          2) Most recent departure in same flight-number history
        """
        if not self.api_key:
            return {
                "found": False,
                "delay_minutes": 0,
                "source": "unavailable",
                "fallback_used": True,
            }

        try:
            records = self._fetch_flights(flight_iata=flight_number.upper(), limit=20)
        except Exception:
            return {
                "found": False,
                "delay_minutes": 0,
                "source": "api_error",
                "fallback_used": True,
            }

        if not records:
            return {
                "found": False,
                "delay_minutes": 0,
                "source": "no_records",
            }

        enriched = []
        for rec in records:
            dep = rec.get("departure", {})
            sched = _parse_iso_datetime(dep.get("scheduled"))
            if not sched:
                continue
            enriched.append(
                {
                    "scheduled": sched,
                    "delay": _safe_delay_minutes(dep.get("delay")),
                    "aircraft_registration": rec.get("aircraft", {}).get("registration"),
                    "origin": dep.get("iata"),
                    "dest": rec.get("arrival", {}).get("iata"),
                    "flight_date": rec.get("flight_date"),
                }
            )

        if not enriched:
            return {
                "found": False,
                "delay_minutes": 0,
                "source": "invalid_records",
            }

        # Candidate records must be before current booking time
        candidates = [x for x in enriched if x["scheduled"] < scheduled_time]
        if not candidates:
            return {
                "found": False,
                "delay_minutes": 0,
                "source": "no_previous_leg",
            }

        candidates.sort(key=lambda x: x["scheduled"], reverse=True)
        best = candidates[0]

        # Propagate delay: transfer 50% of prior-leg delay (max 30 min)
        propagated_delay = max(0, min(int(best["delay"] * 0.5), 30))

        return {
            "found": True,
            "delay_minutes": int(best["delay"]),
            "propagated_delay": propagated_delay,
            "source": "same_flight_history",
            "previous_scheduled": best["scheduled"].isoformat(),
            "route": f"{best.get('origin', 'N/A')}-{best.get('dest', 'N/A')}",
            "aircraft_registration": best.get("aircraft_registration"),
        }
