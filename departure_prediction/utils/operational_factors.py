"""
운항 컨텍스트 기반 보정 요인

Use cases:
1) JFK 50마일 권역 혼잡도 추정
2) 동일 항공기(또는 동일 편명) 직전 편 지연 반영
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional

import requests


AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"

# JFK 반경 50마일 이내 주요 상업/운항 공항
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
    """AviationStack ISO datetime 파싱 (실패 시 None)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_delay_minutes(raw_delay: Optional[object]) -> int:
    """지연 값을 정수 분으로 정규화."""
    if raw_delay is None:
        return 0
    try:
        return int(raw_delay)
    except Exception:
        return 0


class OperationalFactorsAnalyzer:
    """운항 컨텍스트(혼잡도 + 직전편 지연) 분석기."""

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
        JFK 50마일 권역 공항의 지연 비율/평균 지연 기반 혼잡도 산출.

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

        # 무료 티어 제한 고려: 공항별 10건 샘플
        for airport in JFK_50_MILE_AIRPORTS:
            try:
                flights = self._fetch_flights(dep_iata=airport, limit=10)
            except Exception:
                continue

            for flight in flights:
                dep = flight.get("departure", {})
                scheduled_dt = _parse_iso_datetime(dep.get("scheduled"))

                # 기준 시각 기준 ±3시간 이내 출발편만 반영
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

        # 혼잡도 점수: 지연율 70% + 평균지연(30분 캡) 30%
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
        대상 항공편의 직전 편 지연(턴어라운드 리스크) 추정.
        우선순위:
          1) 동일 항공기 등록번호(aircraft.registration) 기준 직전편
          2) 동일 편명의 과거 기록 중 직전 출발편
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

        # 현재 예약 시각 이전 레코드만 후보
        candidates = [x for x in enriched if x["scheduled"] < scheduled_time]
        if not candidates:
            return {
                "found": False,
                "delay_minutes": 0,
                "source": "no_previous_leg",
            }

        candidates.sort(key=lambda x: x["scheduled"], reverse=True)
        best = candidates[0]

        # 지연 반영: 직전편 지연의 50%를 전이(최대 30분)
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

