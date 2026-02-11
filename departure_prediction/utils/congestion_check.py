"""
JFK 공역 혼잡도 판단 모듈

과거 실측 flight count 데이터(CSV)를 기반으로 시간대별 평균/표준편차를 계산하고,
RUI(또는 ADS-B tracker)로부터 실시간 flight count를 받아 현재 혼잡도를 판단한다.

사용 흐름:
  1. RUI에서 JFK 버튼 → 실시간 flight count 획득
  2. 이 모듈의 check_congestion(count, hour) 호출
  3. 과거 시간대별 평균과 비교 → congestion level 반환
  4. hybrid_predictor.py 에서 지연 보정에 활용

출력 형식은 기존 operational_factors.py / hybrid_predictor.py 의
congestion_info dict 포맷과 호환된다.
"""

from __future__ import annotations

import csv
import math
import os
import socket
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────
# 과거 JFK 인근 Flight Count 시간대별 통계
# (JFK_FlightCount_20260115_101513.csv 에서 사전 계산)
#
# key: hour (0-23)
# value: (mean, std)
# ──────────────────────────────────────────────
_DEFAULT_HOURLY_STATS: Dict[int, Tuple[float, float]] = {
    0:  (13.14,  2.27),
    1:  ( 6.40,  1.35),
    2:  ( 6.52,  1.60),
    3:  ( 7.52,  1.31),
    4:  ( 7.84,  3.15),
    5:  (15.21,  3.87),
    6:  (38.04,  6.15),
    7:  (59.99, 10.87),
    8:  (68.59,  5.51),
    9:  (71.45,  5.65),   # 보간값 (8시·10시 평균)
    10: (74.31,  5.79),
    11: (76.01,  7.72),
    12: (72.83,  6.48),
    13: (80.81,  4.53),
    14: (91.85,  4.62),
    15: (98.73,  8.61),
    16: (92.25,  8.27),
    17: (90.86,  4.36),
    18: (93.72,  6.61),
    19: (80.71,  6.75),
    20: (74.74,  4.83),
    21: (63.22,  3.93),
    22: (53.36,  8.36),
    23: (32.60, 12.31),
}


class JFKCongestionChecker:
    """
    JFK 인근 공역 혼잡도 판단기.

    - 과거 실측 데이터(CSV 또는 내장 통계)에서 시간대별 평균·표준편차를 계산
    - 실시간 flight count와 비교하여 z-score 기반 혼잡도 반환
    - ADS-B tracker에 직접 연결하여 실시간 count를 얻는 기능 제공
    """

    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: 과거 flight count CSV 파일 경로.
                      None이면 내장 기본 통계(_DEFAULT_HOURLY_STATS)를 사용.
                      CSV 컬럼: timestamp, datetime, flight_count, elapsed_hours
        """
        if csv_path and os.path.isfile(csv_path):
            self.hourly_stats = self._load_stats_from_csv(csv_path)
            self._csv_loaded = True
        else:
            self.hourly_stats = dict(_DEFAULT_HOURLY_STATS)
            self._csv_loaded = False

    # ──────────────────────────────────────
    # CSV에서 시간대별 통계 계산
    # ──────────────────────────────────────
    @staticmethod
    def _load_stats_from_csv(csv_path: str) -> Dict[int, Tuple[float, float]]:
        """CSV 파일에서 시간대별 (mean, std)를 계산한다."""
        hourly_values: Dict[int, List[float]] = {}

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
                    count = float(row["flight_count"])
                except (KeyError, ValueError):
                    continue
                hourly_values.setdefault(dt.hour, []).append(count)

        stats: Dict[int, Tuple[float, float]] = {}
        for hour, values in hourly_values.items():
            n = len(values)
            mean = sum(values) / n
            if n > 1:
                variance = sum((v - mean) ** 2 for v in values) / (n - 1)
                std = math.sqrt(variance)
            else:
                std = 1.0  # 데이터 1개면 기본 std=1
            stats[hour] = (round(mean, 2), round(std, 2))

        # 빠진 시간대가 있으면 기본값으로 보간
        for hour in range(24):
            if hour not in stats:
                stats[hour] = _DEFAULT_HOURLY_STATS.get(hour, (50.0, 10.0))

        return stats

    # ──────────────────────────────────────
    # 혼잡도 판단 (핵심 로직)
    # ──────────────────────────────────────
    def check_congestion(
        self,
        current_flight_count: int,
        hour: Optional[int] = None,
        reference_time: Optional[datetime] = None,
    ) -> Dict:
        """
        현재 flight count를 해당 시간대 과거 평균과 비교하여 혼잡도를 판단한다.

        Args:
            current_flight_count: RUI 또는 ADS-B tracker에서 얻은 실시간 flight count
            hour: 비교 대상 시간대 (0-23). None이면 현재 시각 사용.
            reference_time: 참조 시각 (hour 파라미터가 None일 때 사용)

        Returns:
            hybrid_predictor.py의 congestion_info 포맷과 호환되는 dict:
            {
                'level': 'low' | 'medium' | 'high',
                'score': float (0.0 ~ 1.0),
                'sample_size': int,
                'recommended_extra_delay': int (분),
                'source': 'historical_comparison',
                'details': {
                    'current_count': int,
                    'hour': int,
                    'historical_mean': float,
                    'historical_std': float,
                    'z_score': float,
                    'ratio': float,  # current / mean
                }
            }
        """
        # 시간대 결정
        if hour is None:
            if reference_time is not None:
                hour = reference_time.hour
            else:
                hour = datetime.now().hour

        hour = int(hour) % 24

        mean, std = self.hourly_stats.get(hour, (50.0, 10.0))

        # z-score 계산 (std=0 방지)
        if std < 0.01:
            std = 1.0
        z_score = (current_flight_count - mean) / std

        # 비율 (현재 / 평균)
        ratio = current_flight_count / mean if mean > 0 else 1.0

        # 혼잡도 점수 (0~1 범위, z-score 기반)
        score = min(max(z_score / 3.0, 0.0), 1.0)

        # 혼잡도 레벨 및 권장 추가 지연
        if z_score > 1.5:
            level = "high"
            extra_delay = 20
        elif z_score > 0.5:
            level = "medium"
            extra_delay = 10
        else:
            level = "low"
            extra_delay = 0

        return {
            "level": level,
            "score": round(score, 3),
            "sample_size": current_flight_count,
            "recommended_extra_delay": extra_delay,
            "source": "historical_comparison",
            "details": {
                "current_count": current_flight_count,
                "hour": hour,
                "historical_mean": mean,
                "historical_std": std,
                "z_score": round(z_score, 3),
                "ratio": round(ratio, 3),
            },
        }

    # ──────────────────────────────────────
    # 거리 계산 (Haversine) - C++ RUI와 동일
    # ──────────────────────────────────────
    @staticmethod
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        두 좌표 사이의 거리를 statute miles로 계산 (Haversine 공식).
        C++ RUI의 CalculateDistanceMiles()와 동일한 로직.
        """
        EARTH_RADIUS_MILES = 3958.8

        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)

        d_lat = lat2_r - lat1_r
        d_lon = lon2_r - lon1_r

        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(lat1_r) * math.cos(lat2_r) *
             math.sin(d_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return EARTH_RADIUS_MILES * c

    # ──────────────────────────────────────
    # SBS 피드에서 실시간 flight count 수신
    # (RUI의 SBS Connect와 동일한 방식)
    # ──────────────────────────────────────
    def get_realtime_count_from_sbs(
        self,
        host: str = "128.237.96.41",
        port: int = 5002,
        collect_seconds: int = 30,
        center_lat: float = 40.6413,
        center_lon: float = -73.7781,
        radius_miles: float = 50.0,
    ) -> Tuple[int, int, Dict]:
        """
        SBS BaseStation 피드에 접속하여 JFK 반경 내 항공기 수를 실시간 집계.

        RUI의 JFK 버튼 → CountFlightsInRadius()와 동일한 로직을 Python으로 구현.
        SBS 피드 포트는 5002 (C++ RUI의 IdTCPClientSBS->Port=5002와 동일).

        SBS 메시지 포맷:
          MSG,type,sessionID,aircraftID,hexIdent,flightID,
          dateGen,timeGen,dateLog,timeLog,
          callsign,altitude,groundSpeed,track,lat,lon,
          verticalRate,squawk,...

        Args:
            host: SBS 피드 서버 주소 (기본: 128.237.96.41)
            port: SBS 피드 포트 (기본: 5002, RUI와 동일)
            collect_seconds: 수집 시간(초). 충분한 위치 데이터를 받으려면 15-30초 권장.
            center_lat: 중심 위도 (기본: JFK 40.6413)
            center_lon: 중심 경도 (기본: JFK -73.7781)
            radius_miles: 반경 (기본: 50 마일, RUI JFK 버튼과 동일)

        Returns:
            (jfk_count, total_count, aircraft_info) 튜플:
              - jfk_count: JFK 반경 내 항공기 수
              - total_count: 전체 고유 항공기 수
              - aircraft_info: 각 항공기의 상세 정보 dict
        """
        # 항공기별 최신 위치/정보 저장
        aircraft: Dict[str, Dict] = {}

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(collect_seconds + 10)
            sock.connect((host, port))

            buf = ""
            start = time.time()
            while time.time() - start < collect_seconds:
                try:
                    data = sock.recv(4096).decode("utf-8", errors="replace")
                except socket.timeout:
                    break
                if not data:
                    break
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    parts = line.strip().split(",")

                    # SBS format: MSG,type,...,hexIdent(4),...,
                    #   callsign(10),alt(11),gs(12),track(13),lat(14),lon(15),...
                    if len(parts) < 11 or parts[0] != "MSG":
                        continue

                    hex_ident = parts[4].strip()
                    if not hex_ident:
                        continue

                    # 항공기 엔트리 생성/업데이트
                    if hex_ident not in aircraft:
                        aircraft[hex_ident] = {
                            "icao": hex_ident,
                            "callsign": "",
                            "lat": None,
                            "lon": None,
                            "altitude": None,
                            "ground_speed": None,
                        }

                    ac = aircraft[hex_ident]

                    # 콜사인 (필드 10)
                    if len(parts) > 10 and parts[10].strip():
                        ac["callsign"] = parts[10].strip()

                    # 고도 (필드 11)
                    if len(parts) > 11 and parts[11].strip():
                        try:
                            ac["altitude"] = int(float(parts[11].strip()))
                        except ValueError:
                            pass

                    # 속도 (필드 12)
                    if len(parts) > 12 and parts[12].strip():
                        try:
                            ac["ground_speed"] = float(parts[12].strip())
                        except ValueError:
                            pass

                    # 위도/경도 (필드 14, 15) - 핵심: JFK 거리 계산에 필요
                    if len(parts) > 15:
                        lat_s = parts[14].strip()
                        lon_s = parts[15].strip()
                        if lat_s and lon_s:
                            try:
                                ac["lat"] = float(lat_s)
                                ac["lon"] = float(lon_s)
                            except ValueError:
                                pass

        except (socket.error, OSError) as e:
            print(f"   [WARNING] SBS feed connection failed ({host}:{port}): {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        # JFK 반경 내 항공기 카운트 (RUI의 CountFlightsInRadius와 동일)
        jfk_count = 0
        for ac in aircraft.values():
            if ac["lat"] is not None and ac["lon"] is not None:
                dist = self._haversine_miles(
                    center_lat, center_lon, ac["lat"], ac["lon"]
                )
                ac["distance_miles"] = round(dist, 1)
                if dist <= radius_miles:
                    ac["within_jfk_radius"] = True
                    jfk_count += 1
                else:
                    ac["within_jfk_radius"] = False

        return jfk_count, len(aircraft), aircraft

    # ──────────────────────────────────────
    # 기존 호환: 단순 ICAO 카운트 방식
    # ──────────────────────────────────────
    def get_realtime_count_from_tracker(
        self,
        host: str = "128.237.96.41",
        port: int = 5002,
        collect_seconds: int = 30,
    ) -> int:
        """
        SBS 피드에 접속하여 JFK 50마일 반경 내 항공기 수를 반환.
        (RUI JFK 버튼과 동일한 결과)

        Args:
            host: SBS 피드 호스트 (기본: 128.237.96.41)
            port: SBS 피드 포트 (기본: 5002)
            collect_seconds: 수집 시간(초)

        Returns:
            JFK 반경 50마일 내 항공기 수
        """
        jfk_count, total_count, _ = self.get_realtime_count_from_sbs(
            host=host, port=port, collect_seconds=collect_seconds
        )
        return jfk_count

    # ──────────────────────────────────────
    # 통합: SBS 실시간 수집 + 혼잡도 판단
    # ──────────────────────────────────────
    def check_realtime_congestion(
        self,
        host: str = "128.237.96.41",
        port: int = 5002,
        collect_seconds: int = 30,
        hour: Optional[int] = None,
        radius_miles: float = 50.0,
    ) -> Dict:
        """
        SBS 피드에서 실시간 JFK 반경 내 항공기 수를 집계하고 혼잡도를 판단.

        RUI의 SBS Connect (128.237.96.41:5002) → JFK 버튼과 동일한 동작을
        Python에서 자동으로 수행한다.

        Args:
            host: SBS 피드 호스트 (기본: 128.237.96.41)
            port: SBS 피드 포트 (기본: 5002)
            collect_seconds: 수집 시간(초). 30초 권장.
            hour: 비교 대상 시간대 (0-23). None이면 현재 시각.
            radius_miles: JFK 중심 반경 (기본: 50마일)

        Returns:
            check_congestion()과 동일한 형식의 dict + SBS 상세 정보 추가
        """
        print(f"   [SBS] Connecting to {host}:{port} ...")
        print(f"   [SBS] Collecting data for {collect_seconds} seconds ...")

        jfk_count, total_count, aircraft = self.get_realtime_count_from_sbs(
            host=host, port=port, collect_seconds=collect_seconds,
            radius_miles=radius_miles,
        )

        print(f"   [SBS] Total unique aircraft: {total_count}")
        print(f"   [SBS] Aircraft within {radius_miles}mi of JFK: {jfk_count}")

        result = self.check_congestion(jfk_count, hour=hour)

        # SBS 실시간 상세 정보 추가
        result["source"] = "sbs_realtime"
        result["details"]["total_aircraft"] = total_count
        result["details"]["jfk_radius_miles"] = radius_miles
        result["details"]["sbs_host"] = host
        result["details"]["sbs_port"] = port
        result["details"]["collect_seconds"] = collect_seconds

        return result

    # ──────────────────────────────────────
    # RUI에서 JFK 카운트 읽기 (공유 파일)
    # ──────────────────────────────────────
    @staticmethod
    def get_count_from_rui(
        file_path: Optional[str] = None,
        max_age_seconds: float = 300,
    ) -> Optional[Dict]:
        """
        RUI가 JFK 버튼 클릭 시 기록한 공유 파일에서 flight count를 읽는다.

        RUI (C++ ADS-B Display)는 JFK 버튼을 누르면
        departure_prediction/data/jfk_realtime_count.json 에 아래 형식으로 기록:
        {
            "flight_count": 87,
            "airport": "JFK",
            "radius_miles": 50.0,
            "latitude": 40.6413,
            "longitude": -73.7781,
            "timestamp": "2026-02-11 14:30:00",
            "source": "RUI"
        }

        Args:
            file_path: JSON 파일 경로. None이면 기본 경로 사용.
            max_age_seconds: 데이터 유효 시간(초). 기본 300초(5분).
                             이보다 오래된 데이터는 None 반환.

        Returns:
            파일의 JSON 내용을 dict로 반환. 파일이 없거나 유효 기간 초과 시 None.
        """
        if file_path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            file_path = str(data_dir / "jfk_realtime_count.json")

        if not os.path.isfile(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 유효 기간 확인
            if "timestamp" in data and max_age_seconds > 0:
                ts = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
                age = (datetime.now() - ts).total_seconds()
                if age > max_age_seconds:
                    print(f"   [RUI] Data is {age:.0f}s old (max {max_age_seconds}s). Stale.")
                    return None

            return data

        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            print(f"   [RUI] Error reading file: {e}")
            return None

    def check_rui_congestion(
        self,
        file_path: Optional[str] = None,
        max_age_seconds: float = 300,
        hour: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        RUI의 JFK 버튼 결과(공유 파일)에서 flight count를 읽어 혼잡도를 판단.

        사용 흐름:
          1. RUI에서 SBS Connect (128.237.96.41:5002)
          2. RUI에서 JFK 버튼 클릭 → 파일에 count 기록
          3. 이 함수 호출 → 파일에서 count 읽어 혼잡도 분석

        Args:
            file_path: 공유 JSON 파일 경로. None이면 기본 경로.
            max_age_seconds: 데이터 유효 시간(초). 기본 300초(5분).
            hour: 비교 대상 시간대 (0-23). None이면 현재 시각.

        Returns:
            congestion_info dict. 파일이 없거나 데이터가 오래되면 None.
        """
        rui_data = self.get_count_from_rui(
            file_path=file_path, max_age_seconds=max_age_seconds
        )
        if rui_data is None:
            print("   [RUI] No valid data from RUI.")
            print("   [RUI] Make sure RUI is running, SBS connected, and JFK button clicked.")
            return None

        count = rui_data.get("flight_count", 0)
        radius = rui_data.get("radius_miles", 50.0)
        timestamp = rui_data.get("timestamp", "")

        print(f"   [RUI] Got flight count from RUI: {count}")
        print(f"   [RUI] Radius: {radius} miles | Timestamp: {timestamp}")

        result = self.check_congestion(count, hour=hour)

        # RUI 소스 정보 추가
        result["source"] = "rui"
        result["details"]["rui_timestamp"] = timestamp
        result["details"]["jfk_radius_miles"] = radius

        return result

    # ──────────────────────────────────────
    # 시간대별 통계 요약 (디버깅/확인용)
    # ──────────────────────────────────────
    def get_hourly_summary(self) -> str:
        """시간대별 평균/표준편차 요약 테이블을 문자열로 반환."""
        lines = [
            f"{'Hour':>4}  {'Mean':>8}  {'Std':>8}  {'Source'}",
            "-" * 40,
        ]
        src = "CSV" if self._csv_loaded else "Default"
        for hour in range(24):
            mean, std = self.hourly_stats.get(hour, (0.0, 0.0))
            lines.append(f"{hour:>4}  {mean:>8.2f}  {std:>8.2f}  {src}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# 편의 함수 (모듈 레벨)
# ──────────────────────────────────────────────

# 싱글톤 인스턴스 (CSV가 있으면 자동 로드)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_CSV = _DATA_DIR / "jfk_historical_flight_counts.csv"

_checker_instance: Optional[JFKCongestionChecker] = None


def get_checker(csv_path: Optional[str] = None) -> JFKCongestionChecker:
    """싱글톤 JFKCongestionChecker 인스턴스를 반환."""
    global _checker_instance
    if _checker_instance is None:
        path = csv_path or (str(_DEFAULT_CSV) if _DEFAULT_CSV.exists() else None)
        _checker_instance = JFKCongestionChecker(csv_path=path)
    return _checker_instance


def check_jfk_congestion(
    current_flight_count: int,
    hour: Optional[int] = None,
    reference_time: Optional[datetime] = None,
) -> Dict:
    """
    간편 호출 함수: 실시간 flight count로 JFK 혼잡도를 판단.

    Args:
        current_flight_count: 현재 JFK 인근 flight count (RUI에서 전달)
        hour: 비교 대상 시간대 (0-23). None이면 현재 시각.
        reference_time: 참조 시각 (hour 대신 사용 가능)

    Returns:
        congestion_info dict (hybrid_predictor.py 호환)

    Example:
        >>> from utils.congestion_check import check_jfk_congestion
        >>> result = check_jfk_congestion(current_flight_count=105, hour=15)
        >>> print(result['level'])  # 'medium' or 'high'
        >>> print(result['recommended_extra_delay'])  # 10 or 20
    """
    checker = get_checker()
    return checker.check_congestion(
        current_flight_count=current_flight_count,
        hour=hour,
        reference_time=reference_time,
    )


def check_jfk_congestion_from_rui(
    hour: Optional[int] = None,
    max_age_seconds: float = 300,
) -> Optional[Dict]:
    """
    RUI의 JFK 버튼 결과에서 flight count를 읽어 혼잡도를 판단.

    사용 흐름:
      1. RUI에서 SBS Connect → JFK 버튼 클릭
      2. 이 함수 호출 → 파일에서 최신 count 읽어 혼잡도 분석

    Args:
        hour: 비교 대상 시간대 (0-23). None이면 현재 시각.
        max_age_seconds: 데이터 유효 시간(초). 기본 300초(5분).

    Returns:
        congestion_info dict. RUI 데이터가 없으면 None.

    Example:
        >>> from utils.congestion_check import check_jfk_congestion_from_rui
        >>> result = check_jfk_congestion_from_rui()
        >>> if result:
        ...     print(result['level'], result['details']['current_count'])
    """
    checker = get_checker()
    return checker.check_rui_congestion(
        max_age_seconds=max_age_seconds, hour=hour,
    )


# ──────────────────────────────────────────────
# CLI 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import sys
    
    # Fix Windows console encoding
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    parser = argparse.ArgumentParser(description="JFK 공역 혼잡도 판단")
    parser.add_argument(
        "--count", type=int, default=None,
        help="현재 flight count (지정하지 않으면 ADS-B tracker에서 수집)"
    )
    parser.add_argument(
        "--hour", type=int, default=None,
        help="비교 대상 시간대 (0-23, 기본: 현재 시각)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="과거 flight count CSV 파일 경로"
    )
    parser.add_argument(
        "--host", type=str, default="128.237.96.41",
        help="SBS 피드 호스트 (기본: 128.237.96.41)"
    )
    parser.add_argument(
        "--port", type=int, default=5002,
        help="SBS 피드 포트 (기본: 5002, RUI와 동일)"
    )
    parser.add_argument(
        "--collect", type=int, default=3,
        help="SBS 수집 시간(초) (기본: 3)"
    )
    parser.add_argument(
        "--radius", type=float, default=50.0,
        help="JFK 중심 반경 마일 (기본: 50.0, RUI와 동일)"
    )
    parser.add_argument(
        "--from-rui", action="store_true",
        help="RUI JFK 버튼 결과(공유 파일)에서 flight count 읽기"
    )
    parser.add_argument(
        "--max-age", type=float, default=300,
        help="RUI 데이터 유효 시간(초) (기본: 300 = 5분)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="시간대별 통계 요약 출력"
    )
    args = parser.parse_args()

    checker = JFKCongestionChecker(csv_path=args.csv)

    if args.summary:
        print("\n=== JFK 인근 시간대별 Flight Count 통계 ===\n")
        print(checker.get_hourly_summary())
        print()

    if args.count is not None:
        # 직접 count 전달
        result = checker.check_congestion(args.count, hour=args.hour)
    elif args.from_rui:
        # RUI JFK 버튼 결과에서 읽기
        result = checker.check_rui_congestion(
            max_age_seconds=args.max_age, hour=args.hour
        )
        if result is None:
            print("\n[ERROR] No valid data from RUI. Exiting.")
            print("Steps to use --from-rui:")
            print("  1. Open ADS-B Display (RUI)")
            print("  2. Click 'SBS Connect' (connects to 128.237.96.41:5002)")
            print("  3. Click 'JFK' button")
            print("  4. Run this script with --from-rui")
            sys.exit(1)
    else:
        # SBS 피드에서 실시간 수집 (RUI SBS Connect와 동일)
        result = checker.check_realtime_congestion(
            host=args.host, port=args.port,
            collect_seconds=args.collect, hour=args.hour,
            radius_miles=args.radius,
        )

    print(f"\n=== JFK 공역 혼잡도 결과 ===")
    print(f"  현재 Flight Count : {result['details']['current_count']}")
    print(f"  비교 시간대       : {result['details']['hour']}시")
    print(f"  과거 평균         : {result['details']['historical_mean']:.1f}")
    print(f"  과거 표준편차     : {result['details']['historical_std']:.1f}")
    print(f"  Z-Score           : {result['details']['z_score']:.2f}")
    print(f"  비율 (현재/평균)  : {result['details']['ratio']:.2f}")
    print(f"  ---")
    print(f"  혼잡도 레벨       : {result['level'].upper()}")
    print(f"  혼잡도 점수       : {result['score']:.3f}")
    print(f"  권장 추가 지연    : +{result['recommended_extra_delay']}분")
