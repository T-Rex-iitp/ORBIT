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
    # 실시간 flight count 수신 (RUI 연동)
    # ──────────────────────────────────────
    def get_realtime_count_from_tracker(
        self,
        host: str = "127.0.0.1",
        port: int = 30003,
        collect_seconds: int = 10,
    ) -> int:
        """
        ADS-B tracker(SBS feed)에 접속하여 일정 시간 동안 고유 ICAO 수를 세어
        실시간 flight count를 반환한다.

        이 방법은 RUI가 직접 count를 전달하지 못할 때 대안으로 사용.
        RUI에서 직접 count를 받는 것이 더 정확하고 권장됨.

        Args:
            host: SBS feed 호스트 (기본 localhost)
            port: SBS feed 포트 (기본 30003 = SBS BaseStation)
            collect_seconds: 수집 시간(초). 길수록 정확하나 대기 시간 증가.

        Returns:
            고유 항공기 수 (flight count)
        """
        icao_set: set = set()

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(collect_seconds + 5)
            sock.connect((host, port))

            buf = ""
            start = time.time()
            while time.time() - start < collect_seconds:
                try:
                    data = sock.recv(4096).decode("ascii", errors="ignore")
                except socket.timeout:
                    break
                if not data:
                    break
                buf += data
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    parts = line.strip().split(",")
                    # SBS format: MSG,type,...,hex_ident,...
                    if len(parts) >= 5 and parts[0] == "MSG":
                        hex_ident = parts[4].strip()
                        if hex_ident:
                            icao_set.add(hex_ident)

        except (socket.error, OSError) as e:
            print(f"   ⚠️ ADS-B tracker 연결 실패 ({host}:{port}): {e}")
        finally:
            try:
                sock.close()
            except Exception:
                pass

        return len(icao_set)

    # ──────────────────────────────────────
    # 통합: 실시간 수집 + 혼잡도 판단
    # ──────────────────────────────────────
    def check_realtime_congestion(
        self,
        host: str = "127.0.0.1",
        port: int = 30003,
        collect_seconds: int = 10,
        hour: Optional[int] = None,
    ) -> Dict:
        """
        ADS-B tracker에 직접 접속하여 실시간 flight count를 수집한 뒤
        혼잡도를 판단한다.

        RUI에서 count를 직접 전달하지 못할 때 사용.

        Returns:
            check_congestion()과 동일한 형식의 dict
        """
        print(f"   📡 ADS-B tracker ({host}:{port})에서 {collect_seconds}초간 수집 중...")
        count = self.get_realtime_count_from_tracker(host, port, collect_seconds)
        print(f"   📊 수집된 항공기 수: {count}")
        return self.check_congestion(count, hour=hour)

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


# ──────────────────────────────────────────────
# CLI 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

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
        "--host", type=str, default="127.0.0.1",
        help="ADS-B tracker 호스트 (기본: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=30003,
        help="ADS-B tracker 포트 (기본: 30003)"
    )
    parser.add_argument(
        "--collect", type=int, default=10,
        help="ADS-B 수집 시간(초) (기본: 10)"
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
    else:
        # ADS-B tracker에서 실시간 수집
        result = checker.check_realtime_congestion(
            host=args.host, port=args.port,
            collect_seconds=args.collect, hour=args.hour
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
