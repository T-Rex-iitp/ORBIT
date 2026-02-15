"""
JFK airspace congestion evaluation module.

Calculates hourly mean/standard deviation from historical measured
flight-count data (CSV), then evaluates current congestion using
real-time flight count from RUI (or ADS-B tracker).

Typical flow:
  1. Click JFK button in RUI -> obtain real-time flight count
  2. Call check_congestion(count, hour) in this module
  3. Compare against historical hourly mean -> return congestion level
  4. Use result in hybrid_predictor.py for delay adjustment

Output format is compatible with the congestion_info dict format used by
operational_factors.py / hybrid_predictor.py.
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
# Historical hourly Flight Count statistics around JFK
# (precomputed from JFK_FlightCount_20260115_101513.csv)
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
    9:  (71.45,  5.65),   # Interpolated value (average of 08:00 and 10:00)
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
    JFK nearby airspace congestion evaluator.

    - Computes hourly mean/std from historical measured data (CSV or built-in stats)
    - Returns z-score based congestion by comparing with real-time flight count
    - Provides direct ADS-B tracker connectivity for real-time count retrieval
    """

    def __init__(self, csv_path: Optional[str] = None):
        """
        Args:
            csv_path: Path to historical flight-count CSV file.
                      If None, uses built-in default stats (_DEFAULT_HOURLY_STATS).
                      CSV columns: timestamp, datetime, flight_count, elapsed_hours
        """
        if csv_path and os.path.isfile(csv_path):
            self.hourly_stats = self._load_stats_from_csv(csv_path)
            self._csv_loaded = True
        else:
            self.hourly_stats = dict(_DEFAULT_HOURLY_STATS)
            self._csv_loaded = False

    # ──────────────────────────────────────
    # Compute hourly statistics from CSV
    # ──────────────────────────────────────
    @staticmethod
    def _load_stats_from_csv(csv_path: str) -> Dict[int, Tuple[float, float]]:
        """Compute hourly (mean, std) from CSV file."""
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
                std = 1.0  # If only one sample exists, use default std=1
            stats[hour] = (round(mean, 2), round(std, 2))

        # Interpolate missing hours with default values
        for hour in range(24):
            if hour not in stats:
                stats[hour] = _DEFAULT_HOURLY_STATS.get(hour, (50.0, 10.0))

        return stats

    # ──────────────────────────────────────
    # Determine congestion (core logic)
    # ──────────────────────────────────────
    def check_congestion(
        self,
        current_flight_count: int,
        hour: Optional[int] = None,
        reference_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Determine congestion by comparing current flight count with historical
        average for the same hour.

        Args:
            current_flight_count: Real-time flight count from RUI or ADS-B tracker
            hour: Hour bucket to compare (0-23). If None, uses current hour.
            reference_time: Reference time (used when hour is None)

        Returns:
            Dict compatible with hybrid_predictor.py congestion_info format:
            {
                'level': 'low' | 'medium' | 'high',
                'score': float (0.0 ~ 1.0),
                'sample_size': int,
                'recommended_extra_delay': int (minutes),
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
        # Determine hour bucket
        if hour is None:
            if reference_time is not None:
                hour = reference_time.hour
            else:
                hour = datetime.now().hour

        hour = int(hour) % 24

        mean, std = self.hourly_stats.get(hour, (50.0, 10.0))

        # Compute z-score (avoid std=0)
        if std < 0.01:
            std = 1.0
        z_score = (current_flight_count - mean) / std

        # Ratio (current / average)
        ratio = current_flight_count / mean if mean > 0 else 1.0

        # Congestion score (0-1 range, z-score based)
        score = min(max(z_score / 3.0, 0.0), 1.0)

        # Congestion level and recommended extra delay
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
    # Distance calculation (Haversine) - same as C++ RUI
    # ──────────────────────────────────────
    @staticmethod
    def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Compute distance between two coordinates in statute miles
        using Haversine formula.
        Same logic as C++ RUI CalculateDistanceMiles().
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
    # Receive real-time flight count from SBS feed
    # (same method as RUI's SBS Connect)
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
        Connect to SBS BaseStation feed and count aircraft in JFK radius
        in real time.

        Python implementation of same logic as RUI JFK button ->
        CountFlightsInRadius().
        SBS feed port is 5002 (same as C++ RUI IdTCPClientSBS->Port=5002).

        SBS message format:
          MSG,type,sessionID,aircraftID,hexIdent,flightID,
          dateGen,timeGen,dateLog,timeLog,
          callsign,altitude,groundSpeed,track,lat,lon,
          verticalRate,squawk,...

        Args:
            host: SBS feed server host (default: 128.237.96.41)
            port: SBS feed port (default: 5002, same as RUI)
            collect_seconds: Collection duration in seconds. 15-30s recommended
                             to gather enough position data.
            center_lat: Center latitude (default: JFK 40.6413)
            center_lon: Center longitude (default: JFK -73.7781)
            radius_miles: Radius in miles (default: 50, same as RUI JFK button)

        Returns:
            (jfk_count, total_count, aircraft_info) tuple:
              - jfk_count: aircraft count within JFK radius
              - total_count: total unique aircraft count
              - aircraft_info: detailed info dict per aircraft
        """
        # Store latest position/info per aircraft
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

                    # Create/update aircraft entry
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

                    # Callsign (field 10)
                    if len(parts) > 10 and parts[10].strip():
                        ac["callsign"] = parts[10].strip()

                    # Altitude (field 11)
                    if len(parts) > 11 and parts[11].strip():
                        try:
                            ac["altitude"] = int(float(parts[11].strip()))
                        except ValueError:
                            pass

                    # Speed (field 12)
                    if len(parts) > 12 and parts[12].strip():
                        try:
                            ac["ground_speed"] = float(parts[12].strip())
                        except ValueError:
                            pass

                    # Latitude/longitude (fields 14, 15) - required for JFK distance calculation
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

        # Count aircraft within JFK radius (same as RUI's CountFlightsInRadius)
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
    # Legacy compatibility: simple ICAO counting method
    # ──────────────────────────────────────
    def get_realtime_count_from_tracker(
        self,
        host: str = "128.237.96.41",
        port: int = 5002,
        collect_seconds: int = 30,
    ) -> int:
        """
        Connect to SBS feed and return aircraft count within 50 miles of JFK.
        (Same result as RUI JFK button)

        Args:
            host: SBS feed host (default: 128.237.96.41)
            port: SBS feed port (default: 5002)
            collect_seconds: Collection duration (seconds)

        Returns:
            Aircraft count within 50-mile JFK radius
        """
        jfk_count, total_count, _ = self.get_realtime_count_from_sbs(
            host=host, port=port, collect_seconds=collect_seconds
        )
        return jfk_count

    # ──────────────────────────────────────
    # Integrated flow: SBS real-time collection + congestion evaluation
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
        Aggregate real-time aircraft count in JFK radius from SBS feed and
        evaluate congestion.

        Automatically performs Python equivalent of RUI SBS Connect
        (128.237.96.41:5002) -> JFK button action.

        Args:
            host: SBS feed host (default: 128.237.96.41)
            port: SBS feed port (default: 5002)
            collect_seconds: Collection duration (seconds). 30s recommended.
            hour: Hour bucket to compare (0-23). If None, uses current hour.
            radius_miles: Radius around JFK center (default: 50 miles)

        Returns:
            Dict in check_congestion() format plus SBS detail fields
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

        # Add SBS real-time detailed info
        result["source"] = "sbs_realtime"
        result["details"]["total_aircraft"] = total_count
        result["details"]["jfk_radius_miles"] = radius_miles
        result["details"]["sbs_host"] = host
        result["details"]["sbs_port"] = port
        result["details"]["collect_seconds"] = collect_seconds

        return result

    # ──────────────────────────────────────
    # Read JFK count from RUI (shared file)
    # ──────────────────────────────────────
    @staticmethod
    def get_count_from_rui(
        file_path: Optional[str] = None,
        max_age_seconds: float = 300,
    ) -> Optional[Dict]:
        """
        Read flight count from shared file written when RUI JFK button is clicked.

        RUI (C++ ADS-B Display) writes the following format to
        departure_prediction/data/jfk_realtime_count.json when JFK button is pressed:
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
            file_path: JSON file path. If None, default path is used.
            max_age_seconds: Data validity window in seconds. Default 300s (5 min).
                             Older data returns None.

        Returns:
            Returns JSON content as dict. Returns None if file does not exist or
            data is expired.
        """
        if file_path is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            file_path = str(data_dir / "jfk_realtime_count.json")

        if not os.path.isfile(file_path):
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check validity period
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
        Read flight count from RUI JFK-button output (shared file) and
        evaluate congestion.

        Flow:
          1. SBS Connect in RUI (128.237.96.41:5002)
          2. Click JFK button in RUI -> count written to file
          3. Call this function -> read file and analyze congestion

        Args:
            file_path: Shared JSON file path. If None, uses default path.
            max_age_seconds: Data validity window in seconds. Default 300s (5 min).
            hour: Hour bucket to compare (0-23). If None, uses current hour.

        Returns:
            congestion_info dict. Returns None if file is missing or data is stale.
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

        # Add RUI source info
        result["source"] = "rui"
        result["details"]["rui_timestamp"] = timestamp
        result["details"]["jfk_radius_miles"] = radius

        return result

    # ──────────────────────────────────────
    # Hourly statistics summary (for debugging/verification)
    # ──────────────────────────────────────
    def get_hourly_summary(self) -> str:
        """Return hourly mean/std summary table as a string."""
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
# Convenience function (module level)
# ──────────────────────────────────────────────

# Singleton instance (auto-load when CSV exists)
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_CSV = _DATA_DIR / "jfk_historical_flight_counts.csv"

_checker_instance: Optional[JFKCongestionChecker] = None


def get_checker(csv_path: Optional[str] = None) -> JFKCongestionChecker:
    """Return singleton JFKCongestionChecker instance."""
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
    Convenience function: evaluate JFK congestion from real-time flight count.

    Args:
        current_flight_count: Current flight count around JFK (from RUI)
        hour: Hour bucket to compare (0-23). If None, uses current hour.
        reference_time: Reference time (can be used instead of hour)

    Returns:
        congestion_info dict (compatible with hybrid_predictor.py)

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
    Read flight count from RUI JFK-button output and evaluate congestion.

    Flow:
      1. SBS Connect in RUI -> click JFK button
      2. Call this function -> read latest count from file and analyze congestion

    Args:
        hour: Hour bucket to compare (0-23). If None, uses current hour.
        max_age_seconds: Data validity window in seconds. Default 300s (5 min).

    Returns:
        congestion_info dict. Returns None if no RUI data is available.

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
# CLI test
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

    parser = argparse.ArgumentParser(description="Evaluate JFK airspace congestion")
    parser.add_argument(
        "--count", type=int, default=None,
        help="Current flight count (if omitted, collect from ADS-B tracker)"
    )
    parser.add_argument(
        "--hour", type=int, default=None,
        help="Hour bucket for comparison (0-23, default: current hour)"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to historical flight-count CSV file"
    )
    parser.add_argument(
        "--host", type=str, default="128.237.96.41",
        help="SBS feed host (default: 128.237.96.41)"
    )
    parser.add_argument(
        "--port", type=int, default=5002,
        help="SBS feed port (default: 5002, same as RUI)"
    )
    parser.add_argument(
        "--collect", type=int, default=3,
        help="SBS collection duration in seconds (default: 3)"
    )
    parser.add_argument(
        "--radius", type=float, default=50.0,
        help="Radius in miles around JFK center (default: 50.0, same as RUI)"
    )
    parser.add_argument(
        "--from-rui", action="store_true",
        help="Read flight count from RUI JFK-button output (shared file)"
    )
    parser.add_argument(
        "--max-age", type=float, default=300,
        help="RUI data validity in seconds (default: 300 = 5 min)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print hourly statistics summary"
    )
    args = parser.parse_args()

    checker = JFKCongestionChecker(csv_path=args.csv)

    if args.summary:
        print("\n=== Hourly Flight Count Statistics Around JFK ===\n")
        print(checker.get_hourly_summary())
        print()

    if args.count is not None:
        # Pass count directly
        result = checker.check_congestion(args.count, hour=args.hour)
    elif args.from_rui:
        # Read from RUI JFK button output
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
        # Real-time collection from SBS feed (same as RUI SBS Connect)
        result = checker.check_realtime_congestion(
            host=args.host, port=args.port,
            collect_seconds=args.collect, hour=args.hour,
            radius_miles=args.radius,
        )

    print(f"\n=== JFK Airspace Congestion Result ===")
    print(f"  Current Flight Count : {result['details']['current_count']}")
    print(f"  Compared Hour        : {result['details']['hour']}:00")
    print(f"  Historical Mean      : {result['details']['historical_mean']:.1f}")
    print(f"  Historical Std Dev   : {result['details']['historical_std']:.1f}")
    print(f"  Z-Score           : {result['details']['z_score']:.2f}")
    print(f"  Ratio (current/mean) : {result['details']['ratio']:.2f}")
    print(f"  ---")
    print(f"  Congestion Level     : {result['level'].upper()}")
    print(f"  Congestion Score     : {result['score']:.3f}")
    print(f"  Recommended Delay    : +{result['recommended_extra_delay']} min")
