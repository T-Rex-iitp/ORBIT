#!/usr/bin/env python3
"""Connect to ADS-B Raw and/or SBS TCP feeds with human-readable decoding.

Supports two display styles:
  --table : Interactive table view (like dump1090) – one row per aircraft,
            continuously refreshed. Shows Hex, Flight, Altitude, Speed,
            Lat, Lon, Track, Messages, Seen.
  (default) : Line-by-line decoded output.

Use --mode to select which feed(s) to connect to.

Examples:
    # Interactive table (recommended)
    python sbs_raw_connector.py --host 127.0.0.1 --mode raw --table

    # Line-by-line decoded output
    python sbs_raw_connector.py --host 127.0.0.1 --mode raw

    # SBS table view
    python sbs_raw_connector.py --host 127.0.0.1 --mode sbs --table

    # Both feeds, table view, save raw data to files
    python sbs_raw_connector.py --host 10.0.0.15 --mode both --table \\
        --raw-output raw.log --sbs-output sbs.log
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TextIO


# ═══════════════════════════════════════════════════════════════════════════════
#  ADS-B RAW Message Decoder  (line-by-line display)
# ═══════════════════════════════════════════════════════════════════════════════

def _adsb_char(code: int) -> str:
    """Convert a 6-bit ADS-B character code to ASCII (ICAO Annex 10 charset).

    Standard mapping (RTCA DO-260B Table 2-75):
      0x01-0x1A (1-26)  → A-Z
      0x20       (32)   → space
      0x30-0x39 (48-57) → 0-9
    """
    if 1 <= code <= 26:
        return chr(ord("A") + code - 1)
    if code == 32:
        return " "
    if 48 <= code <= 57:
        return chr(ord("0") + code - 48)
    return ""


def _decode_altitude(alt_bits: int) -> Optional[int]:
    """Decode the 12-bit altitude field.  Returns altitude in feet or *None*."""
    q_bit = (alt_bits >> 4) & 1
    if q_bit:
        n = ((alt_bits >> 5) << 4) | (alt_bits & 0xF)
        return n * 25 - 1000
    return None  # Gillham code – not decoded


# ── Type-Code specific decoders (for line-by-line mode) ─────────────────────

def _decode_identification(icao: str, tc: int, me: int) -> str:
    cat = (me >> 48) & 0x07
    callsign = "".join(
        _adsb_char((me >> (42 - i * 6)) & 0x3F) for i in range(8)
    ).strip()
    cat_map = {
        (4, 1): "Light", (4, 2): "Medium-1", (4, 3): "Medium-2",
        (4, 4): "High-Vortex", (4, 5): "Heavy", (4, 6): "High-Perf",
        (4, 7): "Rotorcraft",
        (3, 1): "Glider", (3, 2): "Lighter-than-air", (3, 3): "Parachutist",
        (3, 4): "Ultralight", (3, 6): "UAV", (3, 7): "Space vehicle",
    }
    cat_desc = cat_map.get((tc, cat), "")
    cat_str = f" ({cat_desc})" if cat_desc else ""
    return f"ICAO={icao} | Ident    | Callsign: {callsign or '?'}{cat_str}"


def _decode_airborne_position(icao: str, tc: int, me: int,
                              alt_type: str = "baro") -> str:
    ss = (me >> 49) & 0x03
    alt_bits = (me >> 36) & 0xFFF
    f_flag = (me >> 34) & 0x01
    lat_cpr = (me >> 17) & 0x1FFFF
    lon_cpr = me & 0x1FFFF
    altitude = _decode_altitude(alt_bits)
    alt_str = f"{altitude:,} ft" if altitude is not None else "N/A"
    cpr_type = "odd " if f_flag else "even"
    ss_labels = {0: "", 1: " [ALERT]", 2: " [TEMP-ALERT]", 3: " [SPI]"}
    return (
        f"ICAO={icao} | Position | Alt: {alt_str:>10s} ({alt_type}) | "
        f"CPR: {cpr_type} lat={lat_cpr:5d} lon={lon_cpr:5d}"
        f"{ss_labels.get(ss, '')}"
    )


def _decode_surface_position(icao: str, tc: int, me: int) -> str:
    s_hdg = (me >> 43) & 0x01
    hdg = ((me >> 36) & 0x7F) * 360.0 / 128.0 if s_hdg else None
    f_flag = (me >> 34) & 0x01
    lat_cpr = (me >> 17) & 0x1FFFF
    lon_cpr = me & 0x1FFFF
    hdg_str = f"Hdg: {hdg:5.1f}deg" if hdg is not None else "Hdg: N/A   "
    cpr_type = "odd " if f_flag else "even"
    return (
        f"ICAO={icao} | Surface  | {hdg_str} | "
        f"CPR: {cpr_type} lat={lat_cpr:5d} lon={lon_cpr:5d}"
    )


def _decode_airborne_velocity(icao: str, me: int) -> str:
    st = (me >> 48) & 0x07
    if st in (1, 2):
        s_ew = (me >> 42) & 0x01
        v_ew = (me >> 32) & 0x3FF
        s_ns = (me >> 31) & 0x01
        v_ns = (me >> 21) & 0x3FF
        vr_src = (me >> 20) & 0x01
        s_vr = (me >> 19) & 0x01
        vr = (me >> 10) & 0x1FF
        v_we = -(v_ew - 1) if s_ew else (v_ew - 1)
        v_sn = -(v_ns - 1) if s_ns else (v_ns - 1)
        if st == 2:
            v_we *= 4
            v_sn *= 4
        spd = math.sqrt(v_we ** 2 + v_sn ** 2)
        hdg = math.degrees(math.atan2(v_we, v_sn)) % 360
        if vr == 0:
            vr_str = "N/A"
        else:
            vr_val = (vr - 1) * 64 * (-1 if s_vr else 1)
            vr_str = f"{vr_val:+d} ft/min"
        vr_source = "GNSS" if vr_src else "baro"
        return (
            f"ICAO={icao} | Velocity | GS: {spd:3.0f} kt  "
            f"Hdg: {hdg:5.1f}deg  VR: {vr_str} ({vr_source})"
        )
    if st in (3, 4):
        s_hdg = (me >> 42) & 0x01
        hdg_raw = (me >> 32) & 0x3FF
        as_type = (me >> 31) & 0x01
        airspeed = (me >> 21) & 0x3FF
        hdg = hdg_raw * 360.0 / 1024.0 if s_hdg else None
        hdg_str = f"{hdg:5.1f}deg" if hdg is not None else "  N/A"
        as_label = "TAS" if as_type else "IAS"
        if st == 4:
            airspeed *= 4
        return f"ICAO={icao} | Velocity | {as_label}: {airspeed:3d} kt  Hdg: {hdg_str}"
    return f"ICAO={icao} | Velocity | subtype={st} (reserved)"


_DF_NAMES = {
    0: "Short ACAS", 4: "Surveillance,Alt", 5: "Surveillance,Ident",
    11: "All-Call Reply", 16: "Long ACAS", 20: "Comm-A,Alt",
    21: "Comm-A,Ident", 24: "Comm-C ELM",
}


def decode_adsb_raw(raw_line: str) -> str:
    """Decode a raw ADS-B hex string into a human-readable summary."""
    hex_str = raw_line.strip().lstrip("*").rstrip(";").strip()
    if len(hex_str) < 14:
        return f"(short msg, {len(hex_str)} hex chars) {hex_str}"
    try:
        msg_bytes = bytes.fromhex(hex_str)
    except ValueError:
        return f"(invalid hex) {hex_str}"
    df = (msg_bytes[0] >> 3) & 0x1F
    icao = f"{msg_bytes[1]:02X}{msg_bytes[2]:02X}{msg_bytes[3]:02X}"
    if df not in (17, 18) or len(msg_bytes) < 14:
        df_name = _DF_NAMES.get(df, f"DF={df}")
        return f"ICAO={icao} | {df_name}"
    me = int.from_bytes(msg_bytes[4:11], "big")
    tc = (me >> 51) & 0x1F
    if df == 18:
        cf = msg_bytes[0] & 0x07
        return f"ICAO={icao} | TIS-B/ADS-R (DF=18, CF={cf}) TC={tc}"
    if 1 <= tc <= 4:
        return _decode_identification(icao, tc, me)
    if 5 <= tc <= 8:
        return _decode_surface_position(icao, tc, me)
    if 9 <= tc <= 18:
        return _decode_airborne_position(icao, tc, me, alt_type="baro")
    if tc == 19:
        return _decode_airborne_velocity(icao, me)
    if 20 <= tc <= 22:
        return _decode_airborne_position(icao, tc, me, alt_type="GNSS")
    if tc == 28:
        return f"ICAO={icao} | Aircraft Status (TC=28)"
    if tc == 29:
        return f"ICAO={icao} | Target State & Status (TC=29)"
    if tc == 31:
        return f"ICAO={icao} | Operational Status (TC=31)"
    return f"ICAO={icao} | TC={tc} (unknown)"


# ═══════════════════════════════════════════════════════════════════════════════
#  SBS BaseStation Format Parser  (line-by-line display)
# ═══════════════════════════════════════════════════════════════════════════════

_SBS_MSG_LABELS = {
    "1": "Ident", "2": "Surface", "3": "Airborne", "4": "Velocity",
    "5": "Surv.Alt", "6": "Surv.ID", "7": "Air-Air", "8": "All-Call",
}


def format_sbs_line(line: str) -> str:
    """Parse an SBS BaseStation CSV line into a labelled, human-readable string."""
    parts = line.split(",")
    if len(parts) < 11 or parts[0] != "MSG":
        return line
    msg_type = parts[1].strip()
    icao = parts[4].strip()

    def _f(idx: int) -> str:
        return parts[idx].strip() if idx < len(parts) and parts[idx].strip() else ""

    callsign = _f(10)
    altitude = _f(11)
    gs = _f(12)
    track = _f(13)
    lat = _f(14)
    lon = _f(15)
    vr = _f(16)
    squawk = _f(17)
    on_ground = _f(21) if len(parts) > 21 else ""
    label = _SBS_MSG_LABELS.get(msg_type, f"Type{msg_type}")
    pieces = [f"ICAO={icao}", label]
    if callsign:
        pieces.append(f"CS={callsign}")
    if altitude:
        pieces.append(f"Alt={altitude}ft")
    if gs:
        pieces.append(f"GS={gs}kt")
    if track:
        pieces.append(f"Trk={track}deg")
    if lat and lon:
        pieces.append(f"Pos=({lat},{lon})")
    if vr:
        pieces.append(f"VR={vr}ft/min")
    if squawk:
        pieces.append(f"Sqk={squawk}")
    if on_ground in ("-1", "1"):
        pieces.append("[GROUND]")
    return " | ".join(pieces)


# ═══════════════════════════════════════════════════════════════════════════════
#  CPR Position Decoding  (Compact Position Reporting → lat/lon)
# ═══════════════════════════════════════════════════════════════════════════════

_CPR_MAX = 2 ** 17  # 131072
_NZ = 15            # Number of latitude zones


def _cpr_nl(lat: float) -> int:
    """Number of longitude zones at a given latitude (NL function)."""
    if abs(lat) >= 87.0:
        return 1
    a = 1.0 - math.cos(math.pi / (2.0 * _NZ))
    b = math.cos(math.pi * abs(lat) / 180.0) ** 2
    val = 1.0 - a / b
    if val < -1.0:
        return 1
    if val > 1.0:
        return 59
    return int(math.floor(2.0 * math.pi / math.acos(val)))


def _cpr_decode_global(
    even_lat: int, even_lon: int,
    odd_lat: int, odd_lon: int,
    last_is_odd: bool,
) -> Optional[tuple[float, float]]:
    """Decode a CPR even/odd pair into (lat, lon) using global unambiguity.

    Returns *None* when the pair crosses a latitude zone boundary.
    """
    d_lat0 = 360.0 / 60.0  # even  (4 * NZ zones)
    d_lat1 = 360.0 / 59.0  # odd   (4 * NZ - 1 zones)

    lat_even = even_lat / _CPR_MAX
    lat_odd = odd_lat / _CPR_MAX
    lon_even = even_lon / _CPR_MAX
    lon_odd = odd_lon / _CPR_MAX

    # ICAO Annex 10 / RTCA DO-260B: multipliers are (4*NZ-1) and (4*NZ)
    j = math.floor(59.0 * lat_even - 60.0 * lat_odd + 0.5)

    lat0 = d_lat0 * ((j % 60) + lat_even)
    lat1 = d_lat1 * ((j % 59) + lat_odd)

    if lat0 >= 270.0:
        lat0 -= 360.0
    if lat1 >= 270.0:
        lat1 -= 360.0

    # Zone consistency check
    nl0 = _cpr_nl(lat0)
    nl1 = _cpr_nl(lat1)
    if nl0 != nl1:
        return None  # zone crossing – cannot decode

    if last_is_odd:
        lat = lat1
        nl = nl1
        ni = max(nl - 1, 1)
        m = math.floor(lon_even * (nl - 1) - lon_odd * nl + 0.5)
        lon = (360.0 / ni) * ((m % ni) + lon_odd)
    else:
        lat = lat0
        nl = nl0
        ni = max(nl, 1)
        m = math.floor(lon_even * (nl - 1) - lon_odd * nl + 0.5)
        lon = (360.0 / ni) * ((m % ni) + lon_even)

    if lon > 180.0:
        lon -= 360.0

    return (lat, lon)


# ═══════════════════════════════════════════════════════════════════════════════
#  Aircraft State & Tracker  (for --table mode)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AircraftState:
    """Accumulated state for a single aircraft."""
    icao: str
    callsign: str = ""
    altitude: Optional[int] = None
    ground_speed: Optional[float] = None
    track: Optional[float] = None
    vertical_rate: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    squawk: str = ""
    messages: int = 0
    last_seen: float = 0.0
    # CPR state
    cpr_even_lat: Optional[int] = None
    cpr_even_lon: Optional[int] = None
    cpr_even_time: float = 0.0
    cpr_odd_lat: Optional[int] = None
    cpr_odd_lon: Optional[int] = None
    cpr_odd_time: float = 0.0


class AircraftTracker:
    """Accumulates per-aircraft data from RAW and/or SBS feeds."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.aircraft: dict[str, AircraftState] = {}
        self.lock = threading.Lock()
        self.timeout = timeout

    # ── RAW feed update ──────────────────────────────────────────────────────

    def update_raw(self, raw_line: str) -> None:
        hex_str = raw_line.strip().lstrip("*").rstrip(";").strip()
        if len(hex_str) < 28:
            return
        try:
            msg = bytes.fromhex(hex_str)
        except ValueError:
            return

        df = (msg[0] >> 3) & 0x1F
        if df not in (17, 18) or len(msg) < 14:
            return

        icao = f"{msg[1]:02x}{msg[2]:02x}{msg[3]:02x}"
        me = int.from_bytes(msg[4:11], "big")
        tc = (me >> 51) & 0x1F
        now = time.time()

        with self.lock:
            ac = self.aircraft.setdefault(icao, AircraftState(icao=icao))
            ac.messages += 1
            ac.last_seen = now

            if 1 <= tc <= 4:
                self._parse_ident(ac, me)
            elif 5 <= tc <= 8:
                self._parse_surface_pos(ac, me, now)
            elif 9 <= tc <= 18:
                self._parse_airborne_pos(ac, me, now)
            elif tc == 19:
                self._parse_velocity(ac, me)
            elif 20 <= tc <= 22:
                self._parse_airborne_pos(ac, me, now)

    def _parse_ident(self, ac: AircraftState, me: int) -> None:
        cs = "".join(
            _adsb_char((me >> (42 - i * 6)) & 0x3F) for i in range(8)
        ).strip()
        if cs:
            ac.callsign = cs

    def _parse_surface_pos(self, ac: AircraftState, me: int, now: float) -> None:
        f_flag = (me >> 34) & 0x01
        lat_cpr = (me >> 17) & 0x1FFFF
        lon_cpr = me & 0x1FFFF
        if f_flag:
            ac.cpr_odd_lat, ac.cpr_odd_lon, ac.cpr_odd_time = lat_cpr, lon_cpr, now
        else:
            ac.cpr_even_lat, ac.cpr_even_lon, ac.cpr_even_time = lat_cpr, lon_cpr, now
        self._try_cpr(ac)

    def _parse_airborne_pos(self, ac: AircraftState, me: int, now: float) -> None:
        alt_bits = (me >> 36) & 0xFFF
        f_flag = (me >> 34) & 0x01
        lat_cpr = (me >> 17) & 0x1FFFF
        lon_cpr = me & 0x1FFFF

        alt = _decode_altitude(alt_bits)
        if alt is not None:
            ac.altitude = alt

        if f_flag:
            ac.cpr_odd_lat, ac.cpr_odd_lon, ac.cpr_odd_time = lat_cpr, lon_cpr, now
        else:
            ac.cpr_even_lat, ac.cpr_even_lon, ac.cpr_even_time = lat_cpr, lon_cpr, now
        self._try_cpr(ac)

    def _parse_velocity(self, ac: AircraftState, me: int) -> None:
        st = (me >> 48) & 0x07
        if st in (1, 2):
            s_ew = (me >> 42) & 0x01
            v_ew = (me >> 32) & 0x3FF
            s_ns = (me >> 31) & 0x01
            v_ns = (me >> 21) & 0x3FF
            s_vr = (me >> 19) & 0x01
            vr = (me >> 10) & 0x1FF

            v_we = -(v_ew - 1) if s_ew else (v_ew - 1)
            v_sn = -(v_ns - 1) if s_ns else (v_ns - 1)
            if st == 2:
                v_we *= 4
                v_sn *= 4

            ac.ground_speed = math.sqrt(v_we ** 2 + v_sn ** 2)
            ac.track = math.degrees(math.atan2(v_we, v_sn)) % 360.0

            if vr > 0:
                ac.vertical_rate = (vr - 1) * 64 * (-1 if s_vr else 1)

    def _try_cpr(self, ac: AircraftState) -> None:
        """Attempt CPR global decode when both even and odd frames are available."""
        if ac.cpr_even_lat is None or ac.cpr_odd_lat is None:
            return
        if abs(ac.cpr_even_time - ac.cpr_odd_time) > 10.0:
            return  # frames too far apart
        last_is_odd = ac.cpr_odd_time >= ac.cpr_even_time
        result = _cpr_decode_global(
            ac.cpr_even_lat, ac.cpr_even_lon,
            ac.cpr_odd_lat, ac.cpr_odd_lon,
            last_is_odd,
        )
        if result:
            ac.lat, ac.lon = result

    # ── SBS feed update ──────────────────────────────────────────────────────

    def update_sbs(self, line: str) -> None:
        parts = line.split(",")
        if len(parts) < 11 or parts[0] != "MSG":
            return
        icao = parts[4].strip().lower()
        if not icao:
            return
        now = time.time()

        def _f(idx: int) -> str:
            return parts[idx].strip() if idx < len(parts) and parts[idx].strip() else ""

        with self.lock:
            ac = self.aircraft.setdefault(icao, AircraftState(icao=icao))
            ac.messages += 1
            ac.last_seen = now

            cs = _f(10)
            if cs:
                ac.callsign = cs
            val = _f(11)
            if val:
                try:
                    ac.altitude = int(float(val))
                except ValueError:
                    pass
            val = _f(12)
            if val:
                try:
                    ac.ground_speed = float(val)
                except ValueError:
                    pass
            val = _f(13)
            if val:
                try:
                    ac.track = float(val)
                except ValueError:
                    pass
            lat_s, lon_s = _f(14), _f(15)
            if lat_s and lon_s:
                try:
                    ac.lat, ac.lon = float(lat_s), float(lon_s)
                except ValueError:
                    pass
            val = _f(16)
            if val:
                try:
                    ac.vertical_rate = int(float(val))
                except ValueError:
                    pass
            val = _f(17)
            if val:
                ac.squawk = val

    # ── Table rendering ──────────────────────────────────────────────────────

    def render_table(self) -> str:
        """Return a full table string for terminal display."""
        now = time.time()
        with self.lock:
            # Prune stale aircraft
            stale = [k for k, v in self.aircraft.items()
                     if now - v.last_seen > self.timeout]
            for k in stale:
                del self.aircraft[k]
            # Sort: most recently seen first
            ac_list = sorted(self.aircraft.values(),
                             key=lambda a: a.last_seen, reverse=True)
            count = len(ac_list)

        # Column header (matches dump1090 style)
        header = (
            f" {'Hex':<7s} {'Flight':<9s} {'Altitude':>8s} {'Speed':>6s}"
            f" {'Lat':>8s} {'Lon':>10s} {'Track':>6s}"
            f" {'Msgs':>5s} {'Seen':>6s}"
        )
        sep = "\u2500" * len(header)

        lines: list[str] = [header, sep]
        for ac in ac_list:
            seen = int(now - ac.last_seen)
            if seen < 60:
                seen_s = f"{seen} sec"
            elif seen < 3600:
                seen_s = f"{seen // 60} min"
            else:
                seen_s = f"{seen // 3600}h"

            alt_s = str(ac.altitude) if ac.altitude is not None else ""
            spd_s = f"{ac.ground_speed:.0f}" if ac.ground_speed is not None else ""
            lat_s = f"{ac.lat:.3f}" if ac.lat is not None else ""
            lon_s = f"{ac.lon:.3f}" if ac.lon is not None else ""
            trk_s = f"{ac.track:.0f}" if ac.track is not None else ""

            lines.append(
                f" {ac.icao:<7s} {ac.callsign:<9s} {alt_s:>8s} {spd_s:>6s}"
                f" {lat_s:>8s} {lon_s:>10s} {trk_s:>6s}"
                f" {ac.messages:>5d} {seen_s:>6s}"
            )

        lines.append(sep)
        lines.append(f" Aircraft: {count}   (timeout: {int(self.timeout)}s)")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Network Feed Workers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeedConfig:
    name: str
    host: str
    port: int
    output_path: Optional[Path]
    formatter: Callable[[str], str]


class FeedWorker(threading.Thread):
    """Line-by-line mode: read → format → print."""

    def __init__(self, config: FeedConfig, stop_event: threading.Event,
                 reconnect_delay: float) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.reconnect_delay = reconnect_delay
        self.output_fp: Optional[TextIO] = None

    def run(self) -> None:
        try:
            if self.config.output_path is not None:
                self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_fp = self.config.output_path.open("a", encoding="utf-8")
            while not self.stop_event.is_set():
                try:
                    self._stream_once()
                except (ConnectionError, OSError, socket.timeout) as exc:
                    if self.stop_event.is_set():
                        break
                    print(
                        f"[{self.config.name}] connection issue: {exc}. "
                        f"reconnecting in {self.reconnect_delay:.1f}s"
                    )
                    self.stop_event.wait(self.reconnect_delay)
        finally:
            if self.output_fp is not None:
                self.output_fp.close()

    def _stream_once(self) -> None:
        print(f"[{self.config.name}] connecting to {self.config.host}:{self.config.port} ...")
        with socket.create_connection((self.config.host, self.config.port), timeout=10) as conn:
            conn.settimeout(1.0)
            print(f"[{self.config.name}] connected")
            with conn.makefile("r", encoding="utf-8", errors="replace",
                               newline="\n") as reader:
                while not self.stop_event.is_set():
                    try:
                        line = reader.readline()
                    except socket.timeout:
                        continue
                    if not line:
                        raise ConnectionError("remote peer closed the connection")
                    line = line.rstrip("\r\n")
                    if not line:
                        continue
                    try:
                        display = self.config.formatter(line)
                    except Exception:
                        display = line
                    timestamp_ms = int(time.time() * 1000)
                    print(f"[{self.config.name}] {display}")
                    if self.output_fp is not None:
                        self.output_fp.write(f"{timestamp_ms}\t{line}\n")
                        self.output_fp.flush()


@dataclass(frozen=True)
class TrackerFeedConfig:
    name: str
    host: str
    port: int
    output_path: Optional[Path]
    update_fn: Callable[[str], None]   # tracker.update_raw or tracker.update_sbs


class TrackerFeedWorker(threading.Thread):
    """Table mode: read → feed into AircraftTracker (no per-line printing)."""

    def __init__(self, config: TrackerFeedConfig, stop_event: threading.Event,
                 reconnect_delay: float) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.stop_event = stop_event
        self.reconnect_delay = reconnect_delay
        self.output_fp: Optional[TextIO] = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def run(self) -> None:
        try:
            if self.config.output_path is not None:
                self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_fp = self.config.output_path.open("a", encoding="utf-8")
            while not self.stop_event.is_set():
                try:
                    self._stream_once()
                except (ConnectionError, OSError, socket.timeout):
                    self._connected = False
                    if self.stop_event.is_set():
                        break
                    self.stop_event.wait(self.reconnect_delay)
        finally:
            self._connected = False
            if self.output_fp is not None:
                self.output_fp.close()

    def _stream_once(self) -> None:
        with socket.create_connection((self.config.host, self.config.port), timeout=10) as conn:
            conn.settimeout(1.0)
            self._connected = True
            with conn.makefile("r", encoding="utf-8", errors="replace",
                               newline="\n") as reader:
                while not self.stop_event.is_set():
                    try:
                        line = reader.readline()
                    except socket.timeout:
                        continue
                    if not line:
                        raise ConnectionError("remote peer closed")
                    line = line.rstrip("\r\n")
                    if not line:
                        continue
                    try:
                        self.config.update_fn(line)
                    except Exception:
                        pass
                    if self.output_fp is not None:
                        ts = int(time.time() * 1000)
                        self.output_fp.write(f"{ts}\t{line}\n")
                        self.output_fp.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  Table Display (ANSI terminal refresh)
# ═══════════════════════════════════════════════════════════════════════════════

class TableDisplay(threading.Thread):
    """Periodically clears terminal and redraws the aircraft table."""

    def __init__(self, tracker: AircraftTracker, stop_event: threading.Event,
                 workers: list[TrackerFeedWorker],
                 refresh: float = 1.0, host: str = "", mode: str = "") -> None:
        super().__init__(daemon=True)
        self.tracker = tracker
        self.stop_event = stop_event
        self.workers = workers
        self.refresh = refresh
        self.host = host
        self.mode = mode

    def run(self) -> None:
        while not self.stop_event.is_set():
            self._draw()
            self.stop_event.wait(self.refresh)
        # Final draw before exit
        self._draw()

    def _draw(self) -> None:
        # Build connection status line
        status_parts = []
        for w in self.workers:
            st = "\033[32mOK\033[0m" if w.connected else "\033[31mWAIT\033[0m"
            status_parts.append(f"{w.config.name}:{w.config.port}={st}")
        conn_status = "  ".join(status_parts)

        table = self.tracker.render_table()

        # ANSI: move cursor home + clear to end of screen
        output = (
            f"\033[H\033[J"
            f"\033[1m ADS-B Live Table\033[0m  [{self.host}]  {conn_status}\n"
            f"\n{table}\n\n"
            f" Press Ctrl+C to stop.\n"
        )
        sys.stdout.write(output)
        sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADS-B feed connector with human-readable decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --host 127.0.0.1 --mode raw --table     # interactive table
  %(prog)s --host 127.0.0.1 --mode raw              # line-by-line decoded
  %(prog)s --host 127.0.0.1 --mode sbs --table      # SBS table
  %(prog)s --host 10.0.0.15 --mode both --table     # both feeds, table
  %(prog)s --host 10.0.0.15 --mode raw --raw-hex    # raw hex (no decode)
  %(prog)s --host 10.0.0.15 --mode both --table \\
      --raw-output raw.log --sbs-output sbs.log --duration 60
""",
    )

    conn = parser.add_argument_group("connection")
    conn.add_argument("--host", default="127.0.0.1",
                      help="Feed server host (default: %(default)s)")
    conn.add_argument("--mode", choices=["raw", "sbs", "both"], default="both",
                      help="Feed: raw (port 30002), sbs (port 5002), both (default: %(default)s)")
    conn.add_argument("--raw-port", type=int, default=30002,
                      help="Raw feed TCP port (default: %(default)s)")
    conn.add_argument("--sbs-port", type=int, default=5002,
                      help="SBS feed TCP port (default: %(default)s)")

    disp = parser.add_argument_group("display")
    disp.add_argument("--table", action="store_true",
                      help="Interactive table view (like dump1090)")
    disp.add_argument("--refresh", type=float, default=1.0,
                      help="Table refresh interval in seconds (default: %(default)s)")
    disp.add_argument("--timeout", type=float, default=60.0,
                      help="Remove aircraft not seen for N seconds (default: %(default)s)")
    disp.add_argument("--raw-hex", action="store_true",
                      help="Show raw hex without decoding (line-by-line mode)")
    disp.add_argument("--sbs-csv", action="store_true",
                      help="Show raw CSV without formatting (line-by-line mode)")

    rec = parser.add_argument_group("recording")
    rec.add_argument("--raw-output", type=Path, help="File to save raw stream")
    rec.add_argument("--sbs-output", type=Path, help="File to save SBS stream")

    misc = parser.add_argument_group("misc")
    misc.add_argument("--duration", type=float, default=0.0,
                      help="Auto-stop after N seconds (0 = forever, default: %(default)s)")
    misc.add_argument("--reconnect-delay", type=float, default=3.0,
                      help="Reconnect delay in seconds (default: %(default)s)")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    args = parse_args()
    stop_event = threading.Event()

    def handle_signal(_sig: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ── Table mode ───────────────────────────────────────────────────────────
    if args.table:
        tracker = AircraftTracker(timeout=args.timeout)
        workers: list[TrackerFeedWorker] = []

        if args.mode in ("raw", "both"):
            workers.append(TrackerFeedWorker(
                TrackerFeedConfig("RAW", args.host, args.raw_port,
                                  args.raw_output, tracker.update_raw),
                stop_event=stop_event,
                reconnect_delay=args.reconnect_delay,
            ))
        if args.mode in ("sbs", "both"):
            workers.append(TrackerFeedWorker(
                TrackerFeedConfig("SBS", args.host, args.sbs_port,
                                  args.sbs_output, tracker.update_sbs),
                stop_event=stop_event,
                reconnect_delay=args.reconnect_delay,
            ))

        display = TableDisplay(
            tracker, stop_event, workers,
            refresh=args.refresh, host=args.host, mode=args.mode,
        )

        for w in workers:
            w.start()
        display.start()

        if args.duration > 0:
            stop_event.wait(args.duration)
            stop_event.set()
        else:
            while not stop_event.is_set():
                time.sleep(0.2)

        for w in workers:
            w.join(timeout=2)
        display.join(timeout=2)

        # Restore terminal
        sys.stdout.write("\033[?25h")  # show cursor
        sys.stdout.flush()
        print("\nStopped.")
        return 0

    # ── Line-by-line mode (original) ─────────────────────────────────────────
    raw_fmt: Callable[[str], str] = (lambda l: l) if args.raw_hex else decode_adsb_raw
    sbs_fmt: Callable[[str], str] = (lambda l: l) if args.sbs_csv else format_sbs_line

    line_workers: list[FeedWorker] = []
    if args.mode in ("raw", "both"):
        line_workers.append(FeedWorker(
            FeedConfig("RAW", args.host, args.raw_port, args.raw_output, raw_fmt),
            stop_event=stop_event, reconnect_delay=args.reconnect_delay,
        ))
    if args.mode in ("sbs", "both"):
        line_workers.append(FeedWorker(
            FeedConfig("SBS", args.host, args.sbs_port, args.sbs_output, sbs_fmt),
            stop_event=stop_event, reconnect_delay=args.reconnect_delay,
        ))

    print("=" * 65)
    print("  ADS-B Feed Connector")
    print("=" * 65)
    print(f"  Host : {args.host}")
    print(f"  Mode : {args.mode.upper()}")
    if args.mode in ("raw", "both"):
        lbl = "hex (raw)" if args.raw_hex else "decoded"
        print(f"  RAW  : port {args.raw_port}  [{lbl}]")
    if args.mode in ("sbs", "both"):
        lbl = "csv (raw)" if args.sbs_csv else "parsed"
        print(f"  SBS  : port {args.sbs_port}  [{lbl}]")
    if args.duration > 0:
        print(f"  Stop : after {args.duration}s")
    print("=" * 65)
    print("Press Ctrl+C to stop.\n")

    for w in line_workers:
        w.start()

    if args.duration > 0:
        stop_event.wait(args.duration)
        stop_event.set()
    else:
        while not stop_event.is_set():
            time.sleep(0.2)

    for w in line_workers:
        w.join()

    print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
