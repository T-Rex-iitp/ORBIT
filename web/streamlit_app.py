#!/usr/bin/env python3
"""
IFTA Departure Assistant — Enhanced Streamlit Dashboard

Features:
  - Visual departure timeline with step-by-step breakdown
  - Interactive map with airport & flight position markers (folium)
  - Google Maps traffic-aware routing (embed + link)
  - Real-time flight position tracking (AviationStack / hooked ADS-B)
  - Airport security & check-in information panel
  - Modern dark-themed dashboard UI

Run:
  pip install -r web/requirements.txt
  streamlit run web/streamlit_app.py --server.port 8051
"""

from __future__ import annotations

import os
import pickle
import re
import sys
import tempfile
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

# ── Project paths ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
SPEECH_DIR = ROOT_DIR / "speech"
DEPARTURE_PRED_DIR = ROOT_DIR / "departure_prediction"
ADSB_PY_DIR = ROOT_DIR / "ADS-B-Display" / "python"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(DEPARTURE_PRED_DIR) not in sys.path:
    sys.path.insert(0, str(DEPARTURE_PRED_DIR))
if str(ADSB_PY_DIR) not in sys.path:
    sys.path.insert(0, str(ADSB_PY_DIR))

# Add speech directory to Python path for direct imports
if str(SPEECH_DIR) not in sys.path:
    sys.path.insert(0, str(SPEECH_DIR))

# ── Import departure_brief functions ───────────────────────────
try:
    from departure_brief import (
        load_config,
        detect_departure_intent,
        fetch_drive_info,
        fetch_aviationstack_position,
        build_maps_url,
        format_dt,
        _get_now_local,
        _parse_departure_time,
        _extract_time_from_query,
        _extract_airport_from_query,
        _extract_airport_from_route,
        _normalize_airport_code,
        _extract_flight_number_from_query,
    )

    HAS_DEPARTURE_BRIEF = True
except ImportError:
    HAS_DEPARTURE_BRIEF = False

# ── Optional: folium for interactive maps ──────────────────────
try:
    import folium
    from streamlit_folium import st_folium

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

# ── Optional: zoneinfo ─────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# ── Optional: Gemini AI ───────────────────────────────────────
try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# ── Optional: Gemini ticket OCR module ─────────────────────────
try:
    from utils.gemini_direct_client import GeminiTicketOCR

    HAS_GEMINI_TICKET_OCR = True
except Exception:
    HAS_GEMINI_TICKET_OCR = False

# ── Optional: Transformer delay predictor deps ────────────────
try:
    import numpy as np
    import torch
    import torch.nn as nn

    HAS_TRANSFORMER_DEPS = True
except ImportError:
    HAS_TRANSFORMER_DEPS = False

# ── Optional: Airport intel (TSA crawl / gate walk) ───────────
try:
    from utils.tsa_wait_time import get_tsa_wait_time
    from utils.gate_walk_time import get_gate_walk_time

    HAS_AIRPORT_INTEL = True
except Exception:
    HAS_AIRPORT_INTEL = False

# ── Optional: Operational factors (JFK congestion + FR24/ADS-B previous leg) ──
try:
    from utils.operational_factors import OperationalFactorsAnalyzer
    from utils.congestion_check import JFKCongestionChecker

    HAS_OPERATIONAL_FACTORS = True
except Exception:
    HAS_OPERATIONAL_FACTORS = False

try:
    from previous_flight_finder import estimate_previous_leg_delay_minutes

    HAS_PREVIOUS_LEG_ESTIMATOR = True
except Exception:
    HAS_PREVIOUS_LEG_ESTIMATOR = False

# ── Default config path ───────────────────────────────────────
DEFAULT_CONFIG_PATH = SPEECH_DIR / "departure_config.json"
DEFAULT_TRANSFORMER_MODEL_PATH = (
    ROOT_DIR / "departure_prediction" / "models" / "delay_predictor_full.pkl"
)

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

AIRPORT_COORDS: Dict[str, Tuple[float, float]] = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "SFO": (37.6213, -122.3790),
    "LAX": (33.9425, -118.4081),
    "SEA": (47.4502, -122.3088),
    "ICN": (37.4602, 126.4407),
    "GMP": (37.5583, 126.7906),
    "ORD": (41.9742, -87.9073),
    "ATL": (33.6407, -84.4277),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),
    "MIA": (25.7959, -80.2870),
    "BOS": (42.3656, -71.0096),
    "PHL": (39.8744, -75.2424),
    "IAD": (38.9531, -77.4565),
    "DCA": (38.8512, -77.0402),
    "PIT": (40.4919, -80.2329),
    "HND": (35.5494, 139.7798),
    "NRT": (35.7647, 140.3864),
    "PEK": (40.0799, 116.6031),
    "PVG": (31.1443, 121.8083),
    "CDG": (49.0097, 2.5479),
    "LHR": (51.4700, -0.4543),
    "FRA": (50.0379, 8.5622),
}

TRAFFIC_COLORS: Dict[str, str] = {
    "light": "#4CAF50",
    "moderate": "#FF9800",
    "heavy": "#F44336",
    "unknown": "#78909C",
}

TRAFFIC_EMOJI: Dict[str, str] = {
    "light": "🟢",
    "moderate": "🟡",
    "heavy": "🔴",
    "unknown": "⚪",
}


def _as_bool(value: Any, default: bool = False) -> bool:
    """Normalize bool-ish values from env/config."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n"}:
            return False
    return default


def _resolve_bool_setting(
    config: Dict[str, Any],
    config_key: str,
    env_key: str,
    default: bool,
) -> bool:
    """
    Resolve a feature flag with env-var override.
    Priority: environment variable > config file > default.
    """
    env_raw = os.environ.get(env_key)
    if env_raw is not None:
        return _as_bool(env_raw, default)
    return _as_bool(config.get(config_key, default), default)

# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Page layout ───────────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1300px;
}

/* ── Metric cards ──────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid #2a5a8f;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
div[data-testid="stMetric"] label {
    color: #8ab4f8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #8ab4f8 !important;
}

/* ── Info card ─────────────────────────────────────── */
.info-card {
    background: linear-gradient(135deg, #1a2f4a 0%, #0f1f33 100%);
    border: 1px solid #2a4a6f;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}
.info-card h4 {
    color: #8ab4f8;
    margin: 0 0 12px 0;
    font-size: 1.05rem;
}
.info-card p {
    color: #d0d0d0;
    margin: 4px 0;
    font-size: 0.9rem;
}
.info-card .val {
    color: #ffffff;
    font-weight: 600;
}
.info-card .subtle {
    color: #9bb8d6;
    font-size: 0.82rem;
}

/* ── KPI strip inside cards ───────────────────────── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin: 6px 0 12px 0;
}
.kpi-item {
    background: rgba(138, 180, 248, 0.08);
    border: 1px solid rgba(138, 180, 248, 0.28);
    border-radius: 10px;
    padding: 8px 10px;
}
.kpi-label {
    color: #9bb8d6;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.kpi-value {
    color: #ffffff;
    font-size: 1.08rem;
    font-weight: 700;
    line-height: 1.3;
}
.tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0 12px 0;
}
.tag {
    display: inline-block;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 600;
    color: #d9ebff;
    background: rgba(66, 133, 244, 0.2);
    border: 1px solid rgba(138, 180, 248, 0.35);
}
.kv-table {
    width: 100%;
    border-collapse: collapse;
}
.kv-table td {
    padding: 7px 0;
    font-size: 0.9rem;
    color: #d7e3f3 !important;
}
.kv-table td:first-child {
    color: #d7e3f3 !important;
    font-weight: 520;
}
.kv-table td:last-child {
    text-align: right;
    color: #fff;
    font-weight: 700;
}
.section-block {
    background: rgba(18, 43, 69, 0.55);
    border: 1px solid rgba(138, 180, 248, 0.2);
    border-radius: 10px;
    padding: 10px 12px;
    margin: 10px 0;
}
.section-block.delay-block {
    background: rgba(58, 41, 87, 0.42);
    border-color: rgba(173, 145, 255, 0.3);
}
.section-title {
    color: #cfe5ff;
    font-size: 0.84rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin: 0 0 6px 0;
}
.total-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 12px;
}
.total-box {
    background: rgba(138, 180, 248, 0.08);
    border: 1px solid rgba(138, 180, 248, 0.28);
    border-radius: 10px;
    padding: 9px 10px;
}
.total-box.main {
    background: rgba(76, 175, 80, 0.14);
    border-color: rgba(129, 199, 132, 0.48);
}
.total-box .lbl {
    color: #9bb8d6;
    font-size: 0.75rem;
    font-weight: 600;
}
.total-box .num {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 800;
}

@media (max-width: 900px) {
    .kpi-strip {
        grid-template-columns: 1fr;
    }
    .total-strip {
        grid-template-columns: 1fr;
    }
}

/* ── Section header ────────────────────────────────── */
.sec-hdr {
    color: #8ab4f8;
    font-size: 1.15rem;
    font-weight: 600;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #1e3a5f;
}

/* ── Recommendation banner ─────────────────────────── */
.rec-banner {
    background: linear-gradient(135deg, #1a3a2a 0%, #0d2137 100%);
    border: 1px solid #2a6a3f;
    border-radius: 12px;
    padding: 16px 24px;
    margin: 10px 0 20px 0;
}
.rec-banner .rec-title {
    font-size: 1.25rem;
    color: #4CAF50;
    font-weight: 700;
}
.rec-banner .rec-body {
    color: #e0e0e0;
    font-size: 1.05rem;
    margin: 8px 0 0 0;
    line-height: 1.6;
}
.rec-banner .rec-body b.leave {
    color: #4CAF50;
    font-size: 1.2rem;
}
.rec-banner .rec-body b.airport {
    color: #8ab4f8;
}

/* ── Google Maps button ────────────────────────────── */
.gmaps-btn {
    display: inline-block;
    background: linear-gradient(135deg, #4285f4 0%, #1a73e8 100%);
    color: white !important;
    padding: 10px 24px;
    border-radius: 8px;
    text-decoration: none !important;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(66,133,244,0.3);
    transition: all 0.2s;
    margin-top: 10px;
}
.gmaps-btn:hover {
    box-shadow: 0 4px 16px rgba(66,133,244,0.5);
    transform: translateY(-1px);
}

/* ── Timeline ──────────────────────────────────────── */
.tl-wrap {
    padding: 10px 12px;
    background: linear-gradient(180deg, #10233a 0%, #132b46 100%);
    border: 1px solid #27496d;
    border-radius: 12px;
}
.tl-item {
    display: flex;
    align-items: flex-start;
    margin-bottom: 6px;
    position: relative;
}
.tl-item.key-row {
    margin-top: 2px;
    margin-bottom: 8px;
}
.tl-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    margin-right: 14px;
    margin-top: 4px;
    flex-shrink: 0;
    z-index: 1;
}
.tl-dot.key {
    width: 14px; height: 14px;
    margin-top: 3px;
}
.tl-dot.substep {
    width: 10px; height: 10px;
    margin-top: 5px;
    opacity: 0.92;
}
.tl-line {
    position: absolute;
    left: 5px; top: 18px; bottom: -6px;
    width: 2px;
    background: #3b6a9e;
}
.tl-body { flex: 1; display: flex; align-items: baseline; flex-wrap: wrap; gap: 2px 8px; }
.tl-item.key-row .tl-body {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(138, 180, 248, 0.35);
    border-radius: 8px;
    padding: 5px 9px;
}
.tl-time {
    font-weight: 700; color: #aecbff;
    font-size: 0.95rem; min-width: 55px;
}
.tl-label { color: #eef4ff; font-size: 0.9rem; font-weight: 520; }
.tl-label.substep {
    color: #d8e6fb;
    font-size: 0.86rem;
}
.tl-dur { color: #b7cae5; font-size: 0.82rem; font-weight: 520; }

/* ── Countdown chip ────────────────────────────────── */
.countdown-chip {
    display: inline-block;
    background: #1b5e20;
    color: #81c784;
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-left: 8px;
}
.countdown-chip.past {
    background: #b71c1c;
    color: #ef9a9a;
}

/* ── Welcome screen ────────────────────────────────── */
.welcome-box {
    text-align: center;
    padding: 60px 20px;
    color: #78909C;
}
.welcome-box .icon { font-size: 4rem; margin-bottom: 16px; }
.welcome-box h2 { color: #90a4ae; margin-bottom: 8px; }
.welcome-box p { font-size: 1rem; line-height: 1.8; }

/* ── Notes badge ───────────────────────────────────── */
.note-badge {
    display: inline-block;
    background: #263238;
    color: #b0bec5;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.8rem;
    margin: 2px;
}
</style>
"""


# ═══════════════════════════════════════════════════════════════
#  GEMINI AI INTEGRATION
# ═══════════════════════════════════════════════════════════════

GEMINI_PARSE_PROMPT = """\
You are a departure-time assistant for an airport travel planner.

Given the user's query below, extract the following information as JSON.
If a field cannot be determined, use null.

Fields:
- is_departure_query (bool): Is the user asking about when to leave / depart for an airport?
- airport_code (string|null): IATA airport code mentioned (e.g. "JFK", "ICN"). Convert airport names like Incheon->ICN, Gimpo->GMP.
- flight_number (string|null): Flight number if mentioned (e.g. "KE937", "UA123").
- departure_time (string|null): Time mentioned for flight departure, format "HH:MM" or "YYYY-MM-DD HH:MM".
- origin (string|null): Origin/starting location if mentioned.
- summary (string): One short sentence summarizing the user's intent in English.

User query: "{query}"

Respond ONLY with valid JSON, no markdown, no explanation.
"""

GEMINI_RESPONSE_PROMPT = """\
You are a friendly travel assistant. The user asked about departure timing.
Based on the analysis results below, write a helpful and concise response.
Respond in English only.
Include key information without repeating the same point:
- recommended leave time and why
- drive time and traffic condition
- security/check-in impact
- delay context (if available)
- one actionable next step
Use ONLY facts in the Results section below.
Do NOT invent locations, transit routes, train/subway/bus lines, or extra assumptions.
If information is missing, say it is not available.
Keep the tone natural and practical.
Limit to 4-6 short sentences (or bullet-like lines). No markdown headings/tables.

User query: "{query}"
Results:
- Recommended leave time: {leave_time}
- Drive time: {drive_minutes} min ({traffic_level} traffic)
- Airport: {airport_code}
- Security wait: {security_minutes} min
- Check-in: {checkin_minutes} min
- Flight departure: {departure_time}
- Flight position: {flight_position}

Response:
"""


def _init_gemini(use_gemini: bool = False) -> Optional[Any]:
    """Initialize Gemini model. Returns model instance or None."""
    if not use_gemini:
        return None
    if not HAS_GEMINI:
        return None

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model
    except Exception:
        return None


def gemini_parse_query(model: Any, query: str) -> Optional[Dict[str, Any]]:
    """Use Gemini to parse the user query into structured data."""
    if model is None:
        return None

    try:
        prompt = GEMINI_PARSE_PROMPT.format(query=query)
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        import json
        parsed = json.loads(text)
        return parsed
    except Exception:
        return None


def gemini_generate_response(
    model: Any,
    query: str,
    brief: Dict[str, Any],
) -> Optional[str]:
    """Use Gemini to generate a natural-language response from the results."""
    if model is None:
        return None

    fp = brief.get("flight_position")
    fp_str = "Not available"
    if fp:
        fp_str = f"{fp.get('flight_number', 'N/A')} @ {fp['lat']:.2f}, {fp['lon']:.2f}"

    try:
        prompt = GEMINI_RESPONSE_PROMPT.format(
            query=query,
            leave_time=brief.get("leave_dt", datetime.now()).strftime("%H:%M"),
            drive_minutes=brief.get("drive_minutes", "?"),
            traffic_level=brief.get("traffic_level", "unknown"),
            airport_code=brief.get("airport_code", "N/A"),
            security_minutes=brief.get("security_minutes", "?"),
            checkin_minutes=brief.get("checkin_minutes", "?"),
            departure_time=brief.get("departure_dt", datetime.now()).strftime("%H:%M"),
            flight_position=fp_str,
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  TRANSFORMER DELAY PREDICTOR (OPTIONAL)
# ═══════════════════════════════════════════════════════════════

if HAS_TRANSFORMER_DEPS:
    class _FeatureTokenizer(nn.Module):
        """Tokenize tabular features for FT-Transformer inference."""

        def __init__(self, num_features: int, d_token: int) -> None:
            super().__init__()
            self.feature_projections = nn.ModuleList(
                [nn.Linear(1, d_token) for _ in range(num_features)]
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.size(0)
            tokens: List[torch.Tensor] = []
            for i, projection in enumerate(self.feature_projections):
                feature_val = x[:, i].unsqueeze(-1)
                token = projection(feature_val)
                tokens.append(token.unsqueeze(1))
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            return torch.cat([cls_tokens] + tokens, dim=1)


    class _FTTransformer(nn.Module):
        """Compatible FT-Transformer architecture for loaded checkpoints."""

        def __init__(
            self,
            num_features: int,
            d_token: int = 64,
            n_blocks: int = 3,
            attention_heads: int = 8,
            ffn_d_hidden: int = 256,
            ffn_dropout: float = 0.1,
            residual_dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.feature_tokenizer = _FeatureTokenizer(num_features, d_token)
            self.blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_token,
                        nhead=attention_heads,
                        dim_feedforward=ffn_d_hidden,
                        dropout=residual_dropout,
                        activation="gelu",
                        batch_first=True,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Sequential(
                nn.Linear(d_token, ffn_d_hidden),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_d_hidden, ffn_d_hidden // 2),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(ffn_d_hidden // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.feature_tokenizer(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return self.head(x[:, 0, :])


    class _FlightDelayTransformer(nn.Module):
        """Fallback architecture for legacy checkpoints."""

        def __init__(
            self,
            input_dim: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.embedding = nn.Linear(input_dim, d_model)
            self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embedding(x)
            x = x.unsqueeze(1)
            x = x + self.pos_encoder
            x = self.transformer(x)
            x = x.squeeze(1)
            return self.fc(x)


    class _TransformerDelayPredictor:
        """Load and run flight delay model from departure_prediction assets."""

        def __init__(self, model_path: str) -> None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._load_model(model_path)

        def _load_model(self, model_path: str) -> None:
            # Some saved packages embed CUDA tensors. Force CPU remap when loading.
            orig_loader = torch.storage._load_from_bytes
            torch.storage._load_from_bytes = (
                lambda b: torch.load(io.BytesIO(b), map_location=self.device, weights_only=False)
            )
            try:
                with open(model_path, "rb") as f:
                    package = pickle.load(f)
            finally:
                torch.storage._load_from_bytes = orig_loader

            config = package["model_config"]
            model_type = config.get("model_type", "")
            if model_type == "FTTransformer":
                model = _FTTransformer(
                    num_features=config["num_features"],
                    d_token=config["d_token"],
                    n_blocks=config["n_blocks"],
                    attention_heads=config["attention_heads"],
                    ffn_d_hidden=config["ffn_d_hidden"],
                    ffn_dropout=config["ffn_dropout"],
                    residual_dropout=config["residual_dropout"],
                )
            else:
                model = _FlightDelayTransformer(
                    input_dim=config["input_dim"],
                    d_model=config["d_model"],
                    nhead=config["nhead"],
                    num_layers=config["num_layers"],
                    dropout=config["dropout"],
                )

            model.load_state_dict(package["model_state_dict"])
            model.eval()

            self.model = model.to(self.device)
            self.label_encoders = package["label_encoders"]
            self.scaler = package["scaler"]
            self.feature_columns = package["feature_columns"]

        def predict_delay(
            self,
            airline_code: str,
            origin: str,
            dest: str,
            flight_datetime: datetime,
        ) -> float:
            features: Dict[str, Any] = {
                "op_unique_carrier": airline_code,
                "origin": origin,
                "dest": dest,
                "hour": flight_datetime.hour,
                "month": flight_datetime.month,
                "day_of_week": flight_datetime.weekday(),
                "day_of_month": flight_datetime.day,
                "is_weekend": 1 if flight_datetime.weekday() >= 5 else 0,
            }

            encoded = features.copy()
            for col in ["op_unique_carrier", "origin", "dest"]:
                if col in self.label_encoders:
                    try:
                        encoded[col] = self.label_encoders[col].transform([features[col]])[0]
                    except ValueError:
                        encoded[col] = 0
                else:
                    encoded[col] = 0

            x = np.array([[encoded[col] for col in self.feature_columns]], dtype=float)
            numeric_cols = ["hour", "month", "day_of_week", "day_of_month"]
            numeric_idx = [
                self.feature_columns.index(col)
                for col in numeric_cols
                if col in self.feature_columns
            ]
            if numeric_idx:
                x[:, numeric_idx] = self.scaler.transform(x[:, numeric_idx])

            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                pred = self.model(x_tensor).cpu().numpy()[0][0]
            return float(pred)


@st.cache_resource(show_spinner=False)
def _load_transformer_delay_predictor(model_path: str) -> Optional[Any]:
    """Cache delay predictor so model weights load only once per Streamlit process."""
    if not HAS_TRANSFORMER_DEPS:
        return None
    return _TransformerDelayPredictor(model_path)


def _init_transformer_predictor(config: Dict[str, Any]) -> Tuple[Optional[Any], str]:
    """
    Initialize optional FT-Transformer predictor.
    Returns (predictor_or_none, status_string).
    """
    use_transformer = _resolve_bool_setting(
        config=config,
        config_key="use_transformer",
        env_key="USE_TRANSFORMER",
        default=False,
    )
    if not use_transformer:
        return None, "disabled"
    if not HAS_TRANSFORMER_DEPS:
        return None, "deps_missing"

    model_path_raw = os.environ.get("TRANSFORMER_MODEL_PATH", "").strip()
    if not model_path_raw:
        model_path_raw = str(config.get("transformer_model_path", "")).strip()
    if not model_path_raw:
        model_path_raw = str(DEFAULT_TRANSFORMER_MODEL_PATH)

    model_path = Path(model_path_raw)
    if not model_path.exists():
        return None, f"model_missing:{model_path}"

    try:
        predictor = _load_transformer_delay_predictor(str(model_path))
    except Exception:
        return None, "load_failed"
    if predictor is None:
        return None, "load_failed"
    return predictor, "ready"


def _extract_airline_code(flight_number: str) -> str:
    """Extract airline prefix from flight number, e.g. KE937 -> KE."""
    text = (flight_number or "").strip().upper()
    m = re.match(r"([A-Z]{1,3})", text)
    return m.group(1) if m else "UNK"


def _fetch_tsa_wait_minutes(
    airport_code: str,
    departure_dt: datetime,
    has_tsa_precheck: bool,
    terminal: str = "",
) -> Tuple[Optional[int], str]:
    """Try live/statistical TSA wait lookup from departure_prediction utils."""
    if not HAS_AIRPORT_INTEL:
        return None, "unavailable"
    try:
        raw = get_tsa_wait_time(
            airport_code=airport_code,
            departure_time=departure_dt,
            has_precheck=has_tsa_precheck,
            terminal=(terminal.strip() or None),
        )
        if isinstance(raw, dict):
            wait = raw.get("wait_time")
            if wait is None:
                return None, str(raw.get("source", "unknown"))
            return int(wait), str(raw.get("source", "unknown"))
        return int(raw), "statistical"
    except Exception:
        return None, "error"


def _fetch_gate_walk_minutes(terminal: str = "", gate: str = "") -> Tuple[Optional[int], str]:
    """Try gate-walk lookup from departure_prediction utils."""
    if not HAS_AIRPORT_INTEL:
        return None, "unavailable"
    term_text = terminal.strip() if terminal else ""
    gate_text = gate.strip() if gate else ""
    if not term_text and not gate_text:
        return None, "no_terminal"
    if not term_text:
        term_text = "Terminal 4"
    try:
        minutes = int(get_gate_walk_time(term_text, gate_text or None))
        return minutes, "gate_model"
    except Exception:
        return None, "error"


@st.cache_resource(show_spinner=False)
def _load_operational_analyzer() -> Optional[Any]:
    """Cache operational analyzer instance across Streamlit reruns."""
    if not HAS_OPERATIONAL_FACTORS:
        return None
    try:
        return OperationalFactorsAnalyzer()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_jfk_congestion_checker() -> Optional[Any]:
    """Cache JFK congestion checker instance across Streamlit reruns."""
    if not HAS_OPERATIONAL_FACTORS:
        return None
    try:
        return JFKCongestionChecker()
    except Exception:
        return None


def _estimate_operational_delay(
    flight_number: str,
    origin: str,
    dest: str,
    scheduled_departure_dt: datetime,
) -> Dict[str, Any]:
    """
    Estimate additional delay from operational factors:
      1) JFK congestion
      2) FR24 + ADS-B previous-leg delay
    """
    result: Dict[str, Any] = {
        "operational_delay_minutes": 0,
        "congestion_delay_minutes": 0,
        "congestion_level": "unknown",
        "congestion_score": 0.0,
        "congestion_sample_size": 0,
        "congestion_source": "none",
        "congestion_details": {},
        "adsb_fr24_delay_minutes": 0,
        "adsb_fr24_source": "none",
        "adsb_fr24_reason": "not_started",
        "adsb_fr24_found": False,
        "adsb_fr24_in_air": False,
        "adsb_fr24_validation_mismatch": False,
        "adsb_fr24_validation_notes": [],
        "operational_notes": [],
    }

    origin_code = (origin or "").strip().upper()
    dest_code = (dest or "").strip().upper()
    flight_no = (flight_number or "").strip().upper()

    # ── A) JFK congestion ─────────────────────────────
    if origin_code == "JFK":
        checker = _load_jfk_congestion_checker()
        analyzer = _load_operational_analyzer()
        congestion_data: Optional[Dict[str, Any]] = None

        if checker is not None:
            try:
                rui_result = checker.check_rui_congestion(
                    max_age_seconds=300,
                    hour=scheduled_departure_dt.hour,
                )
                if isinstance(rui_result, dict):
                    congestion_data = rui_result
            except Exception as exc:
                result["operational_notes"].append(f"RUI congestion unavailable ({exc})")

        # Use crawler output flights_YYYYMMDD.json directly (requested behavior).
        if congestion_data is None and checker is not None:
            try:
                flights_json_result = checker.check_flights_json_congestion(
                    max_age_seconds=0,  # use latest saved flights_*.json without strict staleness cut
                    hour=scheduled_departure_dt.hour,
                )
                if isinstance(flights_json_result, dict):
                    congestion_data = flights_json_result
            except Exception as exc:
                result["operational_notes"].append(f"flights_*.json congestion unavailable ({exc})")

        if congestion_data is None and analyzer is not None and bool(getattr(analyzer, "enabled", False)):
            try:
                api_result = analyzer.get_jfk_area_congestion(scheduled_departure_dt)
                if isinstance(api_result, dict):
                    congestion_data = api_result
            except Exception as exc:
                result["operational_notes"].append(f"AviationStack congestion unavailable ({exc})")

        if congestion_data is None and checker is not None:
            try:
                fallback_result = checker.check_realtime_congestion(collect_seconds=5)
                if isinstance(fallback_result, dict):
                    congestion_data = fallback_result
            except Exception as exc:
                result["operational_notes"].append(f"SBS congestion unavailable ({exc})")

        if isinstance(congestion_data, dict):
            raw_delay = congestion_data.get("recommended_extra_delay", 0)
            try:
                congestion_delay = max(0, int(raw_delay or 0))
            except Exception:
                congestion_delay = 0

            try:
                score = float(congestion_data.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0

            try:
                sample_size = int(congestion_data.get("sample_size", 0) or 0)
            except Exception:
                sample_size = 0

            result["congestion_delay_minutes"] = congestion_delay
            result["congestion_level"] = str(congestion_data.get("level", "unknown"))
            result["congestion_score"] = score
            result["congestion_sample_size"] = sample_size
            result["congestion_source"] = str(congestion_data.get("source", "unknown"))
            result["congestion_details"] = congestion_data.get("details", {}) or {}

    # ── B) FR24 + ADS-B previous-leg delay ───────────
    if flight_no and HAS_PREVIOUS_LEG_ESTIMATOR:
        try:
            adsb_info = estimate_previous_leg_delay_minutes(
                flight_no=flight_no,
                expected_origin=origin_code or None,
                expected_dest=dest_code or None,
                expected_date=scheduled_departure_dt,
                collect_time=5,
            )
            if isinstance(adsb_info, dict):
                found = bool(adsb_info.get("found", False))
                in_air = bool(adsb_info.get("in_air", False))
                raw_source = str(adsb_info.get("source", "none"))
                raw_reason = str(adsb_info.get("reason", "unknown"))

                result["adsb_fr24_found"] = found
                result["adsb_fr24_in_air"] = in_air
                result["adsb_fr24_validation_mismatch"] = bool(
                    adsb_info.get("validation_mismatch", False)
                )
                notes = adsb_info.get("validation_notes", []) or []
                result["adsb_fr24_validation_notes"] = [str(x) for x in notes]

                # Policy: only use previous-leg signal when previous leg is currently in the air.
                if found and in_air:
                    raw_delay = adsb_info.get("delay_minutes", 0)
                    try:
                        adsb_delay = max(0, int(raw_delay or 0))
                    except Exception:
                        adsb_delay = 0
                    result["adsb_fr24_delay_minutes"] = adsb_delay
                    result["adsb_fr24_source"] = raw_source
                    result["adsb_fr24_reason"] = raw_reason
                else:
                    result["adsb_fr24_delay_minutes"] = 0
                    result["adsb_fr24_source"] = "not_applied"
                    result["adsb_fr24_reason"] = "previous_leg_not_in_air"
        except Exception as exc:
            result["operational_notes"].append(f"FR24+ADS-B previous-leg unavailable ({exc})")
    elif flight_no:
        result["operational_notes"].append("FR24+ADS-B estimator module unavailable")

    result["operational_delay_minutes"] = (
        int(result["congestion_delay_minutes"]) + int(result["adsb_fr24_delay_minutes"])
    )
    return result


def _normalize_ticket_time_text(value: str) -> str:
    """Normalize ticket datetime text into YYYY-MM-DD HH:MM when possible."""
    text = (value or "").strip()
    if not text:
        return ""

    # Try strict known formats first
    known_formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
    ]
    for fmt in known_formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue

    # Fallback: ISO text with timezone, if any
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M")
    except Exception:
        pass

    # No-year ticket formats -> assume current year
    current_year = datetime.now().year
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
            return parsed.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue

    return text


def _normalize_ticket_airport_text(value: str) -> str:
    """Normalize OCR airport field into a 3-letter IATA code when possible."""
    text = str(value or "").strip().upper()
    if not text:
        return ""

    # Preferred: explicit token boundary (e.g. "JFK", "to DCA").
    m = re.search(r"\b([A-Z]{3})\b", text)
    if m:
        return m.group(1)

    # Fallback: if text only contains exactly 3 letters after cleanup.
    letters = re.sub(r"[^A-Z]", "", text)
    if len(letters) == 3:
        return letters
    return ""


def _normalize_ticket_data(raw: Dict[str, Any]) -> Dict[str, str]:
    """Clean Gemini OCR output into safe UI-friendly strings."""
    if not isinstance(raw, dict):
        return {}

    fields = [
        "flight_number",
        "airline",
        "departure_airport",
        "arrival_airport",
        "departure_time",
        "arrival_time",
        "passenger_name",
        "seat",
        "gate",
        "terminal",
        "baggage_allowance",
        "checked_baggage",
    ]
    out: Dict[str, str] = {}
    for key in fields:
        val = raw.get(key)
        out[key] = "" if val is None else str(val).strip()

    out["flight_number"] = re.sub(r"\s+", "", out["flight_number"]).upper()
    out["departure_airport"] = _normalize_ticket_airport_text(out["departure_airport"])
    out["arrival_airport"] = _normalize_ticket_airport_text(out["arrival_airport"])
    out["departure_time"] = _normalize_ticket_time_text(out["departure_time"])
    out["arrival_time"] = _normalize_ticket_time_text(out["arrival_time"])
    out["checked_baggage"] = out["checked_baggage"].strip().lower()
    return out


def _infer_checked_baggage(ticket_data: Dict[str, Any]) -> Optional[bool]:
    """
    Infer checked-baggage requirement from OCR fields.
    Returns True/False when confident, otherwise None.
    """
    explicit = str(ticket_data.get("checked_baggage", "")).strip().lower()
    if explicit in {"true", "yes", "y", "1"}:
        return True
    if explicit in {"false", "no", "n", "0"}:
        return False

    allowance = str(ticket_data.get("baggage_allowance", "")).strip().lower()
    if not allowance:
        return None

    # Clear "no checked baggage" signals.
    no_patterns = [
        r"\b0\s*(pc|piece|pieces|bag|bags|kg)\b",
        r"\bnil\b",
        r"\bnone\b",
        r"no\s*checked",
        r"carry[-\s]*on\s*only",
        r"cabin\s*only",
        r"hand\s*baggage\s*only",
    ]
    for p in no_patterns:
        if re.search(p, allowance):
            return False

    # Positive checked-baggage signals.
    yes_patterns = [
        r"\b[1-9]\d*\s*(pc|piece|pieces|bag|bags)\b",
        r"checked\s*baggage",
        r"bag\s*drop",
    ]
    for p in yes_patterns:
        if re.search(p, allowance):
            return True

    return None


def _build_ticket_query(ticket_data: Dict[str, Any]) -> str:
    """Create a useful departure-intent query from ticket OCR fields."""
    flight = str(ticket_data.get("flight_number", "")).strip() or "my flight"
    airport = str(ticket_data.get("departure_airport", "")).strip() or "the airport"
    dep_time = str(ticket_data.get("departure_time", "")).strip()
    if dep_time:
        return (
            f"When should I leave for {airport}? "
            f"My flight {flight} departs at {dep_time}."
        )
    return f"When should I leave for {airport}? Flight {flight}."


def _extract_ticket_with_gemini(
    uploaded_file: Any,
    use_gemini: bool,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Extract structured ticket info using Gemini Vision.
    Returns (ticket_data, error_message).
    """
    if uploaded_file is None:
        return None, "Upload a ticket image first."
    if not use_gemini:
        return None, "Ticket OCR requires `USE_GEMINI=true`."
    if not HAS_GEMINI_TICKET_OCR:
        return None, "Gemini ticket OCR module is unavailable in this environment."
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "GEMINI_API_KEY is not set."

    suffix = Path(getattr(uploaded_file, "name", "ticket.png")).suffix or ".png"
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp.name)

        ocr = GeminiTicketOCR(api_key=api_key)
        raw = ocr.extract_with_vision(str(tmp_path))
    except Exception as exc:
        return None, f"Ticket analysis failed: {exc}"
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    ticket = _normalize_ticket_data(raw if isinstance(raw, dict) else {})
    if not ticket:
        return None, "Gemini returned no ticket fields."

    key_fields = (
        ticket.get("flight_number"),
        ticket.get("departure_airport"),
        ticket.get("departure_time"),
    )
    if not any(key_fields):
        return None, "Ticket parsed, but key fields are missing (flight/airport/time)."

    return ticket, None


# ═══════════════════════════════════════════════════════════════
#  CORE LOGIC
# ═══════════════════════════════════════════════════════════════

def _safe_float(val: str) -> Optional[float]:
    """Parse a string to float, returning None on failure."""
    if not val or not val.strip():
        return None
    try:
        return float(val.strip())
    except (ValueError, TypeError):
        return None


def process_departure_query(
    query: str,
    config: Dict[str, Any],
    origin: str = "",
    airport_code: str = "",
    flight_departure_local: str = "",
    route: str = "",
    flight_number: str = "",
    has_checked_baggage: bool = True,
    has_tsa_precheck: bool = False,
    terminal: str = "",
    gate: str = "",
    flight_lat: Optional[float] = None,
    flight_lon: Optional[float] = None,
    flight_alt: Optional[float] = None,
    flight_speed: Optional[float] = None,
    force_intent: bool = True,
) -> Dict[str, Any]:
    """
    Process a departure query and return structured results.

    This reimplements the logic from departure_brief.py's main() but
    returns a rich dictionary instead of printing key=value lines.
    """
    if not HAS_DEPARTURE_BRIEF:
        return {"error": "departure_brief module not available"}

    result: Dict[str, Any] = {}
    notes: List[str] = []

    terminal = (terminal or "").strip()
    gate = (gate or "").strip().upper()
    result["has_checked_baggage"] = bool(has_checked_baggage)
    result["has_tsa_precheck"] = bool(has_tsa_precheck)
    result["terminal"] = terminal
    result["gate"] = gate

    use_gemini = _resolve_bool_setting(
        config=config,
        config_key="use_gemini",
        env_key="USE_GEMINI",
        default=False,
    )
    use_transformer = _resolve_bool_setting(
        config=config,
        config_key="use_transformer",
        env_key="USE_TRANSFORMER",
        default=False,
    )
    result["use_gemini"] = use_gemini
    result["use_transformer"] = use_transformer

    # ── Gemini AI parsing (if available) ──────────────
    gemini_model = _init_gemini(use_gemini=use_gemini)
    gemini_data: Optional[Dict[str, Any]] = None
    result["ai_model"] = "rule-based"
    if gemini_model and query:
        gemini_data = gemini_parse_query(gemini_model, query)
        result["gemini_parsed"] = gemini_data
        result["ai_model"] = "gemini-2.0-flash"
    elif use_gemini:
        notes.append("USE_GEMINI enabled, but Gemini client is unavailable")

    # ── Intent detection ──────────────────────────────
    if gemini_data and not force_intent:
        intent = bool(gemini_data.get("is_departure_query", False))
    else:
        intent = force_intent or detect_departure_intent(query)
    result["intent"] = intent
    if not intent:
        result["notes"] = ["Not a departure-intent query"] + notes
        # Store Gemini summary even for non-departure queries
        if gemini_data and gemini_data.get("summary"):
            result["gemini_summary"] = gemini_data["summary"]
        return result

    # ── Timezone & current time ───────────────────────
    tz_name: str = config.get("timezone", "America/New_York")
    now_local = _get_now_local(tz_name)
    result["timezone"] = tz_name
    result["now_local"] = now_local

    # ── Origin (Gemini can override) ──────────────────
    if not origin and gemini_data and gemini_data.get("origin"):
        origin = gemini_data["origin"]
    if not origin:
        origin = config.get("default_origin", "Times Square, Manhattan, NY")
    result["origin"] = origin

    # ── Airport code resolution (Gemini-enhanced) ─────
    airport_query_map: Dict[str, str] = config.get("airport_query_by_code", {})
    sec_map: Dict[str, int] = config.get("security_wait_minutes_by_airport", {})
    known_codes: set = set(airport_query_map.keys()) | set(sec_map.keys())

    # Priority: sidebar input > Gemini > query regex > default
    if not airport_code and gemini_data and gemini_data.get("airport_code"):
        airport_code = gemini_data["airport_code"].strip().upper()
    if not airport_code:
        if route:
            airport_code = (_extract_airport_from_route(route) or "").upper()
        elif query:
            airport_code = (
                _extract_airport_from_query(query, known_codes=known_codes) or ""
            ).upper()
    if not airport_code:
        airport_code = str(config.get("default_airport", "JFK")).upper()
    airport_code = _normalize_airport_code(airport_code, config, known_codes)
    result["airport_code"] = airport_code

    destination = airport_query_map.get(airport_code, f"{airport_code} Airport")
    result["destination"] = destination

    # ── Flight number (Gemini-enhanced) ──────────────
    if not flight_number and gemini_data and gemini_data.get("flight_number"):
        flight_number = gemini_data["flight_number"].strip().upper()
    if not flight_number and query:
        extracted = _extract_flight_number_from_query(query)
        if extracted:
            flight_number = extracted
    flight_number = (flight_number or "").strip().upper()
    result["flight_number"] = flight_number

    # ── Flight departure time (Gemini-enhanced) ───────
    departure_dt: Optional[datetime] = None
    dt_source = "default_lead"

    if flight_departure_local:
        departure_dt = _parse_departure_time(flight_departure_local, now_local)
        if departure_dt:
            dt_source = "user_input"

    if departure_dt is None and gemini_data and gemini_data.get("departure_time"):
        departure_dt = _parse_departure_time(gemini_data["departure_time"], now_local)
        if departure_dt:
            dt_source = "gemini_parsed"

    if departure_dt is None and query:
        departure_dt = _extract_time_from_query(query, now_local)
        if departure_dt:
            dt_source = "query_text"

    if departure_dt is None:
        lead = int(config.get("default_flight_lead_minutes", 180))
        departure_dt = now_local + timedelta(minutes=lead)

    scheduled_departure_dt = departure_dt
    result["scheduled_departure_dt"] = scheduled_departure_dt

    # ── Optional: Transformer delay prediction ────────
    model_predicted_delay_minutes: Optional[float] = None
    predicted_delay_minutes: Optional[float] = None
    delay_source = "none"
    if use_transformer:
        predictor, transformer_status = _init_transformer_predictor(config)
        result["transformer_status"] = transformer_status
        if predictor is None:
            if transformer_status != "disabled":
                notes.append(f"Transformer predictor unavailable ({transformer_status})")
        else:
            airline_code = _extract_airline_code(flight_number)
            transformer_dest = (
                str(config.get("transformer_default_destination", "UNK")).strip().upper()
            )
            if not transformer_dest:
                transformer_dest = "UNK"
            try:
                raw_delay = predictor.predict_delay(
                    airline_code=airline_code,
                    origin=airport_code,
                    dest=transformer_dest,
                    flight_datetime=scheduled_departure_dt,
                )
                delay_val = float(raw_delay)
                if -1200.0 < delay_val < 1200.0:
                    # Policy: never output negative delay. 0 or less -> 0.
                    if delay_val <= 0:
                        delay_val = 0.0
                    else:
                        delay_val = min(360.0, delay_val)
                    model_predicted_delay_minutes = round(delay_val, 1)
                    predicted_delay_minutes = model_predicted_delay_minutes
                    result["predicted_delay_minutes"] = predicted_delay_minutes
                    delay_source = "ft_transformer"
                    if delay_val >= 1.0:
                        departure_dt = scheduled_departure_dt + timedelta(minutes=delay_val)
                        dt_source = f"{dt_source}+transformer_delay"
                    if result.get("ai_model") != "gemini-2.0-flash":
                        result["ai_model"] = "transformer+rule-based"
                else:
                    notes.append("Transformer predicted an out-of-range delay; ignored")
            except Exception:
                notes.append("Transformer delay prediction failed")
    else:
        result["transformer_status"] = "disabled"

    result["model_predicted_delay_minutes"] = model_predicted_delay_minutes

    # ── Optional: Operational delay (JFK congestion + FR24/ADS-B) ─────────
    use_operational_factors = _resolve_bool_setting(
        config=config,
        config_key="use_operational_factors",
        env_key="USE_OPERATIONAL_FACTORS",
        default=True,
    )
    result["use_operational_factors"] = use_operational_factors

    operational = {
        "operational_delay_minutes": 0,
        "congestion_delay_minutes": 0,
        "congestion_level": "unknown",
        "congestion_score": 0.0,
        "congestion_sample_size": 0,
        "congestion_source": "none",
        "congestion_details": {},
        "adsb_fr24_delay_minutes": 0,
        "adsb_fr24_source": "none",
        "adsb_fr24_reason": "not_started",
        "adsb_fr24_found": False,
        "adsb_fr24_in_air": False,
        "adsb_fr24_validation_mismatch": False,
        "adsb_fr24_validation_notes": [],
        "operational_notes": [],
    }
    if use_operational_factors:
        operational = _estimate_operational_delay(
            flight_number=flight_number,
            origin=airport_code,
            dest=str(config.get("transformer_default_destination", "UNK")).strip().upper() or "UNK",
            scheduled_departure_dt=scheduled_departure_dt,
        )
        for note in operational.get("operational_notes", []) or []:
            notes.append(str(note))

    result.update(
        {
            "operational_delay_minutes": int(operational.get("operational_delay_minutes", 0) or 0),
            "congestion_delay_minutes": int(operational.get("congestion_delay_minutes", 0) or 0),
            "congestion_level": str(operational.get("congestion_level", "unknown")),
            "congestion_score": float(operational.get("congestion_score", 0.0) or 0.0),
            "congestion_sample_size": int(operational.get("congestion_sample_size", 0) or 0),
            "congestion_source": str(operational.get("congestion_source", "none")),
            "congestion_details": operational.get("congestion_details", {}) or {},
            "adsb_fr24_delay_minutes": int(operational.get("adsb_fr24_delay_minutes", 0) or 0),
            "adsb_fr24_source": str(operational.get("adsb_fr24_source", "none")),
            "adsb_fr24_reason": str(operational.get("adsb_fr24_reason", "unknown")),
            "adsb_fr24_found": bool(operational.get("adsb_fr24_found", False)),
            "adsb_fr24_in_air": bool(operational.get("adsb_fr24_in_air", False)),
            "adsb_fr24_validation_mismatch": bool(
                operational.get("adsb_fr24_validation_mismatch", False)
            ),
            "adsb_fr24_validation_notes": operational.get("adsb_fr24_validation_notes", []) or [],
        }
    )

    # Always aggregate numeric delay components:
    # total delay = model delay + operational delay(congestion + previous-leg), clamped at >= 0.
    operational_delay_minutes = float(result["operational_delay_minutes"] or 0.0)
    base_delay = float(model_predicted_delay_minutes or 0.0)
    total_delay = max(0.0, base_delay + operational_delay_minutes)
    predicted_delay_minutes = round(total_delay, 1)
    result["predicted_delay_minutes"] = predicted_delay_minutes

    if model_predicted_delay_minutes is not None:
        delay_source = "ft_transformer"
    if operational_delay_minutes > 0:
        delay_source = "operational_factors" if delay_source == "none" else f"{delay_source}+operational_factors"
        dt_source = f"{dt_source}+operational_delay"

    if predicted_delay_minutes >= 1.0:
        departure_dt = scheduled_departure_dt + timedelta(minutes=predicted_delay_minutes)
        if model_predicted_delay_minutes is not None and "transformer_delay" not in dt_source:
            dt_source = f"{dt_source}+transformer_delay"

    result["departure_dt"] = departure_dt
    result["departure_time_source"] = dt_source
    result["delay_source"] = delay_source

    # ── Time components ───────────────────────────────
    security_minutes = int(
        sec_map.get(airport_code, config.get("default_security_wait_minutes", 30))
    )
    security_source = "config"

    # Check-in & bag-drop: profile-based (checked baggage -> time required)
    baggage_check_default = int(
        config.get(
            "default_baggage_check_minutes",
            config.get("default_checkin_minutes", 45),
        )
    )
    checkin_minutes = baggage_check_default if has_checked_baggage else 0
    checkin_source = "baggage_profile"

    walk_minutes = int(config.get("default_walk_to_gate_minutes", 15))
    walk_source = "config"
    buffer_minutes = int(config.get("default_buffer_minutes", 10))

    use_live_airport_times = _resolve_bool_setting(
        config=config,
        config_key="use_live_airport_times",
        env_key="USE_LIVE_AIRPORT_TIMES",
        default=True,
    )
    result["use_live_airport_times"] = use_live_airport_times

    if use_live_airport_times and departure_dt is not None:
        live_tsa, live_tsa_source = _fetch_tsa_wait_minutes(
            airport_code=airport_code,
            departure_dt=departure_dt,
            has_tsa_precheck=has_tsa_precheck,
            terminal=terminal,
        )
        if isinstance(live_tsa, int) and live_tsa >= 0:
            security_minutes = live_tsa
            security_source = live_tsa_source
        elif live_tsa_source not in {"unavailable", "no_terminal"}:
            notes.append(f"Live TSA wait unavailable ({live_tsa_source})")

        live_walk, live_walk_source = _fetch_gate_walk_minutes(terminal=terminal, gate=gate)
        if isinstance(live_walk, int) and live_walk >= 0:
            walk_minutes = live_walk
            walk_source = live_walk_source
        elif live_walk_source == "error":
            notes.append("Gate walk lookup failed; using default")

    result["security_minutes"] = security_minutes
    result["checkin_minutes"] = checkin_minutes
    result["walk_minutes"] = walk_minutes
    result["buffer_minutes"] = buffer_minutes
    result["security_source"] = security_source
    result["checkin_source"] = checkin_source
    result["walk_source"] = walk_source

    # ── Drive info (Google Maps) ──────────────────────
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        api_key = str(config.get("google_maps_api_key", "")).strip()
    result["has_google_api_key"] = bool(api_key)

    timeout_sec = int(config.get("traffic_request_timeout_seconds", 6))
    drive_minutes, traffic_level, drive_source = fetch_drive_info(
        origin=origin,
        destination=destination,
        api_key=api_key,
        timeout_seconds=timeout_sec,
    )
    if drive_minutes <= 0:
        drive_minutes = int(config.get("fallback_drive_minutes", 55))
        drive_source = "fallback"

    result["drive_minutes"] = drive_minutes
    result["traffic_level"] = traffic_level
    result["drive_source"] = drive_source

    # ── Calculate key timestamps ──────────────────────
    # Policy: airport process timeline is anchored to scheduled departure
    # (more conservative and easier to reason about for users).
    airport_process_anchor_dt = scheduled_departure_dt
    result["airport_process_basis"] = "scheduled_departure"
    result["airport_process_anchor_dt"] = airport_process_anchor_dt

    total_airport_minutes = security_minutes + checkin_minutes + walk_minutes + buffer_minutes
    airport_arrival_dt = airport_process_anchor_dt - timedelta(minutes=total_airport_minutes)
    leave_dt = airport_arrival_dt - timedelta(minutes=drive_minutes)

    result["total_airport_minutes"] = total_airport_minutes
    result["airport_arrival_dt"] = airport_arrival_dt
    result["leave_dt"] = leave_dt

    # ── Flight position ───────────────────────────────
    flight_position: Optional[Dict[str, Any]] = None
    fp_source = "none"

    if flight_number and flight_lat is not None and flight_lon is not None:
        flight_position = {
            "lat": flight_lat,
            "lon": flight_lon,
            "alt": flight_alt,
            "speed": flight_speed,
            "flight_number": flight_number,
        }
        fp_source = "hooked_adsb"
    elif flight_number:
        av_key = os.environ.get("AVIATIONSTACK_API_KEY", "").strip()
        if not av_key:
            av_key = str(config.get("aviationstack_api_key", "")).strip()
        av_timeout = int(config.get("aviationstack_timeout_seconds", 6))

        live = fetch_aviationstack_position(
            flight_number=flight_number,
            api_key=av_key,
            timeout_seconds=av_timeout,
        )
        if live:
            flight_position = {
                "lat": live["lat"],
                "lon": live["lon"],
                "alt": live.get("altitude"),
                "speed": live.get("speed"),
                "flight_number": live.get("flight_iata", flight_number),
                "status": live.get("status", "unknown"),
                "airline": live.get("airline_name", ""),
            }
            fp_source = "aviationstack"

    result["flight_position"] = flight_position
    result["flight_position_source"] = fp_source

    # ── Google Maps URL ───────────────────────────────
    map_url = build_maps_url(origin, destination)
    if map_url:
        if "hl=" not in map_url:
            map_url += "&hl=en"
        if "gl=" not in map_url:
            map_url += "&gl=us"
    result["map_url"] = map_url
    result["google_maps_api_key"] = api_key

    # ── Notes ─────────────────────────────────────────
    if drive_source == "fallback":
        notes.append("Using fallback drive time (traffic API unavailable)")
    if not api_key:
        notes.append("Google Maps API key not set — traffic data unavailable")
    if flight_number and fp_source == "none":
        notes.append(f"Flight position for {flight_number} not found")
    result["notes"] = notes

    # ── Gemini natural-language response ──────────────
    if gemini_model:
        ai_response = gemini_generate_response(gemini_model, query, result)
        if ai_response:
            result["ai_response"] = ai_response

    return result


# ═══════════════════════════════════════════════════════════════
#  MAP RENDERING
# ═══════════════════════════════════════════════════════════════

def create_map(
    airport_code: str,
    flight_position: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Build a folium Map with airport marker and optional flight marker."""
    if not HAS_FOLIUM:
        return None

    airport_ll = AIRPORT_COORDS.get(airport_code)

    # Determine map centre
    if flight_position and airport_ll:
        clat = (flight_position["lat"] + airport_ll[0]) / 2
        clon = (flight_position["lon"] + airport_ll[1]) / 2
        zoom = 5
    elif airport_ll:
        clat, clon = airport_ll
        zoom = 9
    elif flight_position:
        clat, clon = flight_position["lat"], flight_position["lon"]
        zoom = 6
    else:
        return None

    m = folium.Map(location=[clat, clon], zoom_start=zoom, tiles="CartoDB positron")

    # ── Airport marker ────────────────────────────────
    if airport_ll:
        folium.Marker(
            location=list(airport_ll),
            popup=folium.Popup(
                f"<b>{airport_code} Airport</b>",
                max_width=200,
            ),
            tooltip=f"{airport_code} Airport",
            icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        ).add_to(m)

    # ── Flight position marker ────────────────────────
    if flight_position:
        popup_parts = [f"<b>Flight {flight_position.get('flight_number', 'N/A')}</b>"]
        if flight_position.get("alt") is not None:
            popup_parts.append(f"Alt: {flight_position['alt']:.0f} ft")
        if flight_position.get("speed") is not None:
            popup_parts.append(f"Speed: {flight_position['speed']:.0f} kt")
        if flight_position.get("status"):
            popup_parts.append(f"Status: {flight_position['status']}")
        if flight_position.get("airline"):
            popup_parts.append(f"Airline: {flight_position['airline']}")

        folium.Marker(
            location=[flight_position["lat"], flight_position["lon"]],
            popup=folium.Popup("<br>".join(popup_parts), max_width=250),
            tooltip=f"Flight {flight_position.get('flight_number', '')}",
            icon=folium.Icon(color="red", icon="plane", prefix="fa"),
        ).add_to(m)

        # Dashed line from flight to airport
        if airport_ll:
            folium.PolyLine(
                locations=[
                    [flight_position["lat"], flight_position["lon"]],
                    list(airport_ll),
                ],
                color="#FF5722",
                weight=2,
                dash_array="8",
                opacity=0.7,
            ).add_to(m)

    # ── Fit bounds ────────────────────────────────────
    pts: List[list] = []
    if airport_ll:
        pts.append(list(airport_ll))
    if flight_position:
        pts.append([flight_position["lat"], flight_position["lon"]])
    if len(pts) > 1:
        m.fit_bounds(pts, padding=(60, 60))

    return m


# ═══════════════════════════════════════════════════════════════
#  TIMELINE RENDERING
# ═══════════════════════════════════════════════════════════════

def _tl_item(
    time_str: str,
    label: str,
    duration: str,
    color: str,
    is_key: bool,
    is_last: bool,
    is_substep: bool = False,
) -> str:
    """Return HTML for a single timeline row."""
    dot_cls = "tl-dot key" if is_key else "tl-dot"
    if is_substep and not is_key:
        dot_cls += " substep"
    item_cls = "tl-item key-row" if is_key else "tl-item"
    glow = f"box-shadow:0 0 8px {color};" if is_key else ""
    line_html = "" if is_last else '<div class="tl-line"></div>'
    time_html = (
        f'<span class="tl-time">{time_str}</span>' if time_str
        else '<span class="tl-time"></span>'
    )
    dur_html = f'<span class="tl-dur">({duration})</span>' if duration else ""
    label_cls = "tl-label substep" if is_substep and not is_key else "tl-label"

    return (
        f'<div class="{item_cls}">'
        f'  <div style="position:relative;">'
        f'    <div class="{dot_cls}" style="background:{color};{glow}"></div>'
        f'    {line_html}'
        f'  </div>'
        f'  <div class="tl-body">'
        f'    {time_html}'
        f'    <span class="{label_cls}">{label}</span>'
        f'    {dur_html}'
        f'  </div>'
        f'</div>'
    )


def build_timeline_html(brief: Dict[str, Any]) -> str:
    """Generate the full departure timeline HTML."""
    leave_dt: Optional[datetime] = brief.get("leave_dt")
    if not leave_dt:
        return ""

    drive_min = brief.get("drive_minutes", 0)
    checkin_min = brief.get("checkin_minutes", 0)
    security_min = brief.get("security_minutes", 0)
    walk_min = brief.get("walk_minutes", 0)
    buffer_min = brief.get("buffer_minutes", 0)
    departure_dt: Optional[datetime] = brief.get("departure_dt")
    scheduled_departure_dt: Optional[datetime] = brief.get("scheduled_departure_dt")
    predicted_delay = brief.get("predicted_delay_minutes")
    airport_arrival_dt: Optional[datetime] = brief.get("airport_arrival_dt")
    traffic = brief.get("traffic_level", "unknown")
    traffic_emoji = TRAFFIC_EMOJI.get(traffic, "⚪")
    traffic_color = TRAFFIC_COLORS.get(traffic, "#78909C")
    has_schedule_shift = bool(
        scheduled_departure_dt
        and departure_dt
        and scheduled_departure_dt.strftime("%H:%M") != departure_dt.strftime("%H:%M")
    )

    steps: List[Dict[str, Any]] = [
        {
            "time": leave_dt.strftime("%H:%M"),
            "label": "🏠 Leave origin",
            "dur": "",
            "color": "#4CAF50",
            "key": True,
            "sub": False,
        },
        {
            "time": "",
            "label": f"🚗 Drive to airport ({traffic_emoji} {traffic})",
            "dur": f"{drive_min} min · {traffic} traffic",
            "color": traffic_color,
            "key": False,
            "sub": False,
        },
    ]

    if airport_arrival_dt:
        steps.append(
            {
                "time": airport_arrival_dt.strftime("%H:%M"),
                "label": "🏢 Arrive at airport",
                "dur": "",
                "color": "#2196F3",
                "key": True,
                "sub": False,
            }
        )

    steps.extend(
        [
            {
                "time": "",
                "label": "📋 Check-in & bag drop",
                "dur": f"{checkin_min} min",
                "color": "#9C27B0",
                "key": False,
                "sub": True,
            },
            {
                "time": "",
                "label": "🛡️ Security screening",
                "dur": f"{security_min} min",
                "color": "#FF9800",
                "key": False,
                "sub": True,
            },
            {
                "time": "",
                "label": "🚶 Walk to gate",
                "dur": f"{walk_min} min",
                "color": "#607D8B",
                "key": False,
                "sub": True,
            },
        ]
    )

    if buffer_min > 0:
        steps.append(
            {
                "time": "",
                "label": "⏳ Buffer time",
                "dur": f"{buffer_min} min",
                "color": "#795548",
                "key": False,
                "sub": True,
            }
        )

    departure_duration = ""
    if has_schedule_shift and scheduled_departure_dt:
        if isinstance(predicted_delay, (int, float)):
            departure_duration = (
                f"scheduled {scheduled_departure_dt.strftime('%H:%M')} "
                f"(+{float(predicted_delay):.0f} min delay)"
            )
        else:
            departure_duration = f"scheduled {scheduled_departure_dt.strftime('%H:%M')}"

    if departure_dt:
        steps.append(
            {
                "time": departure_dt.strftime("%H:%M"),
                "label": "✈️ Expected departure" if has_schedule_shift else "✈️ Flight departure",
                "dur": departure_duration,
                "color": "#F44336",
                "key": True,
                "sub": False,
            }
        )

    parts = ['<div class="tl-wrap">']
    for i, s in enumerate(steps):
        parts.append(
            _tl_item(
                time_str=s["time"],
                label=s["label"],
                duration=s["dur"],
                color=s["color"],
                is_key=s["key"],
                is_last=(i == len(steps) - 1),
                is_substep=bool(s.get("sub", False)),
            )
        )
    parts.append("</div>")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
#  GOOGLE MAPS EMBED
# ═══════════════════════════════════════════════════════════════

def render_google_maps_embed(
    origin: str,
    destination: str,
    api_key: str,
) -> None:
    """Render a Google Maps Embed API iframe showing the driving route."""
    if not api_key:
        return

    origin_enc = quote_plus(origin)
    dest_enc = quote_plus(destination)
    url = (
        f"https://www.google.com/maps/embed/v1/directions"
        f"?key={api_key}"
        f"&origin={origin_enc}"
        f"&destination={dest_enc}"
        f"&mode=driving"
        f"&language=en"
        f"&region=US"
    )
    st.components.v1.html(
        f'<iframe src="{url}" width="100%" height="380" '
        f'style="border:0;border-radius:12px;" loading="lazy" '
        f'allowfullscreen></iframe>',
        height=400,
    )


# ═══════════════════════════════════════════════════════════════
#  INFO PANELS
# ═══════════════════════════════════════════════════════════════

def render_flight_card(brief: Dict[str, Any]) -> None:
    """Flight information card."""
    fp = brief.get("flight_position")
    fn = brief.get("flight_number", "")
    source = brief.get("flight_position_source", "none")

    if not fn and not fp:
        return

    html = '<div class="info-card"><h4>🛩️ Flight Information</h4>'
    if fn:
        html += f'<p>Flight: <span class="val">{fn}</span></p>'
    if fp:
        html += f'<p>Position: <span class="val">{fp["lat"]:.4f}°, {fp["lon"]:.4f}°</span></p>'
        if fp.get("alt") is not None:
            html += f'<p>Altitude: <span class="val">{fp["alt"]:.0f} ft</span></p>'
        if fp.get("speed") is not None:
            html += f'<p>Speed: <span class="val">{fp["speed"]:.0f} kt</span></p>'
        if fp.get("status"):
            html += f'<p>Status: <span class="val">{fp["status"]}</span></p>'
        if fp.get("airline"):
            html += f'<p>Airline: <span class="val">{fp["airline"]}</span></p>'
        html += f'<p>Source: <span class="val">{source}</span></p>'
    elif fn:
        html += '<p style="color:#9e9e9e;">Flight position data not available</p>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _pretty_source_name(source: Any) -> str:
    """Convert internal source key to user-friendly label."""
    key = str(source or "unknown").strip().lower()
    mapping = {
        "ft_transformer": "AI model",
        "transformer": "AI model",
        "real_time": "Airline live update",
        "statistical": "Airport statistics",
        "config": "Default setting",
        "baggage_profile": "Your baggage choice",
        "gate_model": "Gate estimate model",
        "flights_json": "Saved ADS-B flight data",
        "aviationstack_api": "AviationStack live API",
        "sbs_fallback": "SBS receiver fallback",
        "rui_usecase_payload": "RUI payload",
        "rui_shared_file": "RUI shared file",
        "not_applied": "Not applied",
        "none": "N/A",
        "unknown": "Unknown",
    }
    return mapping.get(key, str(source or "Unknown"))


def render_airport_card(brief: Dict[str, Any]) -> None:
    """Airport & security information card."""
    code = brief.get("airport_code", "")
    sec = int(brief.get("security_minutes", 0) or 0)
    chk = int(brief.get("checkin_minutes", 0) or 0)
    walk = int(brief.get("walk_minutes", 0) or 0)
    buf = int(brief.get("buffer_minutes", 0) or 0)
    model_delay = brief.get("model_predicted_delay_minutes")
    operational_delay = float(brief.get("operational_delay_minutes", 0) or 0)
    predicted_delay = brief.get("predicted_delay_minutes")
    process_subtotal = int(brief.get("total_airport_minutes", sec + chk + walk + buf) or 0)
    expected_dep = brief.get("departure_dt")
    expected_dep_txt = expected_dep.strftime("%H:%M") if isinstance(expected_dep, datetime) else "N/A"
    process_anchor_dt = brief.get("airport_process_anchor_dt") or brief.get("scheduled_departure_dt")
    process_anchor_txt = (
        process_anchor_dt.strftime("%H:%M") if isinstance(process_anchor_dt, datetime) else "N/A"
    )

    model_delay_value = float(model_delay) if isinstance(model_delay, (int, float)) else 0.0
    operational_delay_value = max(0.0, float(operational_delay or 0.0))
    component_delay_total = round(max(0.0, model_delay_value + operational_delay_value), 1)

    if isinstance(predicted_delay, (int, float)):
        delay_total_value = max(0.0, float(predicted_delay))
    else:
        delay_total_value = component_delay_total
    delay_total_txt = f"{delay_total_value:+.1f} min"
    planning_total = process_subtotal + delay_total_value
    planning_total_txt = f"{planning_total:.1f} min"

    rows = [
        ("📋 Check-in and bag drop", f"{chk} min"),
        ("🛡️ Security check", f"{sec} min"),
        ("🚶 Walk to gate", f"{walk} min"),
        ("⏳ Safety buffer", f"{buf} min"),
    ]
    delay_rows: List[Tuple[str, str]] = [
        ("🧠 AI model delay", f"{model_delay_value:+.1f} min"),
        ("🛰️ Traffic and operations delay", f"{operational_delay_value:+.1f} min"),
    ]

    html = f'<div class="info-card"><h4>🏢 {code} Airport — Time Plan</h4>'
    html += (
        '<div class="kpi-strip">'
        '<div class="kpi-item"><div class="kpi-label">AIRPORT PROCESS TIME</div>'
        f'<div class="kpi-value">{process_subtotal} min</div></div>'
        '<div class="kpi-item"><div class="kpi-label">TOTAL PREDICTED DELAY</div>'
        f'<div class="kpi-value">{delay_total_txt}</div></div>'
        '<div class="kpi-item"><div class="kpi-label">PROCESS + DELAY</div>'
        f'<div class="kpi-value">{planning_total_txt}</div></div>'
        '</div>'
    )
    html += (
        f'<p class="subtle">Process anchor (scheduled): <span class="val">{process_anchor_txt}</span> '
        f'· Expected departure: <span class="val">{expected_dep_txt}</span></p>'
    )
    html += '<div class="section-block"><div class="section-title">Airport process steps</div>'
    html += '<table class="kv-table">'
    for label, value_text in rows:
        html += f"<tr><td>{label}</td><td>{value_text}</td></tr>"
    html += '</table></div>'

    html += '<div class="section-block delay-block"><div class="section-title">Delay components</div>'
    html += '<table class="kv-table">'
    for label, value_text in delay_rows:
        html += f"<tr><td>{label}</td><td>{value_text}</td></tr>"
    html += '</table></div>'

    html += (
        '<div class="total-strip">'
        '<div class="total-box">'
        '<div class="lbl">Process subtotal</div>'
        f'<div class="num">{process_subtotal} min</div>'
        '</div>'
        '<div class="total-box">'
        '<div class="lbl">Predicted delay</div>'
        f'<div class="num">{delay_total_txt}</div>'
        '</div>'
        '<div class="total-box main">'
        '<div class="lbl">Planning total</div>'
        f'<div class="num">{planning_total_txt}</div>'
        '</div>'
        '</div>'
    )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_operational_card(brief: Dict[str, Any]) -> None:
    """Operational factors card (JFK congestion + FR24/ADS-B previous leg)."""
    congestion_delay = int(brief.get("congestion_delay_minutes", 0) or 0)
    congestion_level = str(brief.get("congestion_level", "unknown"))
    adsb_delay = int(brief.get("adsb_fr24_delay_minutes", 0) or 0)
    adsb_source_raw = brief.get("adsb_fr24_source", "none")
    prev_leg_in_air = bool(brief.get("adsb_fr24_in_air", False))
    operational_sum = max(0, congestion_delay + adsb_delay)

    if str(adsb_source_raw).strip().lower() == "not_applied":
        prev_leg_status = "Not applied"
    elif prev_leg_in_air:
        prev_leg_status = "In-air"
    else:
        prev_leg_status = "Unavailable"

    html = '<div class="info-card"><h4>🛰️ Operational Factors</h4>'
    html += (
        '<div class="kpi-strip">'
        '<div class="kpi-item"><div class="kpi-label">TOTAL OPERATIONAL DELAY</div>'
        f'<div class="kpi-value">+{operational_sum} min</div></div>'
        '<div class="kpi-item"><div class="kpi-label">JFK CONGESTION</div>'
        f'<div class="kpi-value">{congestion_level}</div></div>'
        '<div class="kpi-item"><div class="kpi-label">PREVIOUS-LEG SIGNAL</div>'
        f'<div class="kpi-value">{prev_leg_status}</div></div>'
        '</div>'
    )

    if operational_sum <= 0:
        html += (
            '<div class="section-block">'
            '<p style="margin:0;"><span class="val">No extra operational delay right now.</span></p>'
            f'<p class="subtle" style="margin-top:6px;">JFK congestion: {congestion_level} · '
            f'Previous-leg signal: {prev_leg_status}</p>'
            '</div>'
        )
    else:
        html += (
            '<div class="section-block">'
            '<div class="section-title">Delay breakdown</div>'
            '<table class="kv-table">'
            f'<tr><td>JFK congestion delay</td><td>+{congestion_delay} min</td></tr>'
            f'<tr><td>Previous-leg delay</td><td>+{adsb_delay} min</td></tr>'
            '</table>'
            '</div>'
            '<p class="subtle">Included in predicted delay above.</p>'
        )

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_input_summary_table(brief: Dict[str, Any]) -> None:
    """Render UI summary equivalent to CLI input confirmation."""
    def _fmt_dt_safe(value: Any) -> str:
        return format_dt(value) if isinstance(value, datetime) else "N/A"

    baggage_txt = "Checked baggage" if brief.get("has_checked_baggage") else "Carry-on only"
    precheck_txt = "Yes" if brief.get("has_tsa_precheck") else "No"
    rows = [
        ("Flight", brief.get("flight_number") or "N/A"),
        ("Scheduled departure", _fmt_dt_safe(brief.get("scheduled_departure_dt"))),
        ("Airport", brief.get("airport_code") or "N/A"),
        ("Origin", brief.get("origin") or "N/A"),
        ("Travel mode", "DRIVE"),
        ("Baggage", baggage_txt),
        ("TSA PreCheck", precheck_txt),
        ("Terminal / Gate", f'{brief.get("terminal") or "N/A"} / {brief.get("gate") or "N/A"}'),
        ("Departure time source", brief.get("departure_time_source") or "unknown"),
    ]
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    st.table(df)


def render_calculation_detail_table(brief: Dict[str, Any]) -> None:
    """Render UI summary equivalent to CLI detailed calculation."""
    def _fmt_dt_safe(value: Any) -> str:
        return format_dt(value) if isinstance(value, datetime) else "N/A"

    delay_val = brief.get("predicted_delay_minutes")
    model_delay_val = brief.get("model_predicted_delay_minutes")
    op_delay_val = brief.get("operational_delay_minutes")
    if isinstance(delay_val, (int, float)):
        delay_text = f"{delay_val:.1f} min"
    else:
        delay_text = "N/A"
    if isinstance(model_delay_val, (int, float)):
        model_delay_text = f"{model_delay_val:.1f} min"
    else:
        model_delay_text = "N/A"
    if isinstance(op_delay_val, (int, float)):
        op_delay_text = f"{op_delay_val:.1f} min"
    else:
        op_delay_text = "N/A"

    rows = [
        ("Recommended leave", _fmt_dt_safe(brief.get("leave_dt"))),
        ("Target airport arrival", _fmt_dt_safe(brief.get("airport_arrival_dt"))),
        ("Flight departure (scheduled)", _fmt_dt_safe(brief.get("scheduled_departure_dt"))),
        ("Flight departure (expected)", _fmt_dt_safe(brief.get("departure_dt"))),
        ("Drive time", f'{brief.get("drive_minutes", "N/A")} min'),
        ("Check-in & bag drop", f'{brief.get("checkin_minutes", "N/A")} min'),
        ("Security screening", f'{brief.get("security_minutes", "N/A")} min'),
        ("Walk to gate", f'{brief.get("walk_minutes", "N/A")} min'),
        ("Buffer", f'{brief.get("buffer_minutes", "N/A")} min'),
        ("Total airport time", f'{brief.get("total_airport_minutes", "N/A")} min'),
        ("Total predicted delay", delay_text),
        ("Model delay (FT-Transformer)", model_delay_text),
        ("Operational delay (JFK+FR24/ADS-B)", op_delay_text),
        (
            "Congestion",
            f'{brief.get("congestion_level", "unknown")} '
            f'(source {brief.get("congestion_source", "none")})',
        ),
        (
            "FR24+ADS-B",
            f'{brief.get("adsb_fr24_source", "none")} / {brief.get("adsb_fr24_reason", "unknown")}',
        ),
    ]
    df = pd.DataFrame(rows, columns=["Item", "Value"])
    st.table(df)
    st.caption(
        "Sources · "
        f'delay: {brief.get("delay_source", "none")} · '
        f'congestion: {brief.get("congestion_source", "none")} · '
        f'fr24/adsb: {brief.get("adsb_fr24_source", "none")} · '
        f'drive: {brief.get("drive_source", "unknown")} · '
        f'security: {brief.get("security_source", "config")} · '
        f'check-in: {brief.get("checkin_source", "baggage_profile")} · '
        f'gate walk: {brief.get("walk_source", "config")}'
    )


def render_compact_recommendation_summary(brief: Dict[str, Any]) -> None:
    """Render a compact, non-duplicative recommendation summary block."""
    def _fmt_dt_safe(value: Any) -> str:
        return format_dt(value) if isinstance(value, datetime) else "N/A"

    leave = _fmt_dt_safe(brief.get("leave_dt"))
    airport_arrival = _fmt_dt_safe(brief.get("airport_arrival_dt"))
    sched_dep = _fmt_dt_safe(brief.get("scheduled_departure_dt"))
    exp_dep = _fmt_dt_safe(brief.get("departure_dt"))
    airport = brief.get("airport_code", "N/A")

    drive = int(brief.get("drive_minutes", 0) or 0)
    checkin = int(brief.get("checkin_minutes", 0) or 0)
    sec = int(brief.get("security_minutes", 0) or 0)
    walk = int(brief.get("walk_minutes", 0) or 0)
    buf = int(brief.get("buffer_minutes", 0) or 0)
    total_airport = int(brief.get("total_airport_minutes", sec + checkin + walk + buf) or 0)

    delay_val = brief.get("predicted_delay_minutes")
    model_delay = brief.get("model_predicted_delay_minutes")
    operational_delay = brief.get("operational_delay_minutes")
    if isinstance(delay_val, (int, float)):
        delay_text = f"{delay_val:+.1f} min"
    else:
        delay_text = "N/A"

    delay_breakdown_parts: List[str] = []
    if isinstance(model_delay, (int, float)):
        delay_breakdown_parts.append(f"model {model_delay:+.1f}")
    if isinstance(operational_delay, (int, float)) and operational_delay > 0:
        delay_breakdown_parts.append(f"operational +{operational_delay:.1f}")
    delay_breakdown = " + ".join(delay_breakdown_parts) if delay_breakdown_parts else "no extra delay"

    html = (
        '<div class="info-card">'
        '<h4>✅ Departure Recommendation (Compact)</h4>'
        f'<p><span class="val">Leave:</span> {leave}</p>'
        f'<p><span class="val">Airport arrival target ({airport}):</span> {airport_arrival}</p>'
        f'<p><span class="val">Flight time:</span> scheduled {sched_dep} · expected {exp_dep} '
        f'(delay {delay_text}; {delay_breakdown})</p>'
        f'<p><span class="val">Time breakdown:</span> drive {drive} + check-in {checkin} + '
        f'security {sec} + walk {walk} + buffer {buf} = total airport {total_airport} min</p>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_countdown(now: datetime, leave_dt: datetime) -> str:
    """Return a small HTML chip showing how long until departure."""
    delta = leave_dt - now
    total_min = int(delta.total_seconds() / 60)

    if total_min < 0:
        abs_min = abs(total_min)
        hrs, mins = divmod(abs_min, 60)
        text = f"{hrs}h {mins}m ago" if hrs else f"{mins}m ago"
        return f'<span class="countdown-chip past">⚠️ {text}</span>'

    hrs, mins = divmod(total_min, 60)
    text = f"{hrs}h {mins}m left" if hrs else f"{mins}m left"
    return f'<span class="countdown-chip">⏱ {text}</span>'


# ═══════════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════

def render_welcome() -> None:
    """Show a nice placeholder when no query has been submitted yet."""
    st.markdown(
        """
        <div class="welcome-box">
            <div class="icon">✈️</div>
            <h2>Welcome to IFTA Departure Assistant</h2>
            <p>
                Ask a question like<br>
                <b>"When should I leave for JFK?"</b><br><br>
                The assistant will show you:<br>
                🕐 Recommended departure time &nbsp;·&nbsp;
                🗺️ Route map &nbsp;·&nbsp;
                🚦 Live traffic<br>
                🛡️ Security wait &nbsp;·&nbsp;
                🛩️ Flight position &nbsp;·&nbsp;
                📋 Step-by-step timeline
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_pending_widget_updates() -> None:
    """
    Apply deferred widget value updates before widgets are instantiated.
    Streamlit forbids mutating a widget key after the widget is created in the same run.
    """
    pending = st.session_state.pop("pending_widget_updates", None)
    if not isinstance(pending, dict):
        return

    for key, value in pending.items():
        if value is None:
            continue
        st.session_state[key] = value


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="IFTA Departure Assistant",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Load config ──────────────────────────────────
    config = load_config(str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)

    # ── Session defaults for user inputs ─────────────
    if "origin_input" not in st.session_state:
        st.session_state["origin_input"] = config.get(
            "default_origin", "Times Square, Manhattan, NY"
        )
    if "airport_code_input" not in st.session_state:
        st.session_state["airport_code_input"] = config.get("default_airport", "JFK")
    if "flight_number_input" not in st.session_state:
        st.session_state["flight_number_input"] = ""
    if "flight_departure_input" not in st.session_state:
        st.session_state["flight_departure_input"] = ""
    if "checked_baggage_input" not in st.session_state:
        st.session_state["checked_baggage_input"] = True
    if "tsa_precheck_input" not in st.session_state:
        st.session_state["tsa_precheck_input"] = False
    if "terminal_input" not in st.session_state:
        st.session_state["terminal_input"] = ""
    if "gate_input" not in st.session_state:
        st.session_state["gate_input"] = ""
    if "query_text_input" not in st.session_state:
        st.session_state["query_text_input"] = "When should I leave for JFK?"

    # Apply deferred autofill values from previous run before widget creation.
    _apply_pending_widget_updates()

    # ── Sidebar ──────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("#### 📍 Route")
        origin = st.text_input(
            "Origin address",
            key="origin_input",
            help="Your starting location",
        )
        airport_code = st.text_input(
            "Airport code (IATA)",
            key="airport_code_input",
            help="e.g. JFK, LGA, EWR, SFO, ICN",
        )

        st.markdown("#### ✈️ Flight")
        flight_number = st.text_input(
            "Flight number",
            key="flight_number_input",
            help="e.g. KE937, UA123",
        )
        flight_departure_local = st.text_input(
            "Flight departure time",
            key="flight_departure_input",
            help="YYYY-MM-DD HH:MM or HH:MM",
        )
        has_checked_baggage = st.checkbox(
            "Checked baggage",
            key="checked_baggage_input",
            help="Include bag-drop/check-in time",
        )
        has_tsa_precheck = st.checkbox(
            "TSA PreCheck",
            key="tsa_precheck_input",
            help="Use lower security wait assumptions where available",
        )
        terminal = st.text_input(
            "Terminal (optional)",
            key="terminal_input",
            help="e.g. Terminal 4 / T4",
        )
        gate = st.text_input(
            "Gate (optional)",
            key="gate_input",
            help="e.g. B27",
        )

        with st.expander("🔧 Advanced", expanded=False):
            route_hint = st.text_input("Route hint", value="")
            adv_flight_lat = st.text_input("Flight lat", value="")
            adv_flight_lon = st.text_input("Flight lon", value="")
            adv_flight_alt = st.text_input("Flight alt (ft)", value="")
            adv_flight_speed = st.text_input("Flight speed (kt)", value="")
            custom_config = st.text_input(
                "Config file",
                value=str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else "",
            )
            if custom_config and Path(custom_config).exists():
                config = load_config(custom_config)

        st.markdown("#### 🤖 AI Backend")
        gemini_on = _resolve_bool_setting(config, "use_gemini", "USE_GEMINI", False)
        transformer_on = _resolve_bool_setting(
            config, "use_transformer", "USE_TRANSFORMER", False
        )
        st.caption(f"Gemini: {'ON' if gemini_on else 'OFF'}")
        st.caption(f"Transformer delay model: {'ON' if transformer_on else 'OFF'}")

        st.markdown("#### 🎫 Ticket OCR (Gemini)")
        ticket_file = st.file_uploader(
            "Upload flight ticket image",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
            help="Gemini Vision extracts flight number / airport / departure time",
        )
        c_ticket_a, c_ticket_b = st.columns(2)
        with c_ticket_a:
            parse_ticket_btn = st.button("Analyze Ticket", use_container_width=True)
        with c_ticket_b:
            clear_ticket_btn = st.button("Clear Ticket", use_container_width=True)

        if clear_ticket_btn:
            st.session_state.pop("ticket_data", None)
            st.session_state.pop("ticket_error", None)
            st.session_state.pop("pending_widget_updates", None)
            st.success("Ticket data cleared.")

        if parse_ticket_btn:
            ticket_data, ticket_error = _extract_ticket_with_gemini(ticket_file, gemini_on)
            if ticket_error:
                st.session_state["ticket_error"] = ticket_error
                st.error(ticket_error)
            else:
                st.session_state["ticket_data"] = ticket_data
                st.session_state.pop("ticket_error", None)

                # Defer widget-key updates to next rerun (before widgets are created).
                pending_updates: Dict[str, Any] = {}
                if ticket_data.get("flight_number"):
                    pending_updates["flight_number_input"] = ticket_data["flight_number"]
                if ticket_data.get("departure_airport"):
                    pending_updates["airport_code_input"] = ticket_data["departure_airport"]
                if ticket_data.get("departure_time"):
                    pending_updates["flight_departure_input"] = ticket_data["departure_time"]
                if ticket_data.get("terminal"):
                    pending_updates["terminal_input"] = ticket_data["terminal"]
                if ticket_data.get("gate"):
                    pending_updates["gate_input"] = ticket_data["gate"]
                inferred_checked_baggage = _infer_checked_baggage(ticket_data)
                if inferred_checked_baggage is not None:
                    pending_updates["checked_baggage_input"] = inferred_checked_baggage

                pending_updates["query_text_input"] = _build_ticket_query(ticket_data)
                st.session_state["pending_widget_updates"] = pending_updates
                st.session_state["run_from_ticket"] = True
                st.rerun()

        ticket_preview: Optional[Dict[str, str]] = st.session_state.get("ticket_data")
        if ticket_preview:
            summary_parts = []
            if ticket_preview.get("flight_number"):
                summary_parts.append(ticket_preview["flight_number"])
            if ticket_preview.get("departure_airport"):
                summary_parts.append(ticket_preview["departure_airport"])
            if ticket_preview.get("departure_time"):
                summary_parts.append(ticket_preview["departure_time"])
            if ticket_preview.get("baggage_allowance"):
                summary_parts.append(f"Baggage {ticket_preview['baggage_allowance']}")
            if summary_parts:
                st.caption("Parsed: " + " | ".join(summary_parts))
        ticket_error_txt = st.session_state.get("ticket_error")
        if ticket_error_txt:
            st.caption(f"Last OCR error: {ticket_error_txt}")

        st.markdown("---")
        st.markdown(
            '<p style="text-align:center;color:#546e7a;font-size:0.78rem;">'
            "IFTA Departure Assistant v2.0<br>Team T-Rex · CMU IITP 2025</p>",
            unsafe_allow_html=True,
        )

    # ── Header ───────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:6px 0 18px 0;">'
        '<h1 style="margin:0;font-size:2.1rem;">✈️ IFTA Departure Assistant</h1>'
        '<p style="color:#8ab4f8;font-size:0.95rem;margin-top:2px;">'
        "AI-powered departure guidance with real-time traffic & flight data</p></div>",
        unsafe_allow_html=True,
    )

    # ── Query bar ────────────────────────────────────
    col_q, col_btn = st.columns([6, 1])
    with col_q:
        query = st.text_input(
            "query_input",
            key="query_text_input",
            label_visibility="collapsed",
            placeholder="When should I leave for JFK?",
        )
    with col_btn:
        run_btn = st.button("🔍 Ask", type="primary", use_container_width=True)

    # ── Process ──────────────────────────────────────
    run_from_ticket = bool(st.session_state.pop("run_from_ticket", False))
    if (run_btn or run_from_ticket) and query.strip():
        with st.spinner("Analyzing departure information …"):
            brief = process_departure_query(
                query=query.strip(),
                config=config,
                origin=origin.strip(),
                airport_code=airport_code.strip().upper() if airport_code.strip() else "",
                flight_departure_local=flight_departure_local.strip(),
                route=route_hint.strip(),
                flight_number=flight_number.strip(),
                has_checked_baggage=has_checked_baggage,
                has_tsa_precheck=has_tsa_precheck,
                terminal=terminal.strip(),
                gate=gate.strip(),
                flight_lat=_safe_float(adv_flight_lat),
                flight_lon=_safe_float(adv_flight_lon),
                flight_alt=_safe_float(adv_flight_alt),
                flight_speed=_safe_float(adv_flight_speed),
            )
        st.session_state["brief"] = brief
        st.session_state["query"] = query.strip()

    ticket_data_for_view: Optional[Dict[str, str]] = st.session_state.get("ticket_data")
    if ticket_data_for_view:
        with st.expander("🎫 Parsed Ticket Data", expanded=False):
            st.json(ticket_data_for_view)

    # ── Display results ──────────────────────────────
    brief: Optional[Dict[str, Any]] = st.session_state.get("brief")

    if brief is None:
        render_welcome()
        return

    if brief.get("error"):
        st.error(f"Error: {brief['error']}")
        return

    if not brief.get("intent"):
        # Show Gemini summary if available for non-departure queries
        gemini_summary = brief.get("gemini_summary")
        if gemini_summary:
            st.info(f"🤖 {gemini_summary}")
        else:
            st.info(
                "🤔 This doesn't seem to be a departure question. "
                'Try "When should I leave for JFK?"'
            )
        return

    # ── Recommendation banner ────────────────────────
    leave_dt: datetime = brief["leave_dt"]
    now_local: datetime = brief["now_local"]
    airport = brief.get("airport_code", "N/A")
    drive_min = brief.get("drive_minutes", "?")
    traffic = brief.get("traffic_level", "unknown")
    t_color = TRAFFIC_COLORS.get(traffic, "#78909C")
    countdown_html = render_countdown(now_local, leave_dt)

    st.markdown(
        f'<div class="rec-banner">'
        f'<span class="rec-title">💡 Recommendation</span>{countdown_html}'
        f'<p class="rec-body">'
        f'Leave by <b class="leave">{leave_dt.strftime("%H:%M")}</b> '
        f'({leave_dt.strftime("%Y-%m-%d")}) '
        f'to reach <b class="airport">{airport}</b> on time. '
        f'Drive ≈ <b>{drive_min} min</b> with '
        f'<b style="color:{t_color};">{traffic}</b> traffic.</p></div>',
        unsafe_allow_html=True,
    )

    # ── Compact summary first (default) ──────────────
    render_compact_recommendation_summary(brief)

    # ── AI narrative (optional, collapsed) ───────────
    ai_response = brief.get("ai_response")
    if ai_response:
        with st.expander(f'🤖 AI Assistant Narrative ({brief.get("ai_model", "gemini")})', expanded=False):
            st.markdown(ai_response)

    with st.expander("📋 Input Summary", expanded=True):
        render_input_summary_table(brief)

    with st.expander("📊 Calculation Details", expanded=True):
        render_calculation_detail_table(brief)

    # ── Metric cards ─────────────────────────────────
    predicted_delay = brief.get("predicted_delay_minutes")
    if isinstance(predicted_delay, (int, float)):
        c1, c2, c3, c4, c5 = st.columns(5)
    else:
        c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🕐 Leave By", leave_dt.strftime("%H:%M"))
    with c2:
        st.metric("🚗 Drive Time", f"{drive_min} min")
    with c3:
        emoji = TRAFFIC_EMOJI.get(traffic, "⚪")
        st.metric("🚦 Traffic", f"{emoji} {traffic.capitalize()}")
    with c4:
        st.metric("🛡️ Security", f"{brief.get('security_minutes', 'N/A')} min")
    if isinstance(predicted_delay, (int, float)):
        sign = "+" if predicted_delay >= 0 else ""
        with c5:
            st.metric("✈️ Delay (Total)", f"{sign}{predicted_delay:.1f} min")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Map  +  Timeline columns ─────────────────────
    col_map, col_info = st.columns([3, 2])

    with col_map:
        st.markdown('<div class="sec-hdr">🗺️ Route & Flight Map</div>', unsafe_allow_html=True)

        # Folium interactive map
        fmap = create_map(
            airport_code=brief.get("airport_code", ""),
            flight_position=brief.get("flight_position"),
        )
        if fmap and HAS_FOLIUM:
            st_folium(fmap, width=None, height=420, returned_objects=[])
        elif not HAS_FOLIUM:
            # Fallback: use st.map if folium unavailable
            pts = []
            ap = AIRPORT_COORDS.get(brief.get("airport_code", ""))
            if ap:
                pts.append({"lat": ap[0], "lon": ap[1]})
            fp = brief.get("flight_position")
            if fp:
                pts.append({"lat": fp["lat"], "lon": fp["lon"]})
            if pts:
                st.map(pd.DataFrame(pts), zoom=5)
            else:
                st.info("Map data not available.")
        else:
            st.info("No map markers to display.")

        # Google Maps embed (if API key available)
        gk = brief.get("google_maps_api_key", "")
        if gk:
            st.markdown(
                '<div class="sec-hdr">🚗 Google Maps — Traffic Route</div>',
                unsafe_allow_html=True,
            )
            render_google_maps_embed(
                origin=brief.get("origin", ""),
                destination=brief.get("destination", ""),
                api_key=gk,
            )

        # Google Maps link (always available, no API key needed)
        map_url = brief.get("map_url", "")
        if map_url:
            st.markdown(
                f'<a href="{map_url}" target="_blank" class="gmaps-btn">'
                f"🗺️ Open in Google Maps (English, traffic)</a>",
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown(
            '<div class="sec-hdr">📋 Departure Timeline</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Airport process is counted backward from scheduled departure "
            "(conservative baseline). Delay details are summarized in the Time Plan card."
        )
        tl_html = build_timeline_html(brief)
        if tl_html:
            st.markdown(tl_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Airport info card
        render_airport_card(brief)
        render_operational_card(brief)

    # ── Flight info card ─────────────────────────────
    render_flight_card(brief)

    # ── Notes / warnings ─────────────────────────────
    notes: List[str] = brief.get("notes", [])
    if notes:
        with st.container():
            for note in notes:
                st.warning(f"⚠️ {note}")

    # ── Raw data ─────────────────────────────────────
    with st.expander("📊 Raw Data"):
        display = {}
        for k, v in brief.items():
            if isinstance(v, datetime):
                display[k] = format_dt(v)
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                display[k] = v
        st.json(display)


if __name__ == "__main__":
    main()
