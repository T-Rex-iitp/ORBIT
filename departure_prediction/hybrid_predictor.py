"""
Hybrid departure-time prediction system.

Architecture:
1. Transformer model: predicts flight delay time (trained model)
2. Google Routes API: address -> airport travel time
3. TSA Wait Time: security screening wait time
4. Baggage Check: checked baggage drop-off time
5. LLM Agent: final departure-time recommendation (English)
"""

import io
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
from utils.google_routes import calculate_travel_time
from utils.tsa_wait_time import get_tsa_wait_time
from utils.weather_google import get_weather
from utils.flight_status_checker import check_flight
from utils.gate_walk_time import get_gate_walk_time
from utils.operational_factors import OperationalFactorsAnalyzer
from utils.congestion_check import JFKCongestionChecker, check_jfk_congestion, check_jfk_congestion_from_rui
from utils.resilience import (
    ResilientAPIWrapper,
    get_fallback_travel_time,
    get_fallback_tsa_wait,
    get_fallback_weather,
    get_fallback_flight_status,
    validate_flight_info,
    ResilienceConfig
)
from utils.cache import (
    cache_manager,
    historical_fallback,
    cached_api_call
)
import requests
import json
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ADSB_PY_DIR = _PROJECT_ROOT / "ADS-B-Display" / "python"
if str(_ADSB_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_ADSB_PY_DIR))

try:
    from previous_flight_finder import estimate_previous_leg_delay_minutes
except Exception:
    estimate_previous_leg_delay_minutes = None


class FeatureTokenizer(nn.Module):
    """Convert each feature into an individual embedding."""
    def __init__(self, num_features, d_token):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        
        # Linear transformation for each feature
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_features)
        ])
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
    
    def forward(self, x):
        batch_size = x.size(0)
        tokens = []
        for i in range(self.num_features):
            feature_val = x[:, i].unsqueeze(-1)
            token = self.feature_projections[i](feature_val)
            tokens.append(token.unsqueeze(1))
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = [cls_tokens] + tokens
        tokens = torch.cat(tokens, dim=1)
        return tokens


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer"""
    def __init__(self, num_features, d_token=64, n_blocks=3, attention_heads=8, 
                 ffn_d_hidden=256, attention_dropout=0.2, ffn_dropout=0.1, residual_dropout=0.0):
        super().__init__()
        
        self.feature_tokenizer = FeatureTokenizer(num_features, d_token)
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=attention_heads,
                dim_feedforward=ffn_d_hidden,
                dropout=residual_dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(n_blocks)
        ])
        
        self.norm = nn.LayerNorm(d_token)
        
        self.head = nn.Sequential(
            nn.Linear(d_token, ffn_d_hidden),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden, ffn_d_hidden // 2),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_d_hidden // 2, 1)
        )
    
    def forward(self, x):
        x = self.feature_tokenizer(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_output = x[:, 0, :]
        output = self.head(cls_output)
        return output


class FlightDelayTransformer(nn.Module):
    """Transformer-based delay prediction model."""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(FlightDelayTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class HybridDeparturePredictor:
    """Hybrid departure-time prediction system."""
    
    def __init__(
        self, 
        model_path='models/delay_predictor_full.pkl', 
        use_gcs=False, 
        gcs_bucket=None,
        use_gemini=False,
        gemini_project_id=None
    ):
        """
        Args:
            model_path: Trained Transformer model path (local or GCS)
            use_gcs: Whether to load model from GCS
            gcs_bucket: GCS bucket name (required when use_gcs=True)
            use_gemini: Whether to use Gemini (if True, uses Gemini instead of Ollama)
            gemini_project_id: GCP project ID (can also come from environment variable)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket or os.getenv('GCS_MODEL_BUCKET')
        self.use_gemini = use_gemini or os.getenv('USE_GEMINI', 'false').lower() == 'true'
        self.gemini_project_id = gemini_project_id or os.getenv('GCP_PROJECT_ID')
        
        self.load_model(model_path)
        
        # Initialize LLM client
        if self.use_gemini:
            print("🤖 Using Google Gemini for LLM")
            from utils.gemini_direct_client import GeminiDirectClient
            self.llm_client = GeminiDirectClient()
        else:
            print("🤖 Using Ollama for LLM")
            self.ollama_url = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
            self.llm_client = None

        # Operational context (congestion) analyzer
        self.operational_analyzer = OperationalFactorsAnalyzer()

        # Congestion evaluator based on historical JFK data
        self.jfk_congestion_checker = JFKCongestionChecker()

    def _normalize_rui_congestion(self, congestion_payload, reference_time=None):
        """Normalize RUI congestion use-case response into internal congestion_info format.

        Supports two modes:
          (A) RUI already sends level/score/extra_delay -> use as-is
          (B) RUI sends only flight_count -> evaluate with JFKCongestionChecker
        """
        if not isinstance(congestion_payload, dict):
            return None

        # -- (B) If only flight_count is provided: assess congestion vs historical data --
        raw_count = congestion_payload.get('flight_count')
        if raw_count is not None:
            try:
                raw_count = int(raw_count)
            except (TypeError, ValueError):
                raw_count = None

        if raw_count is not None:
            hour = congestion_payload.get('hour')
            result = self.jfk_congestion_checker.check_congestion(
                current_flight_count=raw_count,
                hour=hour,
                reference_time=reference_time,
            )
            result['source'] = 'RUI_historical_comparison'
            return result

        # -- (A) Existing mode: congestion result already computed --
        level = congestion_payload.get('level') or congestion_payload.get('congestion_level') or 'unknown'

        try:
            score = float(congestion_payload.get('score', 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        try:
            sample_size = int(congestion_payload.get('sample_size', 0) or 0)
        except (TypeError, ValueError):
            sample_size = 0

        recommended_extra_delay = congestion_payload.get('recommended_extra_delay')
        if recommended_extra_delay is None:
            recommended_extra_delay = congestion_payload.get('extra_delay_minutes')
        if recommended_extra_delay is None:
            recommended_extra_delay = congestion_payload.get('delay_minutes')

        try:
            recommended_extra_delay = int(recommended_extra_delay or 0)
        except (TypeError, ValueError):
            recommended_extra_delay = 0

        return {
            'level': level,
            'score': score,
            'sample_size': sample_size,
            'recommended_extra_delay': recommended_extra_delay,
            'source': 'RUI'
        }

    def _load_pickle_with_device_map(self, file_obj):
        """Safely deserialize torch storage regardless of CPU/GPU environment differences."""
        device = self.device

        class DeviceAwareUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location=device)
                return super().find_class(module, name)

        return DeviceAwareUnpickler(file_obj).load()
        
    def load_model(self, model_path):
        """Load trained model (local or GCS)."""
        if self.use_gcs:
            # Load directly from GCS (in-memory, no download)
            print(f"📦 Loading model from GCS: gs://{self.gcs_bucket}/{model_path}")
            from utils.gcs_model_loader import GCSModelLoader
            
            if not self.gcs_bucket:
                raise ValueError("GCS bucket name required. Set GCS_MODEL_BUCKET env var or pass gcs_bucket parameter")
            
            loader = GCSModelLoader(self.gcs_bucket)
            package = loader.load_pickle_model(model_path)
        else:
            # Load locally
            print(f"📦 Loading model: {model_path}")
            with open(model_path, 'rb') as f:
                package = self._load_pickle_with_device_map(f)
        
        # Rebuild model (FT-Transformer)
        config = package['model_config']
        
        if config['model_type'] == 'FTTransformer':
            self.model = FTTransformer(
                num_features=config['num_features'],
                d_token=config['d_token'],
                n_blocks=config['n_blocks'],
                attention_heads=config['attention_heads'],
                ffn_d_hidden=config['ffn_d_hidden'],
                attention_dropout=config['attention_dropout'],
                ffn_dropout=config['ffn_dropout'],
                residual_dropout=config['residual_dropout']
            ).to(self.device)
        else:
            # Legacy Transformer (compatibility)
            self.model = FlightDelayTransformer(
                input_dim=config['input_dim'],
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(self.device)
        
        self.model.load_state_dict(package['model_state_dict'])
        self.model.eval()
        
        # Preprocessing tools
        self.label_encoders = package['label_encoders']
        self.scaler = package['scaler']
        self.feature_columns = package['feature_columns']
        self.test_metrics = package['test_metrics']
        
        print(f"✅ Model load complete ({config.get('model_type', 'Transformer')})")
        print(f"   MAE: {self.test_metrics['mae']:.2f} min")
        print(f"   RMSE: {self.test_metrics['rmse']:.2f} min")
        print(f"   R²: {self.test_metrics['r2']:.4f}")
    
    def predict_delay(self, airline_code, origin, dest, flight_datetime):
        """
        Predict flight delay time.
        
        Args:
            airline_code: Airline code (e.g., 'B6')
            origin: Origin airport code (e.g., 'JFK')
            dest: Destination airport code (e.g., 'LAX')
            flight_datetime: Scheduled departure time (datetime object)
        
        Returns:
            predicted_delay: Predicted delay time (minutes)
        """
        # Build features
        features = {
            'op_unique_carrier': airline_code,
            'origin': origin,
            'dest': dest,
            'hour': flight_datetime.hour,
            'month': flight_datetime.month,
            'day_of_week': flight_datetime.weekday(),
            'day_of_month': flight_datetime.day,
            'is_weekend': 1 if flight_datetime.weekday() >= 5 else 0
        }
        
        # Encode categorical variables
        unknown_items = []
        try:
            encoded_features = features.copy()
            for col in ['op_unique_carrier', 'origin', 'dest']:
                if col in self.label_encoders:
                    try:
                        encoded_features[col] = self.label_encoders[col].transform([features[col]])[0]
                    except ValueError:
                        # Set unseen values to 0
                        encoded_features[col] = 0
                        col_name = {'op_unique_carrier': 'airline', 'origin': 'origin airport', 'dest': 'destination airport'}[col]
                        unknown_items.append(f"{col_name} '{features[col]}'")
                else:
                    # If the label encoder itself is missing
                    encoded_features[col] = 0
            
            if unknown_items:
                print(f"   ℹ️ Items not seen in training data: {', '.join(unknown_items)} (prediction based on similar patterns)")
        except Exception as e:
            # Return average delay when encoding fails
            print(f"   ⚠️ Prediction error: {str(e)} (using default value)")
            return 15.0  # Default value
        
        # Build feature array
        X = np.array([[encoded_features[col] for col in self.feature_columns]])
        
        # Normalize numeric features
        numeric_indices = [self.feature_columns.index(col) 
                          for col in ['hour', 'month', 'day_of_week', 'day_of_month']]
        X[:, numeric_indices] = self.scaler.transform(X[:, numeric_indices])
        
        # Predict
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predicted_delay = self.model(X_tensor).cpu().numpy()[0][0]
        
        return float(predicted_delay)
    
    def recommend_departure(self, address, flight_info, travel_mode='DRIVE', rui_usecase_data=None):
        """
        Recommend departure time with the hybrid system (resilience enhanced).
        
        Args:
            address: Departure address
            flight_info: Flight information dict
                - airline_code: Airline code
                - flight_number: Flight number
                - origin: Origin airport
                - dest: Destination airport
                - scheduled_time: Scheduled departure time (datetime)
                - has_checked_baggage: Whether checked baggage is present (optional, default=False)
                - has_tsa_precheck: Whether traveler has TSA PreCheck (optional, default=False)
            travel_mode: Travel mode ('DRIVE', 'TRANSIT', 'WALK', 'BICYCLE')
            rui_usecase_data: Operational use-case data passed from RUI (optional)
                - congestion_check: Congestion analysis result dict
        
        Returns:
            recommendation: LLM recommendation result (dict)
        """
        print(f"\n🔍 Hybrid prediction started...")
        
        # Validate and sanitize input data
        try:
            flight_info = validate_flight_info(flight_info)
            print(f"   ✅ Flight information validated")
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid flight information: {e}'
            }
        
        # 1. Check real-time flight status (caching + resilience)
        print(f"   🛫 Checking real-time flight status...")
        
        def fetch_flight_status():
            # Check cache first
            try:
                return cached_api_call(
                    category='flight_status',
                    api_func=lambda: check_flight(flight_info['flight_number']),
                    use_stale_on_error=True,
                    flight_number=flight_info['flight_number'],
                    date=flight_info['scheduled_time'].date().isoformat()
                )
            except:
                # Use historical stats if cache is unavailable
                route = f"{flight_info['origin']}-{flight_info['dest']}"
                avg_delay = historical_fallback.get_avg_flight_delay(
                    flight_info['airline_code'], 
                    route
                )
                print(f"   📊 Using historical average delay: {avg_delay:.1f} min")
                return {
                    'status': 'scheduled',
                    'is_delayed': False,
                    'delay_minutes': avg_delay,
                    'scheduled_departure': None,
                    'fallback_used': True
                }
        
        real_time_status = fetch_flight_status()
        
        # Use only if real-time data exists and dates match
        use_real_time = False
        if real_time_status and real_time_status.get('scheduled_departure') and not real_time_status.get('fallback_used'):
            # Convert scheduled_departure to datetime if it is a string
            scheduled_dep = real_time_status['scheduled_departure']
            if isinstance(scheduled_dep, str):
                from dateutil import parser
                scheduled_dep = parser.parse(scheduled_dep)
            
            api_date = scheduled_dep.date()
            ticket_date = flight_info['scheduled_time'].date()
            
            # Use if date matches and delay info is present
            if api_date == ticket_date and real_time_status.get('is_delayed'):
                real_delay = real_time_status['delay_minutes']
                print(f"   ⚠️ Real-time delay info: {real_delay} min")
                print(f"   📡 Airline announcement: {real_time_status['status_kr']}")
                if real_time_status.get('estimated_departure'):
                    est_dep = real_time_status['estimated_departure']
                    if isinstance(est_dep, str):
                        est_dep = parser.parse(est_dep)
                    print(f"   🕐 Estimated departure: {est_dep.strftime('%H:%M')}")
                
                # Prioritize real-time info
                predicted_delay = real_delay
                use_real_time = True
                
                # Update stats
                route = f"{flight_info['origin']}-{flight_info['dest']}"
                historical_fallback.update_flight_delay(
                    flight_info['airline_code'], 
                    route, 
                    real_delay
                )
            elif api_date != ticket_date:
                print(f"   ⚠️ API date mismatch (API: {api_date}, Ticket: {ticket_date}) - Using ticket info")
        
        if not use_real_time:
            # 2. Predict delay with Transformer when real-time data is unavailable
            try:
                predicted_delay = self.predict_delay(
                    airline_code=flight_info['airline_code'],
                    origin=flight_info['origin'],
                    dest=flight_info['dest'],
                    flight_datetime=flight_info['scheduled_time']
                )
                print(f"   📊 Predicted delay: {predicted_delay:.1f} min (AI prediction)")
            except Exception as e:
                print(f"   ⚠️ AI prediction failed: {e}")
                print(f"   🔄 Using default delay estimate")
                predicted_delay = ResilienceConfig.DEFAULT_FLIGHT_DELAY
            
            use_real_time = False
        
        # 3. Apply operational context (congestion + ADS-B/FR24 prior-leg delay)
        operational_delay = 0
        congestion_info = {
            'level': 'unknown',
            'score': 0.0,
            'sample_size': 0,
            'recommended_extra_delay': 0
        }
        adsb_fr24_info = {
            'found': False,
            'in_air': False,
            'delay_minutes': 0,
            'source': 'none',
            'reason': 'not_started',
            'validation_mismatch': False,
            'validation_notes': []
        }

        rui_congestion_data = None
        if rui_usecase_data is None and isinstance(flight_info, dict):
            # With RUI integration, use-case results can be included in flight_info
            rui_usecase_data = flight_info.get('rui_usecase_data')

        if isinstance(rui_usecase_data, dict):
            rui_congestion_data = self._normalize_rui_congestion(
                rui_usecase_data.get('congestion_check'),
                reference_time=flight_info['scheduled_time']
            )

        # (A) Congestion correction: RUI file -> RUI API payload -> AviationStack -> direct SBS
        if rui_congestion_data:
            # If RUI directly provides API payload
            src = rui_congestion_data.get('source', 'RUI')
            print(f"   🛩️ Applying RUI congestion data (source: {src})...")
            if rui_congestion_data.get('details'):
                d = rui_congestion_data['details']
                print(f"      Flight count: {d.get('current_count', 'N/A')}, "
                      f"Mean: {d.get('historical_mean', 'N/A')}, "
                      f"Z-score: {d.get('z_score', 'N/A')}")
            congestion_info = {**congestion_info, **rui_congestion_data}
        elif flight_info['origin'] == 'JFK':
            # JFK departure -> try RUI shared file first
            print(f"   🛩️ Checking RUI shared file for JFK flight count...")
            try:
                rui_result = self.jfk_congestion_checker.check_rui_congestion(
                    max_age_seconds=300,
                    hour=flight_info['scheduled_time'].hour,
                )
                if rui_result is not None:
                    congestion_info = {**congestion_info, **rui_result}
                    d = rui_result.get('details', {})
                    print(f"      ✅ RUI data: {rui_result['level']} "
                          f"(count={d.get('current_count', '?')}, "
                          f"mean={d.get('historical_mean', '?')}, "
                          f"z={d.get('z_score', '?')})")
                else:
                    raise ValueError("No valid RUI data")
            except Exception as e_rui:
                print(f"   ℹ️ RUI file not available: {e_rui}")
                # Fallback: AviationStack API
                if self.operational_analyzer.enabled:
                    print(f"   🛩️ Falling back to AviationStack API...")
                    try:
                        congestion_info = cached_api_call(
                            category='operational_congestion',
                            api_func=lambda: self.operational_analyzer.get_jfk_area_congestion(
                                flight_info['scheduled_time']
                            ),
                            use_stale_on_error=True,
                            origin=flight_info['origin'],
                            hour=flight_info['scheduled_time'].hour
                        ) or congestion_info
                    except Exception as e_api:
                        print(f"   ⚠️ AviationStack also failed: {e_api}")
                        # Fallback: direct SBS connection
                        print(f"   🔄 Falling back to direct SBS connection...")
                        try:
                            fallback_result = self.jfk_congestion_checker.check_realtime_congestion(
                                collect_seconds=5
                            )
                            if fallback_result.get('details', {}).get('current_count', 0) > 0:
                                congestion_info = {**congestion_info, **fallback_result}
                                print(f"      ✅ SBS data: {fallback_result['level']} "
                                      f"(count={fallback_result['details']['current_count']})")
                        except Exception as e_sbs:
                            print(f"   ⚠️ SBS connection also failed: {e_sbs}")
                else:
                    # No API key → try SBS directly
                    print(f"   🛩️ Falling back to direct SBS connection...")
                    try:
                        fallback_result = self.jfk_congestion_checker.check_realtime_congestion(
                            collect_seconds=5
                        )
                        if fallback_result.get('details', {}).get('current_count', 0) > 0:
                            congestion_info = {**congestion_info, **fallback_result}
                            print(f"      ✅ SBS data: {fallback_result['level']} "
                                  f"(count={fallback_result['details']['current_count']})")
                        else:
                            print(f"   ℹ️ No ADS-B data available, skipping congestion analysis")
                    except Exception as e_sbs:
                        print(f"   ℹ️ Congestion analysis skipped: {e_sbs}")
        else:
            print("   ℹ️ Congestion analysis skipped (non-JFK origin)")

        # (B) Prior-leg delay correction using ADS-B + FR24 (always try, fallback to 0)
        if estimate_previous_leg_delay_minutes is not None and flight_info.get('flight_number'):
            print("   🛰️ Estimating previous-leg delay via FR24 + ADS-B...")
            try:
                adsb_fr24_info = cached_api_call(
                    category='fr24_adsb_delay',
                    api_func=lambda: estimate_previous_leg_delay_minutes(
                        flight_no=flight_info['flight_number'],
                        expected_origin=flight_info.get('origin'),
                        expected_dest=flight_info.get('dest'),
                        expected_date=flight_info.get('scheduled_time'),
                        collect_time=5,
                    ),
                    use_stale_on_error=True,
                    flight_number=flight_info['flight_number'],
                    date=flight_info['scheduled_time'].date().isoformat()
                ) or adsb_fr24_info
            except Exception as e:
                print(f"   ⚠️ FR24+ADS-B delay estimation failed: {e}")
        else:
            print("   ℹ️ FR24+ADS-B estimator unavailable; skipping")

        congestion_delay = int(congestion_info.get('recommended_extra_delay', 0) or 0)
        adsb_fr24_delay = int(adsb_fr24_info.get('delay_minutes', 0) or 0)
        operational_delay = congestion_delay + adsb_fr24_delay

        congestion_detail = congestion_info.get('details', {})
        detail_str = ""
        if congestion_detail.get('z_score') is not None:
            detail_str = (
                f", z={congestion_detail['z_score']:.2f}, "
                f"count={congestion_detail.get('current_count', '?')}/"
                f"mean={congestion_detail.get('historical_mean', '?')}"
            )
        print(
            f"      • Area congestion: {congestion_info.get('level', 'unknown')} "
            f"(score {congestion_info.get('score', 0):.2f}, n={congestion_info.get('sample_size', 0)}"
            f"{detail_str}) "
            f"→ +{congestion_delay} min"
        )
        print(
            f"      • FR24+ADS-B previous leg: {adsb_fr24_info.get('reason', 'unknown')} "
            f"(source={adsb_fr24_info.get('source', 'none')}) "
            f"→ +{adsb_fr24_delay} min"
        )
        if adsb_fr24_info.get('validation_mismatch'):
            notes = ", ".join(adsb_fr24_info.get('validation_notes', []))
            print(f"      ⚠️ Validation mismatch (non-blocking): {notes}")

        if operational_delay > 0:
            print(f"      ⚠️ Operational adjustment applied: +{operational_delay} min")

        predicted_delay += operational_delay

        # 4. Compute actual departure time (scheduled + predicted delay)
        actual_departure = flight_info['scheduled_time'] + timedelta(minutes=predicted_delay)
        
        # 5. Fetch weather info (caching + resilience)
        print(f"   🌤️ Fetching weather information...")
        
        def fetch_weather():
            try:
                return cached_api_call(
                    category='weather',
                    api_func=lambda: get_weather(flight_info['origin'], actual_departure),
                    use_stale_on_error=True,
                    airport=flight_info['origin'],
                    date=actual_departure.date().isoformat(),
                    hour=actual_departure.hour
                )
            except:
                # Use a safe default if cache is also unavailable
                print(f"   📊 Using safe weather default")
                return get_fallback_weather()
        
        weather = fetch_weather()
        if weather:
            hours_left = weather.get('hours_until_flight', 0)
            time_note = ""
            if hours_left > 6:
                time_note = f" ({hours_left:.0f} hours until departure - current weather)"
            elif hours_left > 0:
                time_note = f" ({hours_left:.0f} hours until departure)"
            
            print(f"   🌤️ {weather['airport']}: {weather['condition']} - {weather['description']}{time_note}")
            print(f"      Temperature {weather['temperature']}°C, Wind {weather['wind_speed']} m/s")
            print(f"      Delay risk: {weather['delay_risk'].upper()}")
            if weather['warning']:
                print(f"      ⚠️ {weather['warning']}")
        else:
            print(f"   ⚠️ Weather data unavailable, assuming normal conditions")
        
        # Calculate additional delay due to weather
        weather_delay = 0
        if weather['delay_risk'] == 'high':
            weather_delay = 30  # Add 30 minutes for severe weather
            print(f"      ⚠️ Additional delay expected due to bad weather: +{weather_delay} min")
        elif weather['delay_risk'] == 'medium':
            weather_delay = 15  # Add 15 minutes for moderate weather
            print(f"      ⚠️ Possible additional delay due to weather: +{weather_delay} min")
        
        total_predicted_delay = predicted_delay + weather_delay
        actual_departure = flight_info['scheduled_time'] + timedelta(minutes=total_predicted_delay)
        
        # Target airport arrival time (2 hours before actual departure)
        airport_arrival_target = actual_departure - timedelta(hours=2)
        
        # Estimate departure time (target airport arrival - estimated required time)
        # Initially assume an average of 1.5 hours
        estimated_departure = airport_arrival_target - timedelta(hours=1, minutes=30)
        
        # Check if time is in the past (except flights from tomorrow onward)
        now = datetime.now()
        if estimated_departure < now:
            # Keep estimated_departure if the flight date is after today
            if actual_departure.date() > now.date():
                # Flight is tomorrow or later - keep estimated_departure as is (future date)
                print(f"   ℹ️ Future-date flight ({actual_departure.date()}) - predicting traffic for that date")
            else:
                # Flight is today but the estimated time has already passed
                estimated_departure = now
                print(f"   ⚠️ The flight has already departed or is imminent.")
        
        # 5. Compute travel time via Google Routes API (caching + resilience)
        print(f"   🗺️ Calculating travel time... ({travel_mode})")
        
        def fetch_travel_time():
            try:
                return cached_api_call(
                    category='travel_time',
                    api_func=lambda: calculate_travel_time(
                        origin=address,
                        destination=flight_info['origin'],
                        travel_mode=travel_mode,
                        departure_time=estimated_departure
                    ),
                    use_stale_on_error=True,
                    origin=address[:50],  # Limit address length
                    destination=flight_info['origin'],
                    mode=travel_mode,
                    hour=estimated_departure.hour
                )
            except:
                # Use historical stats if cache is unavailable
                avg_time = historical_fallback.get_avg_travel_time(
                    address[:50],
                    flight_info['origin'],
                    travel_mode
                )
                print(f"   📊 Using historical average: {avg_time} min")
                return get_fallback_travel_time(travel_mode)
        
        travel_time_result = fetch_travel_time()
        
        if not travel_time_result.get('success'):
            print(f"   ⚠️ Using fallback travel time")
            travel_time_result = get_fallback_travel_time(travel_mode)
        else:
            # Update stats on success
            historical_fallback.update_travel_time(
                address[:50],
                flight_info['origin'],
                travel_mode,
                travel_time_result['duration_minutes']
            )
        
        travel_time_minutes = travel_time_result['duration_minutes']
        print(f"   🚗 Travel time: {travel_time_minutes} min")
        
        # Transit route details
        transit_details = travel_time_result.get('transit_details')
        if transit_details and not travel_time_result.get('fallback_used'):
            print(f"   🚇 Public transit route:")
            for i, detail in enumerate(transit_details, 1):
                vehicle_icon = {
                    'SUBWAY': '🚇',
                    'BUS': '🚌',
                    'TRAIN': '🚂',
                    'RAIL': '🚆'
                }.get(detail['vehicle_type'], '🚌')
                print(f"      {i}. {vehicle_icon} {detail['line']} - {detail['from']} → {detail['to']} ({detail['stops']} stops)")
        
        # 6. Compute TSA security wait time (caching + resilience)
        has_tsa_precheck = flight_info.get('has_tsa_precheck', False)
        terminal = flight_info.get('terminal', None)
        
        def fetch_tsa_wait():
            try:
                return cached_api_call(
                    category='tsa_wait',
                    api_func=lambda: get_tsa_wait_time(
                        airport_code=flight_info['origin'],
                        departure_time=flight_info['scheduled_time'],
                        has_precheck=has_tsa_precheck,
                        terminal=terminal
                    ),
                    use_stale_on_error=True,
                    airport=flight_info['origin'],
                    hour=flight_info['scheduled_time'].hour,
                    precheck=has_tsa_precheck,
                    terminal=terminal or 'unknown'
                )
            except:
                # Use historical stats if cache is unavailable
                avg_wait = historical_fallback.get_avg_tsa_wait(
                    flight_info['origin'],
                    flight_info['scheduled_time'].hour,
                    has_tsa_precheck
                )
                print(f"   📊 Using historical TSA average: {avg_wait} min")
                return avg_wait
        
        tsa_wait_minutes = fetch_tsa_wait()
        
        if isinstance(tsa_wait_minutes, dict):
            # If API response is a dict
            tsa_wait_minutes = tsa_wait_minutes.get('wait_time', get_fallback_tsa_wait(has_tsa_precheck))
        
        # Update stats
        if tsa_wait_minutes and tsa_wait_minutes > 0:
            historical_fallback.update_tsa_wait(
                flight_info['origin'],
                flight_info['scheduled_time'].hour,
                tsa_wait_minutes
            )
        
        print(f"   🔒 TSA wait: {tsa_wait_minutes} min {'(PreCheck)' if has_tsa_precheck else ''}")
        
        # 7. Compute baggage check-in time
        has_checked_baggage = flight_info.get('has_checked_baggage', False)
        baggage_check_minutes = 30 if has_checked_baggage else 0
        if has_checked_baggage:
            print(f"   🧳 Baggage check-in: {baggage_check_minutes} min")
        else:
            print(f"   🎒 Carry-on only (no check-in required)")
        
        # 8. Gate walking time (based on terminal/gate info)
        terminal = flight_info.get('terminal', 'Terminal 4')  # Default: Terminal 4 (international)
        gate = flight_info.get('gate', None)
        
        try:
            gate_walk_minutes = get_gate_walk_time(terminal, gate)
        except Exception as e:
            print(f"   ⚠️ Gate walk time calculation failed: {e}")
            gate_walk_minutes = ResilienceConfig.DEFAULT_GATE_WALK
        
        print(f"   🚶 Gate walk: {gate_walk_minutes} min ({terminal}, Gate {gate if gate else 'N/A'})")
        
        # 9. Compute total required time
        total_time = travel_time_minutes + tsa_wait_minutes + baggage_check_minutes + gate_walk_minutes
        
        # 10. Recommended departure time = target airport arrival - total required time
        recommended_departure = airport_arrival_target - timedelta(minutes=total_time)
        
        print(f"\n   ✅ Calculation complete:")
        print(f"      Scheduled flight departure: {flight_info['scheduled_time'].strftime('%H:%M')}")
        print(f"      Actual expected departure: {actual_departure.strftime('%H:%M')} (+{total_predicted_delay} min delay)")
        print(f"      Target airport arrival: {airport_arrival_target.strftime('%H:%M')} (2 hours before departure)")
        print(f"")
        print(f"      📊 Total time required: {total_time} min ({total_time//60}h {total_time%60}min)")
        print(f"         - 🚗 Travel: {travel_time_minutes} min")
        print(f"         - 🔒 TSA: {tsa_wait_minutes} min")
        if baggage_check_minutes > 0:
            print(f"         - 🧳 Baggage: {baggage_check_minutes} min")
        print(f"         - 🚶 Gate: {gate_walk_minutes} min")
        print(f"")
        print(f"      ✈️ Recommended departure time: {recommended_departure.strftime('%H:%M')}")
        
        # 11. Final recommendation via LLM agent (enhanced resilience)
        print(f"   🤖 Generating LLM recommendation...")
        
        # Build transit route information text
        transit_route_text = ""
        if transit_details:
            transit_route_text = "\n\nPublic transit route:\n"
            for i, detail in enumerate(transit_details, 1):
                vehicle_name = {
                    'SUBWAY': 'Subway',
                    'BUS': 'Bus',
                    'TRAIN': 'Train',
                    'RAIL': 'Rail'
                }.get(detail['vehicle_type'], 'Bus')
                transit_route_text += f"  {i}. {vehicle_name} {detail['line']} - Board at {detail['from']} → Get off at {detail['to']} ({detail['stops']} stops)\n"
        
        # Weather information text
        weather_text = ""
        if weather['delay_risk'] != 'unknown':
            weather_text = f"\n\nWeather information ({weather['airport']}):\n"
            weather_text += f"  - Current: {weather['condition']} - {weather['description']}\n"
            weather_text += f"  - Temperature: {weather['temperature']}°C, Wind speed: {weather['wind_speed']} m/s\n"
            weather_text += f"  - Visibility: {weather['visibility']}m\n"
            weather_text += f"  - Delay risk: {weather['delay_risk'].upper()}\n"
            if weather['warning']:
                weather_text += f"  - Warning: {weather['warning']}\n"
            if weather_delay > 0:
                weather_text += f"  - Expected additional delay: +{weather_delay} min\n"
        
        # English mode names
        travel_mode_en = {
            'DRIVE': 'driving',
            'TRANSIT': 'public transit',
            'WALK': 'walking',
            'BICYCLE': 'bicycle'
        }.get(travel_mode, travel_mode)
        
        # Real-time delay information text (with clear source)
        delay_source_text = ""
        if use_real_time:
            delay_source_text = f"""
- Delay information source: Official airline announcement (real-time API)
- Current status: {real_time_status.get('status', 'N/A')}
- Official announced delay: {real_time_status['delay_minutes']} minutes
- Basis: Real-time flight information directly published by the airline"""
        else:
            delay_source_text = f"""
- Delay information source: AI model prediction (FT-Transformer)
- AI predicted delay: {predicted_delay:.0f} minutes
- Basis: Trained on 60,000+ historical flight data, analysis of same airline/route/time"""
        
        context = f"""
Flight Information:
- Flight: {flight_info.get('flight_number', 'N/A')} ({flight_info.get('airline_name', flight_info['airline_code'])})
- Departure Airport: {flight_info['origin']}
- Scheduled Departure: {flight_info['scheduled_time'].strftime('%Y-%m-%d %H:%M')}
{delay_source_text}
- Operational factors (congestion + FR24/ADS-B previous-leg): +{operational_delay} minutes
  • Area congestion: {congestion_info.get('level', 'unknown')} (score {congestion_info.get('score', 0):.2f}, sample={congestion_info.get('sample_size', 0)}, source={congestion_info.get('source', 'unknown')})
  {f"  (flight count: {congestion_detail.get('current_count', 'N/A')}, historical mean: {congestion_detail.get('historical_mean', 'N/A')}, z-score: {congestion_detail.get('z_score', 'N/A')})" if congestion_detail.get('z_score') is not None else ""}
  • FR24+ADS-B previous leg delay: +{adsb_fr24_info.get('delay_minutes', 0)} minutes
- Weather-related delay: {weather_delay} minutes
- Total expected delay: {total_predicted_delay:.0f} minutes
- Actual expected departure: {actual_departure.strftime('%Y-%m-%d %H:%M')}
{weather_text}
Departure Location:
- Address: {address}

Time Breakdown:
- 🚗 Travel time: {travel_time_minutes} minutes ({travel_mode_en}){transit_route_text}
- 🔒 Security screening: {tsa_wait_minutes} minutes {'(TSA PreCheck)' if has_tsa_precheck else ''}
- 🧳 Baggage check-in: {baggage_check_minutes} minutes {'(check-in required)' if has_checked_baggage else '(carry-on only)'}
- 🚶 Gate walk: {gate_walk_minutes} minutes
- ⏱️ Total time needed: {total_time} minutes

Target airport arrival: {airport_arrival_target.strftime('%Y-%m-%d %H:%M')} (2 hours before actual departure)
📍 Recommended departure time: {recommended_departure.strftime('%Y-%m-%d %H:%M')}
"""
        
        prompt = f"""You are a helpful travel assistant who provides clear, friendly guidance in English.
Based on the following flight departure information, please recommend a departure time in natural, conversational English.

{context}

Please include the following in your response in natural, friendly English:
1. Emphasize the recommended departure time
2. Explain each time component (especially detailed transit routes if using public transportation)
3. Explain the delay prediction basis:
   - If real-time airline data available: "According to the airline's official announcement, a delay of XX minutes is currently expected"
   - If AI prediction: "Based on AI analysis of historical data for the same route/time period, an average delay of XX minutes is predicted"
4. Describe weather conditions and delay risk (including precautions for severe weather)
5. Additional tips (transit card top-up for public transport, transfer precautions, weather preparation, etc.)

Please respond in plain text without JSON or markdown formatting."""
        
        # LLM API call (Gemini or Ollama)
        try:
            if self.use_gemini:
                # Use Gemini (direct API)
                recommendation_text = self.llm_client.generate_text(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2048
                )
            else:
                # Use Ollama
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": "gpt-oss:120b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    recommendation_text = result.get('response', '')
                else:
                    raise Exception(f"Ollama returned status {response.status_code}")
        
        except Exception as e:
            print(f"   ⚠️ LLM call failed: {e}")
            # Fallback: template-based recommendation
            recommendation_text = f"""
✈️ Departure Time Recommendation

Flight {flight_info.get('flight_number', 'N/A')} ({flight_info.get('airline_name', flight_info['airline_code'])})
Scheduled: {flight_info['scheduled_time'].strftime('%Y-%m-%d %H:%M')}
Actual departure: {actual_departure.strftime('%Y-%m-%d %H:%M')} ({total_predicted_delay:.0f} min delay)

📍 Recommended departure time: {recommended_departure.strftime('%H:%M')}

Time breakdown:
- Travel: {travel_time_minutes} min ({travel_mode_en}){transit_route_text}
- TSA: {tsa_wait_minutes} min
- Baggage: {baggage_check_minutes} min
- Gate walk: {gate_walk_minutes} min
- Total: {total_time} min

Operational factors: +{operational_delay} min (congestion {congestion_info.get('level', 'unknown')}, FR24+ADS-B previous leg +{adsb_fr24_info.get('delay_minutes', 0)} min)
Weather: {weather['condition']} (delay risk {weather['delay_risk']}, +{weather_delay} min)
"""
        
        return {
            'success': True,
            'recommendation': recommendation_text,
            'details': {
                'recommended_departure': recommended_departure.strftime('%Y-%m-%d %H:%M'),
                'flight_time': flight_info['scheduled_time'].strftime('%Y-%m-%d %H:%M'),
                'actual_departure': actual_departure.strftime('%Y-%m-%d %H:%M'),
                'travel_time': travel_time_minutes,
                'tsa_wait': tsa_wait_minutes,
                'baggage_check': baggage_check_minutes,
                'predicted_delay': predicted_delay,
                'operational_delay': operational_delay,
                'congestion_level': congestion_info.get('level', 'unknown'),
                'congestion_score': congestion_info.get('score', 0.0),
                'congestion_source': congestion_info.get('source', 'unknown'),
                'congestion_details': congestion_info.get('details', {}),
                'adsb_fr24_delay': adsb_fr24_info.get('delay_minutes', 0),
                'adsb_fr24_source': adsb_fr24_info.get('source', 'none'),
                'adsb_fr24_reason': adsb_fr24_info.get('reason', 'unknown'),
                'adsb_fr24_validation_mismatch': bool(adsb_fr24_info.get('validation_mismatch', False)),
                'total_time': total_time
            }
        }


def main():
    """Interactive flight departure prediction."""
    import sys
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 60)
    print("  Hybrid Departure Time Predictor")
    print("=" * 60)

    # ── Collect flight information from user ──
    print("\nEnter your flight information:\n")

    address = input("  Departure address (e.g. 123 Main St, New York, NY): ").strip()
    if not address:
        address = "Times Square, New York, NY"
        print(f"    → Using default: {address}")

    flight_number = input("  Flight number (e.g. B6123): ").strip().upper()
    if not flight_number:
        flight_number = "B6123"
        print(f"    → Using default: {flight_number}")

    airline_code = input(f"  Airline code (e.g. B6, AA, DL) [{flight_number[:2]}]: ").strip().upper()
    if not airline_code:
        # Extract from flight number (first 2 chars)
        airline_code = ''.join(c for c in flight_number if c.isalpha())[:2]
        if not airline_code:
            airline_code = flight_number[:2]
        print(f"    → Using: {airline_code}")

    origin = input("  Origin airport code [JFK]: ").strip().upper()
    if not origin:
        origin = "JFK"
        print(f"    → Using default: {origin}")

    dest = input("  Destination airport code (e.g. LAX): ").strip().upper()
    if not dest:
        dest = "LAX"
        print(f"    → Using default: {dest}")

    # Departure date/time
    date_str = input("  Departure date (YYYY-MM-DD) [today]: ").strip()
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        print(f"    → Using today: {date_str}")

    time_str = input("  Departure time (HH:MM, 24h) [14:00]: ").strip()
    if not time_str:
        time_str = "14:00"
        print(f"    → Using default: {time_str}")

    try:
        scheduled_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        print(f"  ⚠️ Invalid date/time format. Using today 14:00.")
        scheduled_time = datetime.now().replace(hour=14, minute=0, second=0, microsecond=0)

    baggage_input = input("  Checked baggage? (y/n) [n]: ").strip().lower()
    has_checked_baggage = baggage_input in ('y', 'yes')

    precheck_input = input("  TSA PreCheck? (y/n) [n]: ").strip().lower()
    has_tsa_precheck = precheck_input in ('y', 'yes')

    print("\n  Travel mode options: DRIVE, TRANSIT, WALK, BICYCLE")
    travel_mode = input("  Travel mode [DRIVE]: ").strip().upper()
    if travel_mode not in ('DRIVE', 'TRANSIT', 'WALK', 'BICYCLE'):
        travel_mode = "DRIVE"
        print(f"    → Using default: {travel_mode}")

    # ── Build flight_info ──
    flight_info = {
        'airline_code': airline_code,
        'flight_number': flight_number,
        'origin': origin,
        'dest': dest,
        'scheduled_time': scheduled_time,
        'has_checked_baggage': has_checked_baggage,
        'has_tsa_precheck': has_tsa_precheck,
    }

    # ── Summary before prediction ──
    print(f"\n{'='*60}")
    print(f"  Flight: {flight_number} ({airline_code})")
    print(f"  Route:  {origin} → {dest}")
    print(f"  Date:   {scheduled_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  From:   {address}")
    print(f"  Mode:   {travel_mode}")
    print(f"  Baggage: {'Yes' if has_checked_baggage else 'No'}")
    print(f"  TSA PreCheck: {'Yes' if has_tsa_precheck else 'No'}")
    print(f"{'='*60}")

    # ── Load model & predict ──
    print("\nLoading model...")
    predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')

    result = predictor.recommend_departure(
        address=address,
        flight_info=flight_info,
        travel_mode=travel_mode,
    )

    if result['success']:
        print(f"\n{'='*60}")
        print(f"  RECOMMENDATION")
        print(f"{'='*60}")
        print(f"\n{result['recommendation']}")

        details = result.get('details', {})
        print(f"\n{'='*60}")
        print(f"  DETAILS")
        print(f"{'='*60}")
        print(f"  Recommended departure : {details.get('recommended_departure', 'N/A')}")
        print(f"  Scheduled flight time : {details.get('flight_time', 'N/A')}")
        print(f"  Actual expected dept  : {details.get('actual_departure', 'N/A')}")
        print(f"  Predicted delay       : {details.get('predicted_delay', 0):.0f} min")
        print(f"  Congestion level      : {details.get('congestion_level', 'N/A')} "
              f"(source: {details.get('congestion_source', 'N/A')})")
        cong_details = details.get('congestion_details', {})
        if cong_details.get('current_count') is not None:
            print(f"    Flight count: {cong_details.get('current_count')}, "
                  f"Mean: {cong_details.get('historical_mean', 'N/A')}, "
                  f"Z-score: {cong_details.get('z_score', 'N/A')}")
        print(f"  Operational delay     : +{details.get('operational_delay', 0)} min")
        print(f"  Travel time           : {details.get('travel_time', 0)} min")
        print(f"  TSA wait              : {details.get('tsa_wait', 0)} min")
        print(f"  Total time needed     : {details.get('total_time', 0)} min")
    else:
        print(f"\n❌ Error: {result['error']}")


if __name__ == '__main__':
    main()
