"""
하이브리드 출발 시간 예측 시스템

Architecture:
1. Transformer Model: 항공편 지연 시간 예측 (학습된 모델)
2. Google Routes API: 주소 → 공항 이동 시간
3. TSA Wait Time: 보안검색 대기시간
4. Baggage Check: 수하물 체크인 시간
5. LLM Agent: 최종 출발 시간 추천 (한국어)
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.google_routes import calculate_travel_time
from utils.tsa_wait_time import get_tsa_wait_time
from utils.weather_google import get_weather
from utils.flight_status_checker import check_flight
from utils.gate_walk_time import get_gate_walk_time
from utils.operational_factors import OperationalFactorsAnalyzer
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


class FeatureTokenizer(nn.Module):
    """각 Feature를 개별 임베딩으로 변환"""
    def __init__(self, num_features, d_token):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        
        # 각 feature에 대한 linear transformation
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(num_features)
        ])
        
        # CLS 토큰
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
    """Transformer 기반 지연 예측 모델"""
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
    """하이브리드 출발 시간 예측 시스템"""
    
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
            model_path: 학습된 Transformer 모델 경로 (로컬 또는 GCS 경로)
            use_gcs: GCS에서 모델 로드 여부
            gcs_bucket: GCS 버킷 이름 (use_gcs=True일 때 필요)
            use_gemini: Gemini 사용 여부 (True면 Ollama 대신 Gemini)
            gemini_project_id: GCP 프로젝트 ID (환경변수에서도 가능)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket or os.getenv('GCS_MODEL_BUCKET')
        self.use_gemini = use_gemini or os.getenv('USE_GEMINI', 'false').lower() == 'true'
        self.gemini_project_id = gemini_project_id or os.getenv('GCP_PROJECT_ID')
        
        self.load_model(model_path)
        
        # LLM 클라이언트 초기화
        if self.use_gemini:
            print("🤖 Using Google Gemini for LLM")
            from utils.gemini_direct_client import GeminiDirectClient
            self.llm_client = GeminiDirectClient()
        else:
            print("🤖 Using Ollama for LLM")
            self.ollama_url = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
            self.llm_client = None

        # 운항 컨텍스트(혼잡도/직전편 지연) 분석기
        self.operational_analyzer = OperationalFactorsAnalyzer()
        
    def load_model(self, model_path):
        """학습된 모델 로드 (로컬 또는 GCS)"""
        if self.use_gcs:
            # GCS에서 직접 로드 (다운로드 없이 메모리에만)
            print(f"📦 Loading model from GCS: gs://{self.gcs_bucket}/{model_path}")
            from utils.gcs_model_loader import GCSModelLoader
            
            if not self.gcs_bucket:
                raise ValueError("GCS bucket name required. Set GCS_MODEL_BUCKET env var or pass gcs_bucket parameter")
            
            loader = GCSModelLoader(self.gcs_bucket)
            package = loader.load_pickle_model(model_path)
        else:
            # 로컬에서 로드
            print(f"📦 모델 로딩: {model_path}")
            with open(model_path, 'rb') as f:
                package = pickle.load(f)
        
        # 모델 재생성 (FT-Transformer)
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
            # 기존 Transformer (호환성)
            self.model = FlightDelayTransformer(
                input_dim=config['input_dim'],
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            ).to(self.device)
        
        self.model.load_state_dict(package['model_state_dict'])
        self.model.eval()
        
        # 전처리 도구
        self.label_encoders = package['label_encoders']
        self.scaler = package['scaler']
        self.feature_columns = package['feature_columns']
        self.test_metrics = package['test_metrics']
        
        print(f"✅ 모델 로드 완료 ({config.get('model_type', 'Transformer')})")
        print(f"   MAE: {self.test_metrics['mae']:.2f}분")
        print(f"   RMSE: {self.test_metrics['rmse']:.2f}분")
        print(f"   R²: {self.test_metrics['r2']:.4f}")
    
    def predict_delay(self, airline_code, origin, dest, flight_datetime):
        """
        항공편 지연 시간 예측
        
        Args:
            airline_code: 항공사 코드 (예: 'B6')
            origin: 출발지 공항 코드 (예: 'JFK')
            dest: 도착지 공항 코드 (예: 'LAX')
            flight_datetime: 출발 예정 시간 (datetime 객체)
        
        Returns:
            predicted_delay: 예상 지연 시간 (분)
        """
        # Feature 생성
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
        
        # 범주형 변수 인코딩
        unknown_items = []
        try:
            encoded_features = features.copy()
            for col in ['op_unique_carrier', 'origin', 'dest']:
                if col in self.label_encoders:
                    try:
                        encoded_features[col] = self.label_encoders[col].transform([features[col]])[0]
                    except ValueError:
                        # 학습 데이터에 없는 값은 0으로 설정
                        encoded_features[col] = 0
                        col_name = {'op_unique_carrier': '항공사', 'origin': '출발공항', 'dest': '도착공항'}[col]
                        unknown_items.append(f"{col_name} '{features[col]}'")
                else:
                    # label_encoder 자체가 없는 경우
                    encoded_features[col] = 0
            
            if unknown_items:
                print(f"   ℹ️ 학습 데이터에 없는 항목: {', '.join(unknown_items)} (유사 패턴 기반 예측)")
        except Exception as e:
            # 인코딩 실패 시 평균 지연 시간 반환
            print(f"   ⚠️ 예측 오류: {str(e)} (기본값 사용)")
            return 15.0  # 기본값
        
        # Feature 배열 생성
        X = np.array([[encoded_features[col] for col in self.feature_columns]])
        
        # 숫자형 features 정규화
        numeric_indices = [self.feature_columns.index(col) 
                          for col in ['hour', 'month', 'day_of_week', 'day_of_month']]
        X[:, numeric_indices] = self.scaler.transform(X[:, numeric_indices])
        
        # 예측
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predicted_delay = self.model(X_tensor).cpu().numpy()[0][0]
        
        return float(predicted_delay)
    
    def recommend_departure(self, address, flight_info, travel_mode='DRIVE'):
        """
        하이브리드 시스템으로 출발 시간 추천 (복원력 강화)
        
        Args:
            address: 출발 주소
            flight_info: 항공편 정보 dict
                - airline_code: 항공사 코드
                - flight_number: 항공편 번호
                - origin: 출발 공항
                - dest: 도착 공항
                - scheduled_time: 출발 예정 시간 (datetime)
                - has_checked_baggage: 수하물 체크인 여부 (optional, default=False)
                - has_tsa_precheck: TSA PreCheck 보유 여부 (optional, default=False)
            travel_mode: 이동 수단 ('DRIVE', 'TRANSIT', 'WALK', 'BICYCLE')
        
        Returns:
            recommendation: LLM 추천 결과 (dict)
        """
        print(f"\n🔍 Hybrid prediction started...")
        
        # 입력 데이터 검증 및 보정
        try:
            flight_info = validate_flight_info(flight_info)
            print(f"   ✅ Flight information validated")
        except ValueError as e:
            return {
                'success': False,
                'error': f'Invalid flight information: {e}'
            }
        
        # 1. 실시간 항공편 상태 확인 (캐싱 + 복원력)
        print(f"   🛫 Checking real-time flight status...")
        
        def fetch_flight_status():
            # 캐시 확인 먼저
            try:
                return cached_api_call(
                    category='flight_status',
                    api_func=lambda: check_flight(flight_info['flight_number']),
                    use_stale_on_error=True,
                    flight_number=flight_info['flight_number'],
                    date=flight_info['scheduled_time'].date().isoformat()
                )
            except:
                # 캐시도 없으면 과거 통계 사용
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
        
        # 실시간 정보가 있고 날짜가 일치하는 경우만 사용
        use_real_time = False
        if real_time_status and real_time_status.get('scheduled_departure') and not real_time_status.get('fallback_used'):
            # scheduled_departure가 문자열인 경우 datetime으로 변환
            scheduled_dep = real_time_status['scheduled_departure']
            if isinstance(scheduled_dep, str):
                from dateutil import parser
                scheduled_dep = parser.parse(scheduled_dep)
            
            api_date = scheduled_dep.date()
            ticket_date = flight_info['scheduled_time'].date()
            
            # 날짜가 일치하고 지연 정보가 있으면 사용
            if api_date == ticket_date and real_time_status.get('is_delayed'):
                real_delay = real_time_status['delay_minutes']
                print(f"   ⚠️ Real-time delay info: {real_delay} min")
                print(f"   📡 Airline announcement: {real_time_status['status_kr']}")
                if real_time_status.get('estimated_departure'):
                    est_dep = real_time_status['estimated_departure']
                    if isinstance(est_dep, str):
                        est_dep = parser.parse(est_dep)
                    print(f"   🕐 Estimated departure: {est_dep.strftime('%H:%M')}")
                
                # 실시간 정보를 우선 사용
                predicted_delay = real_delay
                use_real_time = True
                
                # 통계 업데이트
                route = f"{flight_info['origin']}-{flight_info['dest']}"
                historical_fallback.update_flight_delay(
                    flight_info['airline_code'], 
                    route, 
                    real_delay
                )
            elif api_date != ticket_date:
                print(f"   ⚠️ API date mismatch (API: {api_date}, Ticket: {ticket_date}) - Using ticket info")
        
        if not use_real_time:
            # 2. 실시간 정보가 없으면 Transformer로 지연 시간 예측
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
        
        # 3. 운항 컨텍스트 반영 (JFK 50마일 혼잡도 + 직전편 지연)
        operational_delay = 0
        congestion_info = {
            'level': 'unknown',
            'score': 0.0,
            'sample_size': 0,
            'recommended_extra_delay': 0
        }
        previous_leg_info = {
            'found': False,
            'delay_minutes': 0,
            'propagated_delay': 0
        }

        if self.operational_analyzer.enabled and flight_info['origin'] == 'JFK':
            print(f"   🛩️ Analyzing JFK-area congestion and previous leg delay...")

            def fetch_congestion():
                return cached_api_call(
                    category='operational_congestion',
                    api_func=lambda: self.operational_analyzer.get_jfk_area_congestion(
                        flight_info['scheduled_time']
                    ),
                    use_stale_on_error=True,
                    origin=flight_info['origin'],
                    hour=flight_info['scheduled_time'].hour
                )

            def fetch_previous_leg():
                return cached_api_call(
                    category='previous_leg_delay',
                    api_func=lambda: self.operational_analyzer.get_previous_leg_delay(
                        flight_info['flight_number'],
                        flight_info['scheduled_time']
                    ),
                    use_stale_on_error=True,
                    flight_number=flight_info['flight_number'],
                    date=flight_info['scheduled_time'].date().isoformat()
                )

            try:
                congestion_info = fetch_congestion() or congestion_info
                previous_leg_info = fetch_previous_leg() or previous_leg_info

                congestion_delay = int(congestion_info.get('recommended_extra_delay', 0) or 0)
                previous_leg_delay = int(previous_leg_info.get('propagated_delay', 0) or 0)
                operational_delay = congestion_delay + previous_leg_delay

                print(
                    f"      • Area congestion: {congestion_info.get('level', 'unknown')} "
                    f"(score {congestion_info.get('score', 0):.2f}, n={congestion_info.get('sample_size', 0)}) "
                    f"→ +{congestion_delay} min"
                )
                if previous_leg_info.get('found'):
                    print(
                        f"      • Previous leg delay: {previous_leg_info.get('delay_minutes', 0)} min "
                        f"(propagated +{previous_leg_delay} min)"
                    )
                else:
                    print("      • Previous leg delay: unavailable (0 min applied)")

                if operational_delay > 0:
                    print(f"      ⚠️ Operational adjustment applied: +{operational_delay} min")
            except Exception as e:
                print(f"   ⚠️ Operational factor analysis failed: {e}")
        else:
            print("   ℹ️ Operational factor analysis skipped (non-JFK origin or no API key)")

        predicted_delay += operational_delay

        # 4. 실제 출발 시간 계산 (scheduled + 예상지연)
        actual_departure = flight_info['scheduled_time'] + timedelta(minutes=predicted_delay)
        
        # 5. 날씨 정보 조회 (캐싱 + 복원력)
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
                # 캐시도 없으면 안전한 기본값
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
        
        # 날씨에 따른 추가 지연 시간 계산
        weather_delay = 0
        if weather['delay_risk'] == 'high':
            weather_delay = 30  # 악천후 시 30분 추가
            print(f"      ⚠️ Additional delay expected due to bad weather: +{weather_delay} min")
        elif weather['delay_risk'] == 'medium':
            weather_delay = 15  # 보통 날씨 15분 추가
            print(f"      ⚠️ Possible additional delay due to weather: +{weather_delay} min")
        
        total_predicted_delay = predicted_delay + weather_delay
        actual_departure = flight_info['scheduled_time'] + timedelta(minutes=total_predicted_delay)
        
        # 공항 도착 목표 시간 (실제 출발 2시간 전)
        airport_arrival_target = actual_departure - timedelta(hours=2)
        
        # 추정 출발 시간 계산 (공항 도착 목표 - 예상 소요시간)
        # 초기에는 평균 1.5시간으로 가정
        estimated_departure = airport_arrival_target - timedelta(hours=1, minutes=30)
        
        # 과거 시간 체크 (단, 내일 이후 비행이면 괜찮음)
        now = datetime.now()
        if estimated_departure < now:
            # 비행기 출발이 오늘보다 미래라면 estimated_departure 유지
            if actual_departure.date() > now.date():
                # 내일 이후 비행 - estimated_departure 그대로 사용 (미래 날짜)
                print(f"   ℹ️ 미래 날짜 항공편 ({actual_departure.date()}) - 해당 날짜 기준 교통량 예측")
            else:
                # 오늘 비행인데 이미 지난 시간
                estimated_departure = now
                print(f"   ⚠️ 비행기가 이미 출발했거나 임박했습니다.")
        
        # 5. Google Routes API로 이동 시간 계산 (캐싱 + 복원력)
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
                    origin=address[:50],  # 주소 길이 제한
                    destination=flight_info['origin'],
                    mode=travel_mode,
                    hour=estimated_departure.hour
                )
            except:
                # 캐시도 없으면 과거 통계 사용
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
            # 성공 시 통계 업데이트
            historical_fallback.update_travel_time(
                address[:50],
                flight_info['origin'],
                travel_mode,
                travel_time_result['duration_minutes']
            )
        
        travel_time_minutes = travel_time_result['duration_minutes']
        print(f"   🚗 Travel time: {travel_time_minutes} min")
        
        # Transit 세부 경로 정보
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
        
        # 6. TSA 보안검색 대기시간 계산 (캐싱 + 복원력)
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
                # 캐시도 없으면 과거 통계 사용
                avg_wait = historical_fallback.get_avg_tsa_wait(
                    flight_info['origin'],
                    flight_info['scheduled_time'].hour,
                    has_tsa_precheck
                )
                print(f"   📊 Using historical TSA average: {avg_wait} min")
                return avg_wait
        
        tsa_wait_minutes = fetch_tsa_wait()
        
        if isinstance(tsa_wait_minutes, dict):
            # API 응답이 dict 형식인 경우
            tsa_wait_minutes = tsa_wait_minutes.get('wait_time', get_fallback_tsa_wait(has_tsa_precheck))
        
        # 통계 업데이트
        if tsa_wait_minutes and tsa_wait_minutes > 0:
            historical_fallback.update_tsa_wait(
                flight_info['origin'],
                flight_info['scheduled_time'].hour,
                tsa_wait_minutes
            )
        
        print(f"   🔒 TSA wait: {tsa_wait_minutes} min {'(PreCheck)' if has_tsa_precheck else ''}")
        
        # 7. 수하물 체크인 시간 계산
        has_checked_baggage = flight_info.get('has_checked_baggage', False)
        baggage_check_minutes = 30 if has_checked_baggage else 0
        if has_checked_baggage:
            print(f"   🧳 Baggage check-in: {baggage_check_minutes} min")
        else:
            print(f"   🎒 Carry-on only (no check-in required)")
        
        # 8. 게이트 이동 시간 (터미널/게이트 정보 기반)
        terminal = flight_info.get('terminal', 'Terminal 4')  # 기본값: Terminal 4 (국제선)
        gate = flight_info.get('gate', None)
        
        try:
            gate_walk_minutes = get_gate_walk_time(terminal, gate)
        except Exception as e:
            print(f"   ⚠️ Gate walk time calculation failed: {e}")
            gate_walk_minutes = ResilienceConfig.DEFAULT_GATE_WALK
        
        print(f"   🚶 Gate walk: {gate_walk_minutes} min ({terminal}, Gate {gate if gate else 'N/A'})")
        
        # 9. 총 소요 시간 계산
        total_time = travel_time_minutes + tsa_wait_minutes + baggage_check_minutes + gate_walk_minutes
        
        # 10. 추천 출발 시간 = 공항 도착 목표 - 총 소요 시간
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
        
        # 11. LLM Agent로 최종 추천 (복원력 강화)
        print(f"   🤖 Generating LLM recommendation...")
        
        # Transit 경로 정보 텍스트 생성
        transit_route_text = ""
        if transit_details:
            transit_route_text = "\n\n대중교통 경로:\n"
            for i, detail in enumerate(transit_details, 1):
                vehicle_name = {
                    'SUBWAY': '지하철',
                    'BUS': '버스',
                    'TRAIN': '기차',
                    'RAIL': '전철'
                }.get(detail['vehicle_type'], '버스')
                transit_route_text += f"  {i}. {vehicle_name} {detail['line']}번 - {detail['from']}에서 탑승 → {detail['to']}에서 하차 ({detail['stops']}정거장)\n"
        
        # 날씨 정보 텍스트
        weather_text = ""
        if weather['delay_risk'] != 'unknown':
            weather_text = f"\n\n날씨 정보 ({weather['airport']}):\n"
            weather_text += f"  - 현재: {weather['condition']} - {weather['description']}\n"
            weather_text += f"  - 온도: {weather['temperature']}°C, 풍속: {weather['wind_speed']} m/s\n"
            weather_text += f"  - 가시거리: {weather['visibility']}m\n"
            weather_text += f"  - 지연 위험도: {weather['delay_risk'].upper()}\n"
            if weather['warning']:
                weather_text += f"  - 경고: {weather['warning']}\n"
            if weather_delay > 0:
                weather_text += f"  - 예상 추가 지연: +{weather_delay}분\n"
        
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
- Operational factors (JFK 50-mile congestion + previous-leg propagation): +{operational_delay} minutes
  • Area congestion: {congestion_info.get('level', 'unknown')} (score {congestion_info.get('score', 0):.2f}, sample={congestion_info.get('sample_size', 0)})
  • Previous leg propagated delay: +{previous_leg_info.get('propagated_delay', 0)} minutes
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
        
        # LLM API 호출 (Gemini 또는 Ollama)
        try:
            if self.use_gemini:
                # Gemini 사용 (Direct API)
                recommendation_text = self.llm_client.generate_text(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=2048
                )
            else:
                # Ollama 사용
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
            # Fallback: 템플릿 기반 추천
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

Operational factors: +{operational_delay} min (congestion {congestion_info.get('level', 'unknown')}, previous leg +{previous_leg_info.get('propagated_delay', 0)} min)
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
                'previous_leg_delay': previous_leg_info.get('delay_minutes', 0),
                'total_time': total_time
            }
        }


def main():
    """사용 예시"""
    # 모델 로드
    predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')
    
    # 테스트 케이스
    test_cases = [
        {
            'address': '123 Main St, New York, NY',
            'flight_info': {
                'airline_code': 'B6',
                'airline_name': 'JetBlue Airways',
                'flight_number': 'B6123',
                'origin': 'JFK',
                'dest': 'LAX',
                'scheduled_time': datetime(2026, 2, 5, 14, 30),
                'has_checked_baggage': True,
                'has_tsa_precheck': False
            },
            'travel_mode': 'DRIVE'
        },
        {
            'address': 'Times Square, New York, NY',
            'flight_info': {
                'airline_code': 'AA',
                'airline_name': 'American Airlines',
                'flight_number': 'AA100',
                'origin': 'JFK',
                'dest': 'MIA',
                'scheduled_time': datetime(2026, 2, 5, 18, 0),
                'has_checked_baggage': False,
                'has_tsa_precheck': True
            },
            'travel_mode': 'TRANSIT'
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"테스트 케이스 {i}")
        print(f"{'='*60}")
        
        result = predictor.recommend_departure(
            address=test['address'],
            flight_info=test['flight_info'],
            travel_mode=test['travel_mode']
        )
        
        if result['success']:
            print(f"\n✅ 추천 결과:")
            print(f"   출발 주소: {test['address']}")
            print(f"   항공편: {test['flight_info']['flight_number']}")
            print(f"   예정 출발: {test['flight_info']['scheduled_time'].strftime('%Y-%m-%d %H:%M')}")
            print(f"\n{result['recommendation']}")
        else:
            print(f"\n❌ 오류: {result['error']}")


if __name__ == '__main__':
    main()
