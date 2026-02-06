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
    
    def __init__(self, model_path='models/delay_predictor_full.pkl'):
        """
        Args:
            model_path: 학습된 Transformer 모델 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        self.ollama_url = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
        
    def load_model(self, model_path):
        """학습된 모델 로드"""
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
        하이브리드 시스템으로 출발 시간 추천
        
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
        print(f"\n🔍 하이브리드 예측 시작...")
        
        # 1. 실시간 항공편 상태 확인 (AviationStack API)
        print(f"   🛫 실시간 항공편 상태 확인 중...")
        real_time_status = check_flight(flight_info['flight_number'])
        
        # 실시간 정보가 있고 날짜가 일치하는 경우만 사용
        use_real_time = False
        if real_time_status and real_time_status.get('scheduled_departure'):
            api_date = real_time_status['scheduled_departure'].date()
            ticket_date = flight_info['scheduled_time'].date()
            
            # 날짜가 일치하고 지연 정보가 있으면 사용
            if api_date == ticket_date and real_time_status.get('is_delayed'):
                real_delay = real_time_status['delay_minutes']
                print(f"   ⚠️ 실시간 지연 정보: {real_delay}분")
                print(f"   📡 항공사 발표: {real_time_status['status_kr']}")
                if real_time_status.get('estimated_departure'):
                    print(f"   🕐 예상 출발: {real_time_status['estimated_departure'].strftime('%H:%M')}")
                
                # 실시간 정보를 우선 사용
                predicted_delay = real_delay
                use_real_time = True
            elif api_date != ticket_date:
                print(f"   ⚠️ API 데이터 날짜 불일치 (API: {api_date}, 티켓: {ticket_date}) - 티켓 정보 우선 사용")
        
        if not use_real_time:
            # 2. 실시간 정보가 없으면 Transformer로 지연 시간 예측
            predicted_delay = self.predict_delay(
                airline_code=flight_info['airline_code'],
                origin=flight_info['origin'],
                dest=flight_info['dest'],
                flight_datetime=flight_info['scheduled_time']
            )
            print(f"   📊 예상 지연: {predicted_delay:.1f}분 (AI 예측)")
            use_real_time = False
        
        # 3. 실제 출발 시간 계산 (scheduled + 예상지연)
        actual_departure = flight_info['scheduled_time'] + timedelta(minutes=predicted_delay)
        
        # 4. 날씨 정보 조회 (출발 시간 기준)
        print(f"   🌤️ 날씨 정보 조회 중...")
        weather = get_weather(flight_info['origin'], actual_departure)  # 실제 출발 시간 기준
        
        if weather['delay_risk'] != 'unknown':
            hours_left = weather.get('hours_until_flight', 0)
            time_note = ""
            if hours_left > 6:
                time_note = f" (출발까지 {hours_left:.0f}시간 - 현재 날씨 기준)"
            elif hours_left > 0:
                time_note = f" (출발까지 {hours_left:.0f}시간)"
            
            print(f"   🌤️ {weather['airport']}: {weather['condition']} - {weather['description']}{time_note}")
            print(f"      온도 {weather['temperature']}°C, 풍속 {weather['wind_speed']} m/s")
            print(f"      지연 위험도: {weather['delay_risk'].upper()}")
            if weather['warning']:
                print(f"      ⚠️ {weather['warning']}")
        
        # 날씨에 따른 추가 지연 시간 계산
        weather_delay = 0
        if weather['delay_risk'] == 'high':
            weather_delay = 30  # 악천후 시 30분 추가
            print(f"      ⚠️ 악천후로 인한 추가 지연 예상: +{weather_delay}분")
        elif weather['delay_risk'] == 'medium':
            weather_delay = 15  # 보통 날씨 15분 추가
            print(f"      ⚠️ 날씨로 인한 추가 지연 가능: +{weather_delay}분")
        
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
        
        # 4. Google Routes API로 이동 시간 계산
        print(f"   🗺️ 이동 시간 계산 중... ({travel_mode})")
        travel_time_result = calculate_travel_time(
            origin=address,
            destination=flight_info['origin'],
            travel_mode=travel_mode,
            departure_time=estimated_departure
        )
        
        if not travel_time_result['success']:
            return {
                'success': False,
                'error': travel_time_result['error']
            }
        
        travel_time_minutes = travel_time_result['duration_minutes']
        print(f"   🚗 이동 시간: {travel_time_minutes}분")
        
        # Transit 세부 경로 정보
        transit_details = travel_time_result.get('transit_details')
        if transit_details:
            print(f"   🚇 대중교통 경로:")
            for i, detail in enumerate(transit_details, 1):
                vehicle_icon = {
                    'SUBWAY': '🚇',
                    'BUS': '🚌',
                    'TRAIN': '🚂',
                    'RAIL': '🚆'
                }.get(detail['vehicle_type'], '🚌')
                print(f"      {i}. {vehicle_icon} {detail['line']} - {detail['from']} → {detail['to']} ({detail['stops']}정거장)")
        
        # 5. TSA 보안검색 대기시간 계산
        has_tsa_precheck = flight_info.get('has_tsa_precheck', False)
        tsa_wait_minutes = get_tsa_wait_time(
            airport_code=flight_info['origin'],
            departure_time=flight_info['scheduled_time'],
            has_precheck=has_tsa_precheck
        )
        print(f"   🔒 TSA 대기시간: {tsa_wait_minutes}분 {'(PreCheck)' if has_tsa_precheck else ''}")
        
        # 6. 수하물 체크인 시간 계산
        has_checked_baggage = flight_info.get('has_checked_baggage', False)
        baggage_check_minutes = 30 if has_checked_baggage else 0
        if has_checked_baggage:
            print(f"   🧳 수하물 체크인: {baggage_check_minutes}분")
        else:
            print(f"   🎒 기내 반입만 (체크인 불필요)")
        
        # 7. 게이트 이동 시간 (터미널/게이트 정보 기반)
        terminal = flight_info.get('terminal', 'Terminal 4')  # 기본값: Terminal 4 (국제선)
        gate = flight_info.get('gate', None)
        gate_walk_minutes = get_gate_walk_time(terminal, gate)
        
        print(f"   🚶 게이트 이동: {gate_walk_minutes}분 ({terminal}, Gate {gate if gate else 'N/A'})")
        
        # 8. 총 소요 시간 계산
        total_time = travel_time_minutes + tsa_wait_minutes + baggage_check_minutes + gate_walk_minutes
        
        # 9. 추천 출발 시간 = 공항 도착 목표 - 총 소요 시간
        recommended_departure = airport_arrival_target - timedelta(minutes=total_time)
        
        print(f"\n   ✅ 계산 완료:")
        print(f"      항공편 예정 출발: {flight_info['scheduled_time'].strftime('%H:%M')}")
        print(f"      예상 실제 출발: {actual_departure.strftime('%H:%M')} (지연 +{total_predicted_delay}분)")
        print(f"      공항 도착 목표: {airport_arrival_target.strftime('%H:%M')} (출발 2시간 전)")
        print(f"")
        print(f"      📊 총 소요 시간: {total_time}분 ({total_time//60}시간 {total_time%60}분)")
        print(f"         - 🚗 이동: {travel_time_minutes}분")
        print(f"         - 🔒 TSA: {tsa_wait_minutes}분")
        if baggage_check_minutes > 0:
            print(f"         - 🧳 수하물: {baggage_check_minutes}분")
        print(f"         - 🚶 게이트: {gate_walk_minutes}분")
        print(f"")
        print(f"      ✈️ 추천 출발 시간: {recommended_departure.strftime('%H:%M')}")
        
        # 10. LLM Agent로 최종 추천
        print(f"   🤖 LLM 추천 생성 중...")
        
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
        
        # 이동 수단 한글명
        travel_mode_kr = {
            'DRIVE': '자동차',
            'TRANSIT': '대중교통',
            'WALK': '도보',
            'BICYCLE': '자전거'
        }.get(travel_mode, travel_mode)
        
        # 실시간 지연 정보 텍스트 (명확한 근거 포함)
        delay_source_text = ""
        if use_real_time:
            delay_source_text = f"""
- 지연 정보 출처: 항공사 공식 발표 (실시간 API)
- 현재 상태: {real_time_status['status_kr']}
- 공식 발표 지연: {real_time_status['delay_minutes']}분
- 근거: 항공사가 직접 발표한 실시간 운항 정보"""
        else:
            delay_source_text = f"""
- 지연 정보 출처: AI 모델 예측 (FT-Transformer)
- AI 예측 지연: {predicted_delay:.0f}분
- 근거: 과거 60,000+ 항공편 데이터 학습, 동일 항공사/노선/시간대 통계 분석"""
        
        context = f"""
비행 정보:
- 항공편: {flight_info.get('flight_number', 'N/A')} ({flight_info.get('airline_name', flight_info['airline_code'])})
- 출발 공항: {flight_info['origin']}
- 예정 출발 시각: {flight_info['scheduled_time'].strftime('%Y-%m-%d %H:%M')}
{delay_source_text}
- 날씨 추가 지연: {weather_delay}분
- 총 예상 지연: {total_predicted_delay:.0f}분
- 실제 예상 출발: {actual_departure.strftime('%Y-%m-%d %H:%M')}
{weather_text}
출발 위치:
- 주소: {address}

소요 시간 계산:
- 🚗 이동 시간: {travel_time_minutes}분 ({travel_mode_kr}){transit_route_text}
- 🔒 보안 검색: {tsa_wait_minutes}분 {'(TSA PreCheck)' if has_tsa_precheck else ''}
- 🧳 수하물 체크인: {baggage_check_minutes}분 {'(체크인 필요)' if has_checked_baggage else '(기내 반입만)'}
- 🚶 게이트 이동: {gate_walk_minutes}분
- ⏱️ 총 소요 시간: {total_time}분

공항 도착 목표: {airport_arrival_target.strftime('%Y-%m-%d %H:%M')} (실제 출발 2시간 전)
📍 추천 출발 시간: {recommended_departure.strftime('%Y-%m-%d %H:%M')}
"""
        
        prompt = f"""당신은 한국어로 친절하게 안내하는 여행 어시스턴트입니다.
다음 항공편 출발 정보를 바탕으로 자연스러운 한국어로 출발 시간을 추천해주세요.

{context}

다음 내용을 포함하여 자연스럽고 친절한 한국어로 답변해주세요:
1. 추천 출발 시간 강조
2. 각 소요 시간 항목 설명 (특히 대중교통 이용 시 환승 경로를 자세히 설명)
3. 지연 예측 근거 설명:
   - 실시간 항공사 정보가 있으면: "항공사 공식 발표에 따르면 현재 XX분 지연이 예상됩니다"
   - AI 예측인 경우: "AI 모델이 과거 동일 노선/시간대 데이터를 분석한 결과 평균 XX분 지연이 예상됩니다"
4. 날씨 상황과 지연 위험도 설명 (악천후 시 주의사항 포함)
5. 추가 팁 (대중교통 이용 시 교통카드 충전, 환승 시 주의사항, 날씨 대비 등)

JSON이나 마크다운 없이 일반 텍스트로 답변해주세요."""
        
        # Ollama API 호출
        try:
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
                recommendation_text = f"""
✈️ 출발 시간 추천

{flight_info.get('flight_number', 'N/A')} 편 ({flight_info.get('airline_name', flight_info['airline_code'])})
출발 예정: {flight_info['scheduled_time'].strftime('%Y-%m-%d %H:%M')}
실제 출발: {actual_departure.strftime('%Y-%m-%d %H:%M')} (지연 {total_predicted_delay:.0f}분)

📍 추천 출발 시간: {recommended_departure.strftime('%H:%M')}

소요 시간:
- 이동: {travel_time_minutes}분 ({travel_mode_kr}){transit_route_text}
- TSA: {tsa_wait_minutes}분
- 수하물: {baggage_check_minutes}분
- 게이트: {gate_walk_minutes}분
- 총: {total_time}분

날씨: {weather['condition']} (지연 위험 {weather['delay_risk']}, +{weather_delay}분)
"""
        except Exception as e:
            print(f"   ⚠️ LLM 호출 실패: {e}")
            recommendation_text = f"추천 출발 시간: {recommended_departure.strftime('%H:%M')}"
        
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
