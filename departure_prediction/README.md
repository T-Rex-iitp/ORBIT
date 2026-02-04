# AI-Enabled Flight Departure Time Predictor

AI 기반 항공편 출발 시간 추천 시스템. Transformer 모델, 실시간 API, LLM을 결합하여 최적의 집 출발 시간을 추천합니다.

## 🎯 주요 기능

1. **다중 입력 방식**
   - 티켓 이미지 업로드 (Ollama LLaVA OCR)
   - 수동 정보 입력

2. **실시간 지연 예측**
   - AviationStack API로 항공사 공식 발표 확인
   - FT-Transformer 모델로 AI 예측 (60,000+ 항공편 학습)
   - Google Weather API로 날씨 영향 분석

3. **통합 소요시간 계산**
   - Google Routes API (교통 정보, 대중교통 상세 경로)
   - TSA 보안 검색 대기시간 (시간대별 통계)
   - 수하물 체크인 시간 (30분 vs 기내반입 0분)
   - 게이트 이동 시간

4. **한국어 LLM 추천**
   - Ollama gpt-oss:120b (65GB 한국어 LLM)
   - 자연스러운 설명과 추가 팁 제공

## 🏗️ 시스템 아키텍처

```
티켓 이미지/수동 입력
    ↓
VLM (LLaVA-Phi3) / 직접 입력
    ↓
실시간 항공편 상태 확인 (AviationStack API)
    ├─ 지연 정보 있음 → 항공사 공식 발표 사용
    └─ 정보 없음 → FT-Transformer AI 예측
    ↓
날씨 API (Google Weather) → 추가 지연 (+0/15/30분)
    ↓
교통 정보 (Google Routes API) + TSA + 수하물
    ↓
공항 도착 목표 = 실제 출발 - 2시간
    ↓
LLM 한국어 추천 (gpt-oss:120b)
```

## 📦 설치

### 1. Python 패키지 설치

```bash
conda create -n flight python=3.10
conda activate flight
pip install -r requirements.txt
```

### 2. Ollama 설치 (로컬 LLM)

```bash
# Ollama 설치 (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# 한국어 LLM 다운로드 (65GB)
ollama pull gpt-oss:120b

# Vision 모델 다운로드 (2.9GB)
ollama pull llava-phi3
```

### 3. 학습된 모델 다운로드

```bash
# models/ 폴더에 다음 파일들이 필요합니다:
# - ft_transformer_full.pkl (FT-Transformer 모델)
# - delay_predictor_full.pkl (전처리 파이프라인)
# - xgboost_predictor.pkl (선택사항: XGBoost 모델)
```

### 4. API 키 설정

`.env` 파일 생성:

```bash
# Google Maps API (Routes + Weather)
GOOGLE_MAPS_API_KEY=your_google_api_key_here

# AviationStack API (실시간 항공편 정보)
AVIATIONSTACK_API_KEY=your_aviationstack_key_here

# Ollama (로컬 서버, API 키 불필요)
OLLAMA_URL=http://localhost:11434
```

**API 키 발급:**
- Google Maps: https://console.cloud.google.com/ ($300 무료 크레딧)
- AviationStack: https://aviationstack.com/ (무료 티어: 100 requests/month)

## 🚀 사용법

### 인터랙티브 앱 실행

```bash
python app_interactive.py
```

**사용 흐름:**
1. 티켓 이미지 업로드 또는 수동 입력 선택
2. 현재 위치 입력 (주소)
3. 이동 수단 선택 (자동차/대중교통/도보/자전거)
4. 수하물 정보 입력
5. AI 추천 결과 확인!

### 직접 코드 사용

```python
from hybrid_predictor import HybridDeparturePredictor
from datetime import datetime

# 모델 로드
predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')

# 항공편 정보
flight_info = {
    'airline_code': 'B6',
    'airline_name': 'JetBlue Airways',
    'flight_number': 'B6623',
    'origin': 'JFK',
    'dest': 'LAX',
    'scheduled_time': datetime(2026, 2, 5, 18, 30),
    'has_checked_baggage': True,
    'has_tsa_precheck': False
}

# 추천 받기
result = predictor.recommend_departure(
    address='450 W 42nd St, New York, NY 10036',
    flight_info=flight_info,
    travel_mode='TRANSIT'
)

print(result['recommendation'])
```

## 📊 모델 학습 (선택사항)

### 데이터 준비

```bash
# Kaggle에서 항공편 데이터 다운로드
# https://www.kaggle.com/datasets/...
# data/flight_data_2024_sample.csv
```

### FT-Transformer 학습

```bash
jupyter notebook train_delay_predictor.ipynb
```

### XGBoost 학습 (대안)

```bash
jupyter notebook train_xgboost.ipynb
```

**학습 결과:**
- FT-Transformer: MAE 26.12분, R² 0.0117
- XGBoost: MAE 28.2분, R² 0.017
- **결론:** 항공편 지연은 근본적으로 예측 불가 (날씨, 기계 결함, 연쇄 지연)
- **해결:** 통계적 기준선 + 실시간 API 조합

## 📁 프로젝트 구조

```
departure_prediction/
├── app_interactive.py          # 메인 인터랙티브 앱
├── hybrid_predictor.py         # 핵심 예측 시스템
├── requirements.txt            # Python 패키지
├── .env.example               # 환경변수 예시
│
├── utils/                      # 유틸리티 모듈
│   ├── flight_status_checker.py  # 실시간 항공편 상태
│   ├── google_routes.py          # Google Routes API
│   ├── weather_google.py         # Google Weather API
│   ├── tsa_wait_time.py          # TSA 대기시간 통계
│   ├── ticket_ocr.py             # 티켓 OCR (LLaVA)
│   ├── real_flight_data.py       # 항공편 데이터 수집
│   └── generate_ticket_image.py  # 테스트 티켓 생성
│
├── models/                     # 학습된 모델 파일
│   ├── ft_transformer_full.pkl
│   ├── delay_predictor_full.pkl
│   └── xgboost_predictor.pkl
│
├── data/                       # 데이터 파일
│   ├── flight_data_2024_sample.csv
│   ├── flights_20260205.json
│   └── test_tickets_today.json
│
├── test_tickets/              # 테스트용 티켓 이미지
│
└── train_*.ipynb              # 모델 학습 노트북
```

## 🔧 설정 옵션

### TSA Wait Time 설정

`utils/tsa_wait_time.py`에서 공항별 대기시간 조정:

```python
TSA_WAIT_TIMES = {
    'JFK': {
        'peak': 45,      # 피크 시간 (07:00-10:00, 16:00-19:00)
        'normal': 25,    # 보통 시간
        'off_peak': 15   # 한가한 시간
    }
}
```

### 날씨 지연 설정

`utils/weather_google.py`에서 지연 위험도 조정:

```python
# High risk: +30분
# Medium risk: +15분
# Low risk: 0분
```

## 📝 API 사용량

**무료 티어 기준 (1회 추천당):**
- Google Routes API: 1 request ($5-10 per 1,000)
- Google Weather API: 1 request
- AviationStack: 1 request (100/month 무료)
- Ollama: 무료 (로컬)

**예상 비용:**
- Google Cloud $300 크레딧으로 수천 회 사용 가능
- 이후 월 $200 무료 크레딧

## 🐛 트러블슈팅

### Ollama 연결 오류
```bash
# Ollama 서버 시작
ollama serve

# 모델 확인
ollama list
```

### Google API 오류
```bash
# API 키 확인
echo $GOOGLE_MAPS_API_KEY

# .env 파일 권한 확인
chmod 600 .env
```

### 모델 로드 오류
```bash
# PyTorch 버전 확인 (2.6+ 필요)
python -c "import torch; print(torch.__version__)"

# weights_only=False 옵션 필요
```

## 🤝 기여

이슈 및 PR 환영합니다!

## 📄 라이선스

MIT License

## 👥 개발팀

IITP AI Project Team

---

**Note:** 이 시스템은 참고용입니다. 실제 항공편 이용 시 항공사 공식 정보를 확인하세요.
