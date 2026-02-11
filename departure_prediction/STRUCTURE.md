# Project Structure

```
departure_prediction/
│
├── �� README.md                    # 프로젝트 설명 및 사용법
├── 📄 requirements.txt             # Python 패키지 의존성
├── 📄 .env.example                 # 환경변수 설정 예시
├── 📄 .gitignore                   # Git 제외 파일 목록
│
├── 🚀 app_interactive.py           # 메인 인터랙티브 앱
├── 🧠 hybrid_predictor.py          # 핵심 예측 시스템
│
├── 📊 train_delay_predictor.ipynb  # FT-Transformer 학습 노트북
├── 📊 train_xgboost.ipynb          # XGBoost 학습 노트북
├── 📊 flight_data_preprocessing.ipynb  # 데이터 전처리
│
├── 📂 utils/                       # 유틸리티 모듈
│   ├── flight_status_checker.py   # 실시간 항공편 상태 (AviationStack API)
│   ├── google_routes.py           # Google Routes API 클라이언트
│   ├── weather_google.py          # Google Weather API 클라이언트
│   ├── tsa_wait_time.py           # TSA 대기시간 통계
│   ├── ticket_ocr.py              # 티켓 OCR (Ollama LLaVA)
│   ├── real_flight_data.py        # 항공편 데이터 수집기
│   └── generate_ticket_image.py   # 테스트 티켓 이미지 생성
│
├── 📂 models/                      # 학습된 모델 (Git에 포함 안됨)
│   ├── ft_transformer_full.pkl    # FT-Transformer 모델
│   ├── delay_predictor_full.pkl   # 전처리 파이프라인
│   └── xgboost_predictor.pkl      # XGBoost 모델 (선택사항)
│
├── 📂 data/                        # 데이터 파일 (대부분 Git에 포함 안됨)
│   ├── flight_data_2024_sample.csv  # Kaggle 항공편 데이터
│   ├── flights_20260205.json      # 크롤링된 실시간 항공편
│   └── test_tickets_today.json    # 테스트용 항공편 정보
│
└── 📂 test_tickets/                # 테스트용 티켓 이미지 (Git에 포함 안됨)
    ├── ticket_1_QR2867.png
    ├── ticket_2_IB4967.png
    └── ...
```

## 핵심 컴포넌트

### 1. app_interactive.py
- 사용자 인터페이스
- 티켓 이미지 업로드 / 수동 입력
- 위치, 교통수단, 수하물 정보 입력
- 최종 추천 결과 출력

### 2. hybrid_predictor.py
- FT-Transformer 모델 로드
- 실시간 API 통합
- 지연 시간 예측
- LLM 추천 생성

### 3. utils/ 모듈
각 API 및 기능별 독립적인 모듈로 구성

## 데이터 플로우

1. **입력** → VLM OCR 또는 수동 입력
2. **실시간 확인** → AviationStack API
3. **AI 예측** → FT-Transformer (실시간 정보 없을 때)
4. **날씨** → Google Weather API
5. **교통** → Google Routes API
6. **계산** → TSA + 수하물 + 게이트
7. **출력** → Ollama LLM (한국어)

## 모델 파일 (별도 다운로드 필요)

Git에는 포함되지 않음 (용량 문제):
- `ft_transformer_full.pkl` (약 50MB)
- `delay_predictor_full.pkl` (약 10MB)
- Ollama 모델: gpt-oss:120b (65GB), llava-phi3 (2.9GB)

## 데이터 파일

- `flight_data_2024_sample.csv`: Kaggle에서 다운로드
- 나머지는 런타임에 생성됨
