# Quick Start Guide

## 1분 설치

```bash
# 1. 저장소 클론
git clone https://github.com/T-Rex-iitp/AI-Enabled-IFTA.git
cd AI-Enabled-IFTA/departure_prediction

# 2. Python 환경 설정
conda create -n flight python=3.10
conda activate flight
pip install -r requirements.txt

# 3. Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gpt-oss:120b
ollama pull llava-phi3

# 4. API 키 설정
cp .env.example .env
# .env 파일에 API 키 입력

# 5. 실행!
python app_interactive.py
```

## API 키 발급

1. **Google Maps** ($300 무료): https://console.cloud.google.com/
2. **AviationStack** (100회/월 무료): https://aviationstack.com/

## 모델 다운로드

학습된 모델이 필요합니다 (별도 제공):
- `models/ft_transformer_full.pkl`
- `models/delay_predictor_full.pkl`

또는 직접 학습:
```bash
jupyter notebook train_delay_predictor.ipynb
```

## 문제 해결

**Ollama 연결 오류?**
```bash
ollama serve
```

**모델 파일 없음?**
- 노트북으로 직접 학습하거나
- 팀에게 요청

**API 키 오류?**
- `.env` 파일 확인
- `echo $GOOGLE_MAPS_API_KEY`로 환경변수 확인

---

더 자세한 내용은 README.md를 참고하세요!
