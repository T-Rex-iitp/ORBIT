# 학습된 모델 파일

이 폴더에는 사전 학습된 AI 모델들이 포함되어 있습니다.

## 포함된 파일

### FT-Transformer 모델 (추천)
- `ft_transformer_full.pkl` (2.2MB) - 전처리 파이프라인 포함
- `delay_predictor_full.pkl` (2.2MB) - 학습된 FT-Transformer
- `best_ft_transformer.pth` (6.6MB) - PyTorch 체크포인트

**성능:**
- MAE: 26.12분
- RMSE: 52.25분
- R²: 0.0117

### XGBoost 모델 (대안)
- `xgboost_predictor.pkl` (5.2MB)

**성능:**
- MAE: 28.2분
- R²: 0.017

### Legacy Transformer
- `best_transformer.pth` (1.3MB) - 이전 버전

## 학습 데이터

- **데이터셋:** Kaggle Flight Delay Dataset
- **샘플 수:** 60,000+ 국내선 항공편
- **기간:** 2024년
- **공항:** JFK, LAX, ORD, ATL, DFW 등

## 재학습 방법

모델을 직접 학습하려면:

```bash
# FT-Transformer 학습
jupyter notebook train_delay_predictor.ipynb

# XGBoost 학습
jupyter notebook train_xgboost.ipynb
```

## 모델 사용

```python
from hybrid_predictor import HybridDeparturePredictor

# 자동으로 모델 로드
predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')
```

## 중요 참고사항

**항공편 지연은 근본적으로 예측이 어렵습니다:**
- R² ≈ 0.01 (설명력 1%)
- 날씨, 기계 결함, 연쇄 지연 등 예측 불가능한 요인들

**해결 방법:**
- 통계적 기준선 제공
- 실시간 API로 보완 (AviationStack, Google Weather)
- LLM으로 상황별 추천

---

학습 과정은 `train_delay_predictor.ipynb` 참고
