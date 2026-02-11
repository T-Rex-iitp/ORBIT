"""
Departure Prediction 모듈

출발 시간 예측 시스템을 Python 패키지로 사용할 수 있도록 합니다.

## 모듈 구성

- `data`: 데이터 수집 및 전처리
- `models`: Transformer 및 LSTM 모델
- `utils`: 유틸리티 함수

## 사용 예시

```python
from departure_prediction.data.data_collector import JFKDataCollector
from departure_prediction.models.transformer_model import create_model
from departure_prediction.predict import DeparturePredictionAPI

# 데이터 수집
collector = JFKDataCollector()
data = collector.collect_security_wait_time()

# 모델 생성
model = create_model('transformer', input_dim=8)

# 예측
api = DeparturePredictionAPI(
    model_path='checkpoints/best_model.pt',
    normalization_params_path='normalization_params.json'
)
prediction = api.predict(historical_data)
```
"""
