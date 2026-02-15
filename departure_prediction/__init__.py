"""
Departure Prediction module.

This package provides a departure-time prediction system as a Python package.

## Module structure

- `data`: data collection and preprocessing
- `models`: Transformer and LSTM models
- `utils`: utility functions

## Usage example

```python
from departure_prediction.data.data_collector import JFKDataCollector
from departure_prediction.models.transformer_model import create_model
from departure_prediction.predict import DeparturePredictionAPI

# Collect data
collector = JFKDataCollector()
data = collector.collect_security_wait_time()

# Create model
model = create_model('transformer', input_dim=8)

# Predict
api = DeparturePredictionAPI(
    model_path='checkpoints/best_model.pt',
    normalization_params_path='normalization_params.json'
)
prediction = api.predict(historical_data)
```
"""
