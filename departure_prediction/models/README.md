# Trained Model Files

This folder contains pretrained AI models.

## Included Files

### FT-Transformer Model (Recommended)
- `ft_transformer_full.pkl` (2.2MB) - Includes preprocessing pipeline
- `delay_predictor_full.pkl` (2.2MB) - Trained FT-Transformer
- `best_ft_transformer.pth` (6.6MB) - PyTorch checkpoint

**Performance:**
- MAE: 26.12 minutes
- RMSE: 52.25 minutes
- R²: 0.0117

### XGBoost Model (Alternative)
- `xgboost_predictor.pkl` (5.2MB)

**Performance:**
- MAE: 28.2 minutes
- R²: 0.017

### Legacy Transformer
- `best_transformer.pth` (1.3MB) - Previous version

## Training Data

- **Dataset:** Kaggle Flight Delay Dataset
- **Sample count:** 60,000+ domestic flights
- **Period:** 2024
- **Airports:** JFK, LAX, ORD, ATL, DFW, etc.

## How to Retrain

To train the models yourself:

```bash
# Train FT-Transformer
jupyter notebook train_delay_predictor.ipynb

# Train XGBoost
jupyter notebook train_xgboost.ipynb
```

## Using the Model

```python
from hybrid_predictor import HybridDeparturePredictor

# Automatically load model
predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')
```

## Important Notes

**Flight delay is fundamentally difficult to predict:**
- R² ≈ 0.01 (1% explanatory power)
- Unpredictable factors such as weather, mechanical failures, and cascading delays

**Approach:**
- Provide a statistical baseline
- Complement with real-time APIs (AviationStack, Google Weather)
- Use LLM for context-specific recommendations

---

See `train_delay_predictor.ipynb` for the training process
