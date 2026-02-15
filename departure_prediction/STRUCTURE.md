# Project Structure

```
departure_prediction/
│
├── 📄 README.md                    # Project description and usage
├── 📄 requirements.txt             # Python package dependencies
├── 📄 .env.example                 # Environment variable setup example
├── 📄 .gitignore                   # Git ignore file list
│
├── 🚀 app_interactive.py           # Main interactive app
├── 🧠 hybrid_predictor.py          # Core prediction system
│
├── 📊 train_delay_predictor.ipynb  # FT-Transformer training notebook
├── 📊 train_xgboost.ipynb          # XGBoost training notebook
├── 📊 flight_data_preprocessing.ipynb  # Data preprocessing
│
├── 📂 utils/                       # Utility modules
│   ├── flight_status_checker.py   # Real-time flight status (AviationStack API)
│   ├── google_routes.py           # Google Routes API client
│   ├── weather_google.py          # Google Weather API client
│   ├── tsa_wait_time.py           # TSA wait time statistics
│   ├── ticket_ocr.py              # Ticket OCR (Ollama LLaVA)
│   ├── real_flight_data.py        # Flight data collector
│   └── generate_ticket_image.py   # Test ticket image generator
│
├── 📂 models/                      # Trained models (not included in Git)
│   ├── ft_transformer_full.pkl    # FT-Transformer model
│   ├── delay_predictor_full.pkl   # Preprocessing pipeline
│   └── xgboost_predictor.pkl      # XGBoost model (optional)
│
├── 📂 data/                        # Data files (mostly not included in Git)
│   ├── flight_data_2024_sample.csv  # Kaggle flight data
│   ├── flights_20260205.json      # Crawled real-time flights
│   └── test_tickets_today.json    # Test flight ticket info
│
└── 📂 test_tickets/                # Test ticket images (not included in Git)
    ├── ticket_1_QR2867.png
    ├── ticket_2_IB4967.png
    └── ...
```

## Core Components

### 1. app_interactive.py
- User interface
- Ticket image upload / manual input
- Input location, travel mode, and baggage info
- Output final recommendation

### 2. hybrid_predictor.py
- Load FT-Transformer model
- Integrate real-time APIs
- Predict delay time
- Generate LLM recommendation

### 3. utils/ module
Composed of independent modules by API and feature.

## Data Flow

1. **Input** -> VLM OCR or manual input
2. **Real-time check** -> AviationStack API
3. **AI prediction** -> FT-Transformer (when no real-time info)
4. **Weather** -> Google Weather API
5. **Traffic** -> Google Routes API
6. **Calculation** -> TSA + baggage + gate
7. **Output** -> Ollama LLM (Korean)

## Model Files (Separate Download Required)

Not included in Git (size constraints):
- `ft_transformer_full.pkl` (about 50MB)
- `delay_predictor_full.pkl` (about 10MB)
- Ollama models: gpt-oss:120b (65GB), llava-phi3 (2.9GB)

## Data Files

- `flight_data_2024_sample.csv`: Download from Kaggle
- Others are generated at runtime
