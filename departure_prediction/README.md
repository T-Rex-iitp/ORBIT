# AI-Enabled Flight Departure Time Predictor

AI-based flight departure recommendation system. It combines a Transformer model, real-time APIs, and Google Gemini to recommend the optimal time to leave home.

## 🎯 Key Features

1. **Multiple input methods**
   - Upload ticket image (Google Gemini Vision OCR)
   - Manual information input

2. **Real-time delay prediction**
   - Check official airline announcements via AviationStack API
   - AI prediction with FT-Transformer model (trained on 60,000+ flights)
   - Analyze weather impact via Google Weather API

3. **Operational-context delay adjustment (New)**
   - JFK 50-mile zone congestion analysis (sampling delay rates at nearby airports such as JFK/LGA/EWR)
   - Reflect delay propagation from the previous flight (previous leg)

4. **Integrated total-time calculation**
   - Google Routes API (traffic info, detailed public transit routes)
   - TSA security screening wait time (time-of-day statistics)
   - Baggage check-in time (30 minutes vs 0 minutes for carry-on only)
   - Gate walking time

4. **AI recommendation**
   - Google Gemini LLM + Vision (unified API)
   - Natural explanation with additional tips

## 🏗️ System Architecture

```
Ticket image / manual input
    ↓
Gemini Vision OCR / direct input
    ↓
Real-time flight status check (AviationStack API)
    ├─ Delay information exists -> use official airline announcement
    └─ No information -> FT-Transformer AI prediction
    ↓
Operational-context adjustment
    ├─ JFK 50-mile zone congestion
    └─ Previous-flight delay propagation
    ↓
Weather API (Google Weather) -> additional delay (+0/15/30 minutes)
    ↓
Traffic info (Google Routes API) + TSA + baggage
    ↓
Airport arrival target = actual departure - 2 hours
    ↓
Gemini LLM natural-language recommendation
```

## 📦 Quick Start

### 1. Move to project directory

```bash
cd departure_prediction
```

### 2. Install libraries

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
# Create .env file or export directly
export GEMINI_API_KEY=your_gemini_api_key
export USE_GEMINI=true
export GOOGLE_MAPS_API_KEY=your_google_maps_key
export AVIATIONSTACK_API_KEY=your_aviationstack_key
```

**Get API keys:**
- **Gemini API**: https://aistudio.google.com/app/apikey (free)
- **Google Maps**: https://console.cloud.google.com/ ($300 free credit)
- **AviationStack**: https://aviationstack.com/ (free: 100 requests/month)

### 4. Run immediately

```bash
python app_interactive.py
```

## 🔧 Detailed Configuration

### Environment variables (.env file)

```bash
# Google Maps API (Routes + Weather)
GOOGLE_MAPS_API_KEY=AIzaSy...

# AviationStack API (real-time flight information)
AVIATIONSTACK_API_KEY=6a3f93...

# Gemini API (LLM + Vision)
GEMINI_API_KEY=AIzaSy...

# Enable Gemini usage
USE_GEMINI=true
```

### Use local Ollama model (optional)

To use a local model instead of Gemini:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download models
ollama pull gpt-oss:120b  # 65GB
ollama pull llava-phi3    # 2.9GB

# Environment variables
export USE_GEMINI=false
export OLLAMA_HOST=http://127.0.0.1:11434
```

## 📦 Installation (Legacy - Ollama only)

### 1. Install Python packages

```bash
conda create -n flight python=3.10
conda activate flight
pip install -r requirements.txt
```

### 2. Install Ollama (local LLM)

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Download Korean LLM (65GB)
ollama pull gpt-oss:120b

# Download Vision model (2.9GB)
ollama pull llava-phi3
```

### 3. Download pretrained models

```bash
# The following files are required in the models/ folder:
# - ft_transformer_full.pkl (FT-Transformer model)
# - delay_predictor_full.pkl (preprocessing pipeline)
# - xgboost_predictor.pkl (optional: XGBoost model)
```

### 4. Set API keys

Create `.env` file:

```bash
# Google Maps API (Routes + Weather)
GOOGLE_MAPS_API_KEY=your_google_api_key_here

# AviationStack API (real-time flight information)
AVIATIONSTACK_API_KEY=your_aviationstack_key_here

# Ollama (local server, no API key required)
OLLAMA_URL=http://localhost:11434
```

**Get API keys:**
- Google Maps: https://console.cloud.google.com/ ($300 free credit)
- AviationStack: https://aviationstack.com/ (free tier: 100 requests/month)

## 🚀 Usage

### Run interactive app

```bash
python app_interactive.py
```

**Usage flow:**
1. Select ticket image upload or manual input
2. Enter current location (address)
3. Select travel mode (car/public transit/walk/bike)
4. Enter baggage information
5. Check AI recommendation result

### Use directly in code

```python
from hybrid_predictor import HybridDeparturePredictor
from datetime import datetime

# Load model
predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')

# Flight information
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

# Get recommendation
result = predictor.recommend_departure(
    address='450 W 42nd St, New York, NY 10036',
    flight_info=flight_info,
    travel_mode='TRANSIT',
    rui_usecase_data={
        'congestion_check': {
            'congestion_level': 'medium',
            'score': 0.48,
            'sample_size': 28,
            'extra_delay_minutes': 10
        },
        'previous_flight_check': {
            'found': True,
            'previous_delay_minutes': 22,
            'propagated_delay_minutes': 11
        }
    }
)

print(result['recommendation'])
```

> If `rui_usecase_data` is not provided, it uses the existing AviationStack-based congestion/previous-flight analysis logic.

## 📊 Model Training (Optional)

### Prepare data

```bash
# Download flight data from Kaggle
# https://www.kaggle.com/datasets/...
# data/flight_data_2024_sample.csv
```

### Train FT-Transformer

```bash
jupyter notebook train_delay_predictor.ipynb
```

### Train XGBoost (Alternative)

```bash
jupyter notebook train_xgboost.ipynb
```

**Training results:**
- FT-Transformer: MAE 26.12 min, R² 0.0117
- XGBoost: MAE 28.2 min, R² 0.017
- **Conclusion:** Flight delays are fundamentally unpredictable (weather, mechanical failures, cascading delays)
- **Solution:** Statistical baseline + real-time API combination

## 📁 Project Structure

```
departure_prediction/
├── app_interactive.py          # Main interactive app
├── hybrid_predictor.py         # Core prediction system
├── requirements.txt            # Python packages
├── .env.example               # Environment variable examples
│
├── utils/                      # Utility modules
│   ├── flight_status_checker.py  # Real-time flight status
│   ├── google_routes.py          # Google Routes API
│   ├── weather_google.py         # Google Weather API
│   ├── tsa_wait_time.py          # TSA wait time statistics
│   ├── ticket_ocr.py             # Ticket OCR (LLaVA)
│   ├── real_flight_data.py       # Flight data collection
│   └── generate_ticket_image.py  # Test ticket generation
│
├── models/                     # Trained model files
│   ├── ft_transformer_full.pkl
│   ├── delay_predictor_full.pkl
│   └── xgboost_predictor.pkl
│
├── data/                       # Data files
│   ├── flight_data_2024_sample.csv
│   ├── flights_20260205.json
│   └── test_tickets_today.json
│
├── test_tickets/              # Test ticket images
│
└── train_*.ipynb              # Model training notebooks
```

## 🔧 Configuration Options

### TSA Wait Time configuration

Adjust airport-specific wait times in `utils/tsa_wait_time.py`:

```python
TSA_WAIT_TIMES = {
    'JFK': {
        'peak': 45,      # Peak time (07:00-10:00, 16:00-19:00)
        'normal': 25,    # Normal time
        'off_peak': 15   # Quiet time
    }
}
```

### Weather delay configuration

Adjust delay risk levels in `utils/weather_google.py`:

```python
# High risk: +30 min
# Medium risk: +15 min
# Low risk: 0 min
```

## 📝 API Usage

**Based on free tier (per recommendation):**
- Google Routes API: 1 request ($5-10 per 1,000)
- Google Weather API: 1 request
- AviationStack: 1 request (100/month free)
- Ollama: free (local)

**Estimated cost:**
- Thousands of uses possible with Google Cloud $300 credit
- Then $200 free monthly credit

## 🐛 Troubleshooting

### Ollama connection error
```bash
# Start Ollama server
ollama serve

# Check models
ollama list
```

### Google API error
```bash
# Check API key
echo $GOOGLE_MAPS_API_KEY

# Check .env file permissions
chmod 600 .env
```

### Model load error
```bash
# Check PyTorch version (2.6+ required)
python -c "import torch; print(torch.__version__)"

# weights_only=False option required
```

## 🤝 Contributing

Issues and PRs are welcome.

## 📄 License

MIT License

---

**Note:** This system is for reference purposes. For actual travel, always check official airline information.
