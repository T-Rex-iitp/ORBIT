# Quick Start Guide

## 1-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/T-Rex-iitp/AI-Enabled-IFTA.git
cd AI-Enabled-IFTA/departure_prediction

# 2. Set up Python environment
conda create -n flight python=3.10
conda activate flight
pip install -r requirements.txt

# 3. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gpt-oss:120b
ollama pull llava-phi3

# 4. Set API keys
cp .env.example .env
# Enter your API keys in the .env file

# 5. Run!
python app_interactive.py
```

## Get API Keys

1. **Google Maps** ($300 free): https://console.cloud.google.com/
2. **AviationStack** (100 requests/month free): https://aviationstack.com/

## Download Models

Pretrained models are required (provided separately):
- `models/ft_transformer_full.pkl`
- `models/delay_predictor_full.pkl`

Or train them yourself:
```bash
jupyter notebook train_delay_predictor.ipynb
```

## Troubleshooting

**Ollama connection error?**
```bash
ollama serve
```

**Model files missing?**
- Train directly with the notebook
- Ask your team

**API key error?**
- Check the `.env` file
- Check environment variables with `echo $GOOGLE_MAPS_API_KEY`

---

For more details, see README.md!
