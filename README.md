# ORBIT: Optimal Recommendation Framework for Boarding with Interpretable Timelines

This repository contains the implementation of **ORBIT**, a multimodal airport departure assistant.

ORBIT is designed to answer a practical question: **"When should I leave for the airport?"**  
Instead of returning a single opaque timestamp, ORBIT combines predictive models and real-time operational signals, then provides an interpretable, component-level timeline.

This project was part of the 2025 IITP Project in CMU.

## What ORBIT Does

- Generates personalized **leave-by recommendations**
- Integrates **flight delay prediction** with real-time context
- Supports multimodal input:
  - text query
  - voice query (Whisper-based STT)
  - boarding pass image OCR (Gemini)
- Produces interpretable output:
  - visual timeline
  - map route view
  - detailed time-budget breakdown

## Core Design Principles

- **Integrated delay intelligence**: combines Transformer-based delay prediction with external operational factors (traffic, weather, airport process, congestion, previous-leg propagation).
- **Personalized planning**: uses origin, transport mode, baggage profile, TSA PreCheck, terminal/gate context.
- **Interpretable decision support**: decomposes the final recommendation into explicit time components so users can inspect the dominant factors.
- **Operational robustness**: uses caching and fallback strategies to continue producing outputs when API calls fail.

## Pipeline (High-Level)

1. User input normalization (text/voice/ticket OCR)
2. Data collection and preprocessing (Google Routes/Weather, TSA, flight status, ADS-B/FR24 signals)
3. Transformer-based delay prediction
4. LLM-based final integration and natural-language recommendation

## Run

```bash
streamlit run web/streamlit_app.py --serverserver.port <server-port>
```

## Required Environment Variables

For accurate execution, set the following:

```bash
export FR24_API_TOKEN="<your_fr24_api_token>"
export GEMINI_API_KEY="<your_gemini_api_key>"
export GOOGLE_MAPS_API_KEY="<your_google_maps_api_key>"

export USE_GEMINI=true
```

## Key Files

- `web/streamlit_app.py`: main Streamlit dashboard
- `departure_prediction/hybrid_predictor.py`: hybrid prediction and recommendation core
