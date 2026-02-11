# AI-Enabled-IFTA: Team T-Rex

This project was part of the 2025 IITP Project in CMU.

## Overview

..

## Features

### ADS-B Display
- Real-time aircraft tracking and visualization
- Interactive map display with zoom controls
- Aircraft database integration
- Traffic filtering and area management

### SBS / Raw connector script (Python)
- A ready-to-run script is available at `ADS-B-Display/python/sbs_raw_connector.py`.
- It connects to both feeds concurrently using the same defaults as the C++ GUI:
  - Raw feed: `30002`
  - SBS feed: `5002`
- Example:

```bash
python ADS-B-Display/python/sbs_raw_connector.py \
  --host 127.0.0.1 \
  --raw-output ADS-B-Display/Recorded/raw_from_python.log \
  --sbs-output ADS-B-Display/Recorded/sbs_from_python.log
```

### Whisper Speech Recognition
- **Local speech-to-text transcription** using OpenAI Whisper (via faster-whisper)
- **Automatic silence detection** - recording stops automatically after 2 seconds of silence
- **Real-time transcription** - results appear directly in the Memo panel
- **No cloud dependency** - all processing happens locally
- **Multi-language support** - configurable language detection (default: English)
