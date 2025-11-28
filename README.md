# AI-Enabled-IFTA: Team T-Rex

This project was part of the 2025 IITP Project in CMU.

## Overview

AI-Enabled-IFTA is an ADS-B (Automatic Dependent Surveillance-Broadcast) display application with integrated Whisper-based speech recognition capabilities. The application provides real-time aircraft tracking visualization and voice-controlled interaction using local Whisper speech-to-text transcription.

## Features

### ADS-B Display
- Real-time aircraft tracking and visualization
- Interactive map display with zoom controls
- Aircraft database integration
- Traffic filtering and area management

### Whisper Speech Recognition
- **Local speech-to-text transcription** using OpenAI Whisper (via faster-whisper)
- **Automatic silence detection** - recording stops automatically after 2 seconds of silence
- **Real-time transcription** - results appear directly in the Memo panel
- **No cloud dependency** - all processing happens locally
- **Multi-language support** - configurable language detection (default: English)