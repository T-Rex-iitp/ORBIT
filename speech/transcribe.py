#!/usr/bin/env python3
"""
Whisper-based speech transcription script for ADS-B Display GUI
Uses faster-whisper for fast local inference
"""

import sys
import os
import argparse
import subprocess
import tempfile
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: faster-whisper not installed. Run: pip install faster-whisper", file=sys.stderr)
    sys.exit(1)

# Default model configuration
DEFAULT_MODEL = "base"
DEFAULT_DEVICE = "cpu"  # "cpu" or "cuda"
DEFAULT_COMPUTE_TYPE = "int8"  # "int8", "int8_float16", "float16", "float32"
DEFAULT_LANGUAGE = "en"

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def resample_audio(input_path, output_path, target_sample_rate=16000, target_channels=1):
    """
    Resample audio to 16kHz mono using ffmpeg
    Returns True if successful, False otherwise
    """
    if not check_ffmpeg():
        # If ffmpeg is not available, try to use the file as-is
        # (faster-whisper can handle some formats directly)
        return False
    
    try:
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ar", str(target_sample_rate),
            "-ac", str(target_channels),
            "-y",  # Overwrite output file
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def transcribe_audio(audio_path, model_name=DEFAULT_MODEL, device=DEFAULT_DEVICE, 
                     compute_type=DEFAULT_COMPUTE_TYPE, language=DEFAULT_LANGUAGE):
    """
    Transcribe audio file using faster-whisper
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: "cpu" or "cuda"
        compute_type: "int8", "int8_float16", "float16", "float32"
        language: Language code (e.g., "ko", "en") or None for auto-detect
    
    Returns:
        Transcribed text string
    """
    try:
        # Debug: Print Whisper initialization
        print(f"WHISPER: Initializing model '{model_name}' on {device} with {compute_type}", file=sys.stderr)
        sys.stderr.flush()
        
        # Initialize model (this will download on first run)
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        
        print(f"WHISPER: Model loaded successfully", file=sys.stderr)
        print(f"WHISPER: Transcribing audio file: {audio_path}", file=sys.stderr)
        sys.stderr.flush()
        
        # Transcribe
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,  # Voice Activity Detection
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        print(f"WHISPER: Transcription started, detected language: {info.language}", file=sys.stderr)
        sys.stderr.flush()
        
        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
            print(f"WHISPER: Segment [{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text.strip()}", file=sys.stderr)
            sys.stderr.flush()
        
        result = " ".join(text_parts).strip()
        print(f"WHISPER: Final transcription: '{result}'", file=sys.stderr)
        sys.stderr.flush()
        
        return result
    
    except Exception as e:
        # Send errors to stderr
        print(f"ERROR: Transcription failed: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file using faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (WAV, MP3, etc.)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Whisper model size (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help=f"Device to use (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=DEFAULT_COMPUTE_TYPE,
        choices=["int8", "int8_float16", "float16", "float32"],
        help=f"Compute type (default: {DEFAULT_COMPUTE_TYPE})"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Language code (ko, en, etc.) or 'auto' for auto-detect (default: ko)"
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Force resample to 16kHz mono using ffmpeg"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
    
    # Handle language
    language = None if args.language.lower() == "auto" else args.language
    
    # Resample if requested or if file format might be problematic
    processed_audio_path = audio_path
    temp_file = None
    
    if args.resample or not audio_path.suffix.lower() == ".wav":
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        
        if resample_audio(audio_path, temp_file.name):
            processed_audio_path = Path(temp_file.name)
        else:
            # If resampling failed, try original file
            processed_audio_path = audio_path
    
    try:
        # Transcribe
        text = transcribe_audio(
            processed_audio_path,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=language
        )
        
        # Output only the transcribed text (no extra messages)
        # Send errors to stderr, only transcription to stdout
        if text:
            print(text, file=sys.stdout)
            sys.stdout.flush()  # Ensure output is sent immediately
        else:
            print("", end="", file=sys.stdout)  # Empty output for silence
            sys.stdout.flush()
        
    except Exception as e:
        # Send all errors to stderr, not stdout
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
    finally:
        # Clean up temp file if created
        if temp_file and Path(temp_file.name).exists():
            try:
                os.unlink(temp_file.name)
            except:
                pass

if __name__ == "__main__":
    main()

