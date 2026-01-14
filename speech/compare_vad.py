#!/usr/bin/env python3
"""
Compare faster-whisper performance with and without VAD filter
"""

from faster_whisper import WhisperModel
import time
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def transcribe(model, path, use_vad: bool):
    """
    Transcribe audio with or without VAD filter
    
    Args:
        model: WhisperModel instance
        path: Path to audio file
        use_vad: Whether to use VAD filter
    
    Returns:
        tuple: (text, duration_seconds, info)
    """
    t0 = time.time()
    
    segments, info = model.transcribe(
        path,
        language="en",
        beam_size=5,
        vad_filter=use_vad,                 # <-- key switch
        vad_parameters=dict(
            min_silence_duration_ms=500,    # tune if needed
            speech_pad_ms=200               # helps avoid clipping edges
        ) if use_vad else None
    )
    
    # Collect segments with timing info
    segment_list = []
    for seg in segments:
        segment_list.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text.strip()
        })
    
    text = " ".join(s['text'] for s in segment_list)
    duration = time.time() - t0
    
    return text, duration, info, segment_list


def main():
    audio_file = "temp_recording.wav"
    
    print("="*70)
    print("Faster-Whisper VAD Comparison")
    print("="*70)
    print(f"Audio file: {audio_file}")
    print(f"Model: base (CPU, int8)")
    print("="*70)
    
    # Initialize model once
    print("\nInitializing Whisper model...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Model loaded successfully!\n")
    
    # Test both configurations
    for use_vad in [False, True]:
        print("-"*70)
        print(f"Testing with VAD = {use_vad}")
        print("-"*70)
        
        try:
            text, sec, info, segments = transcribe(model, audio_file, use_vad)
            
            print(f"\n✓ Transcription completed")
            print(f"  Language detected: {info.language}")
            print(f"  Probability: {info.language_probability:.2%}")
            print(f"  Duration: {sec:.2f} seconds")
            print(f"  Number of segments: {len(segments)}")
            
            print(f"\n📝 Transcription:")
            print(f"  \"{text}\"")
            
            if segments:
                print(f"\n⏱️  Segment details:")
                for i, seg in enumerate(segments, 1):
                    print(f"    [{seg['start']:.2f}s -> {seg['end']:.2f}s]: {seg['text']}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()

