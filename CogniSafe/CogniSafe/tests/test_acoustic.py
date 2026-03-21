import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.transcription import transcribe_audio
from pipeline.acoustic import extract_acoustic_features

def test_acoustic_with_transcript():
    audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_audio.wav')

    # First get transcript
    print("Running Whisper transcription first...")
    transcript = transcribe_audio(audio_path)

    # Now pass transcript into acoustic
    result = extract_acoustic_features(audio_path, transcript_data=transcript)

    print("\n✅ Acoustic Features (with transcript):")
    for key, val in result.items():
        print(f"   {key}: {val}")

    assert result['speech_rate'] > 0,       "❌ speech_rate still 0 — transcript not passed correctly"
    assert result['articulation_rate'] > 0, "❌ articulation_rate still 0"

    print("\n✅ All assertions passed!")

if __name__ == '__main__':
    test_acoustic_with_transcript()