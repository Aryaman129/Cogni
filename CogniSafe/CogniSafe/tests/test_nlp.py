import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.transcription import transcribe_audio
from pipeline.nlp import analyze_text

def test_nlp_pipeline():
    audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_audio.wav')

    assert os.path.exists(audio_path), "❌ test_audio.wav not found in data/ folder"

    # Get transcript first
    print("Running Whisper transcription...")
    transcript = transcribe_audio(audio_path)

    print(f"\nTranscript preview: {transcript['text'][:150]}...")

    # Run NLP analysis
    print("\nRunning NLP analysis...")
    result = analyze_text(transcript)

    print("\n✅ NLP Biomarkers:")
    print(f"   lexical_diversity    : {result['lexical_diversity']}")
    print(f"   semantic_coherence   : {result['semantic_coherence']}")
    print(f"   idea_density         : {result['idea_density']}")
    print(f"   syntactic_complexity : {result['syntactic_complexity']}")

    # Assertions
    expected_keys = [
        'lexical_diversity', 'semantic_coherence',
        'idea_density', 'syntactic_complexity'
    ]
    for key in expected_keys:
        assert key in result,               f"❌ Missing: {key}"
        assert isinstance(result[key], float), f"❌ {key} is not a float"
        assert result[key] >= 0.0,          f"❌ {key} is negative"

    # Sanity range checks
    assert 0.0 <= result['semantic_coherence'] <= 1.0, \
        "❌ semantic_coherence should be between 0 and 1"
    assert 0.0 <= result['idea_density'] <= 1.0, \
        "❌ idea_density should be between 0 and 1"

    print("\n✅ All assertions passed!")

if __name__ == '__main__':
    test_nlp_pipeline()