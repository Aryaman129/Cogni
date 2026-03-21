import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.transcription import transcribe_audio

def test_transcription_pipeline():
    audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_audio.wav')

    assert os.path.exists(audio_path), "❌ test_audio.wav not found in data/ folder"

    result = transcribe_audio(audio_path)

    print("\n✅ Transcription Result:")
    print(f"   Text preview : {result['text'][:100]}...")
    print(f"   Word count   : {result['word_count']}")
    print(f"   Pauses found : {len(result['pause_events'])}")
    print(f"   Duration     : {result['duration']}s")

    if result['pause_events']:
        print(f"\n   First 3 pauses detected:")
        for p in result['pause_events'][:3]:
            print(f"   → '{p['after_word']}' ... '{p['before_word']}' ({p['duration']}s)")

    # Assertions
    assert isinstance(result['text'], str),        "❌ text is not a string"
    assert isinstance(result['words'], list),       "❌ words is not a list"
    assert isinstance(result['pause_events'], list),"❌ pause_events is not a list"
    assert result['word_count'] >= 0,               "❌ word_count is negative"

    if result['word_count'] > 0:
        first_word = result['words'][0]
        assert 'word'  in first_word, "❌ word missing from word entry"
        assert 'start' in first_word, "❌ start time missing from word entry"
        assert 'end'   in first_word, "❌ end time missing from word entry"

    print("\n✅ All assertions passed!")

if __name__ == '__main__':
    test_transcription_pipeline()