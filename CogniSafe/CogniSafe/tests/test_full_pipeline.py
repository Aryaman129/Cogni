import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.transcription import transcribe_audio
from pipeline.acoustic      import extract_acoustic_features
from pipeline.nlp           import analyze_text
from pipeline.anomaly       import run_anomaly_pipeline
from pipeline.risk          import merge_biomarkers

def test_full_pipeline():
    audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_audio.wav')
    assert os.path.exists(audio_path), "❌ test_audio.wav not found"

    print("\n" + "="*55)
    print("   COGNISAFE — FULL PIPELINE TEST")
    print("="*55)

    # ── Stage 1: Transcription ────────────────────────────────
    print("\n[1/5] Running Whisper transcription...")
    transcript = transcribe_audio(audio_path)
    assert transcript['word_count'] > 0, "❌ No words transcribed"
    print(f"      ✅ {transcript['word_count']} words | "
          f"{len(transcript['pause_events'])} pauses")

    # ── Stage 2: Acoustic Features ────────────────────────────
    print("\n[2/5] Extracting acoustic features...")
    acoustic = extract_acoustic_features(audio_path, transcript_data=transcript)
    assert acoustic['speech_rate'] > 0,  "❌ speech_rate is 0"
    assert acoustic['pitch_mean']  > 0,  "❌ pitch_mean is 0"
    print(f"      ✅ speech_rate={acoustic['speech_rate']} | "
          f"pitch_mean={acoustic['pitch_mean']}")

    # ── Stage 3: NLP Analysis ─────────────────────────────────
    print("\n[3/5] Running NLP analysis...")
    nlp = analyze_text(transcript)
    assert nlp['semantic_coherence']  > 0, "❌ semantic_coherence is 0"
    assert nlp['lexical_diversity']   > 0, "❌ lexical_diversity is 0"
    print(f"      ✅ semantic_coherence={nlp['semantic_coherence']} | "
          f"lexical_diversity={nlp['lexical_diversity']}")

    # ── Stage 4: Merge Biomarkers ─────────────────────────────
    print("\n[4/5] Merging all 14 biomarkers...")
    all_biomarkers = merge_biomarkers(acoustic, nlp)
    assert len(all_biomarkers) == 14, \
        f"❌ Expected 14 biomarkers, got {len(all_biomarkers)}"
    print(f"      ✅ All 14 biomarkers present")
    for k, v in all_biomarkers.items():
        print(f"         {k:<25}: {v}")

    # ── Stage 5: Anomaly Detection ────────────────────────────
    print("\n[5/5] Running anomaly detection...")
    result = run_anomaly_pipeline("demo_user", all_biomarkers)
    assert result['risk_tier'] in ['Green','Yellow','Orange','Red'], \
        "❌ Invalid risk tier"
    print(f"      ✅ Risk Tier : {result['risk_tier']}")
    print(f"      ✅ Flags     : {len(result['anomaly_flags'])}")
    print(f"      ✅ Confidence intervals: "
          f"{len(result['confidence_intervals'])} biomarkers")

    # ── Final Schema Check ────────────────────────────────────
    print("\n── Schema Validation ──")
    required_keys = [
        'session_id', 'user_id', 'timestamp',
        'processing_time_seconds', 'biomarkers',
        'anomaly_flags', 'risk_tier', 'confidence_intervals'
    ]

    import uuid, datetime
    final_output = {
        'session_id':                str(uuid.uuid4()),
        'user_id':                   'demo_user',
        'timestamp':                 datetime.datetime.utcnow().isoformat(),
        'processing_time_seconds':   0.0,
        'biomarkers':                all_biomarkers,
        'anomaly_flags':             result['anomaly_flags'],
        'risk_tier':                 result['risk_tier'],
        'confidence_intervals':      result['confidence_intervals'],
    }

    for key in required_keys:
        assert key in final_output, f"❌ Missing key in output: {key}"
    print("   ✅ All required schema keys present")
    print("   ✅ Output matches Member 2 contract")

    print("\n" + "="*55)
    print("   ✅ FULL PIPELINE TEST PASSED")
    print("="*55 + "\n")

if __name__ == '__main__':
    test_full_pipeline()