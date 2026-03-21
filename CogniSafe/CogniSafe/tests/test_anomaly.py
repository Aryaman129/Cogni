import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.anomaly import (
    init_db, save_session, load_sessions,
    detect_anomalies, compute_risk_tier,
    run_anomaly_pipeline
)

# ── Sample biomarkers for testing ─────────────────────────────────────────────
HEALTHY_SESSION = {
    'speech_rate': 145.0, 'articulation_rate': 200.0,
    'pause_frequency': 25.0, 'pause_duration_mean': 0.45,
    'filled_pause_rate': 2.0, 'pitch_mean': 33.0,
    'pitch_range': 8.0, 'jitter': 0.05,
    'shimmer': 1.2, 'HNR': 5.5,
    'lexical_diversity': 175.0, 'semantic_coherence': 0.35,
    'idea_density': 0.42, 'syntactic_complexity': 5.0
}

def make_noisy(base: dict, noise=0.05) -> dict:
    """Add small random noise to simulate natural session variation."""
    import random
    return {k: round(v * (1 + random.uniform(-noise, noise)), 4)
            for k, v in base.items()}

def make_anomalous(base: dict) -> dict:
    """Create a clearly anomalous session — big drop in key biomarkers."""
    anomalous = base.copy()
    anomalous['semantic_coherence']   = 0.05   # huge drop
    anomalous['lexical_diversity']    = 40.0   # huge drop
    anomalous['speech_rate']          = 50.0   # huge drop
    return anomalous


def test_anomaly_pipeline():
    TEST_USER = 'test_user_001'

    print("\n── Step 1: Initialize DB ──")
    init_db()

    print("\n── Step 2: Seed 5 healthy sessions ──")
    for i in range(5):
        session = make_noisy(HEALTHY_SESSION)
        save_session(TEST_USER, session, 'Green', [])
        print(f"   Saved healthy session {i+1}")

    print("\n── Step 3: Load sessions back ──")
    sessions = load_sessions(TEST_USER)
    print(f"   Loaded {len(sessions)} sessions from DB")
    assert len(sessions) >= 5, "❌ Sessions not saved correctly"

    print("\n── Step 4: Run anomaly detection on healthy session ──")
    healthy_result = run_anomaly_pipeline(TEST_USER, make_noisy(HEALTHY_SESSION))
    print(f"   Risk Tier    : {healthy_result['risk_tier']}")
    print(f"   Anomaly Flags: {healthy_result['anomaly_flags']}")
    print(f"   Confidence intervals computed: {len(healthy_result['confidence_intervals'])} biomarkers")

    print("\n── Step 5: Run anomaly detection on ANOMALOUS session ──")
    anomalous_result = run_anomaly_pipeline(TEST_USER, make_anomalous(HEALTHY_SESSION))
    print(f"   Risk Tier    : {anomalous_result['risk_tier']}")
    print(f"   Anomaly Flags:")
    for flag in anomalous_result['anomaly_flags']:
        print(f"   ⚠️  {flag['biomarker']}: {flag['severity']} "
              f"(current={flag['current']}, baseline={flag['baseline']}, "
              f"deviation={flag['deviation']}σ)")

    print("\n── Step 6: Verify risk tier logic ──")
    assert compute_risk_tier([]) == 'Green', "❌ Empty flags should be Green"
    assert compute_risk_tier([{'severity': 'mild'}] * 2) == 'Yellow', "❌ 2 mild should be Yellow"
    assert compute_risk_tier([{'severity': 'moderate'}]) == 'Yellow', "❌ 1 moderate should be Yellow"
    assert compute_risk_tier([{'severity': 'moderate'}] * 2) == 'Orange', "❌ 2 moderate should be Orange"
    assert compute_risk_tier([{'severity': 'severe'}]) == 'Orange', "❌ 1 severe should be Orange"
    assert compute_risk_tier([{'severity': 'severe'}] * 2) == 'Red', "❌ 2 severe should be Red"
    assert compute_risk_tier([{'severity': 'moderate'}] * 3) == 'Red', "❌ 3 moderate should be Red"
    print("   ✅ All risk tier rules verified!")

    print("\n✅ All anomaly detection tests passed!")

if __name__ == '__main__':
    test_anomaly_pipeline()