import requests
import json
import os

def test_api():
    audio_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_audio.wav')

    print("Testing /health endpoint...")
    health = requests.get("http://localhost:8001/health")
    print(f"   Status: {health.json()}")
    assert health.status_code == 200, "❌ Health check failed"

    print("\nTesting /analyze endpoint (this will take ~60-90s on CPU)...")
    with open(audio_path, 'rb') as f:
        response = requests.post(
            "http://localhost:8001/analyze",
            files={"audio": ("test_audio.wav", f, "audio/wav")},
            data={"user_id": "demo_user"}
        )

    assert response.status_code == 200, f"❌ Analyze failed: {response.text}"

    result = response.json()

    print("\n✅ API Response:")
    print(f"   session_id         : {result['session_id']}")
    print(f"   processing_time    : {result['processing_time_seconds']}s")
    print(f"   risk_tier          : {result['risk_tier']}")
    print(f"   anomaly_flags      : {len(result['anomaly_flags'])} flags")
    print(f"\n   Biomarkers:")
    for k, v in result['biomarkers'].items():
        print(f"      {k}: {v}")

    assert 'biomarkers'    in result, "❌ Missing biomarkers"
    assert 'risk_tier'     in result, "❌ Missing risk_tier"
    assert 'anomaly_flags' in result, "❌ Missing anomaly_flags"
    assert len(result['biomarkers']) == 14, "❌ Should have exactly 14 biomarkers"

    print("\n✅ All API tests passed!")

if __name__ == '__main__':
    test_api()