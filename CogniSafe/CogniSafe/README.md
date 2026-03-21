# CogniSafe — AI Pipeline (Member 1)

## Setup

1. Clone the repo
2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

4. Start the service:
   cd api
   python main.py

Service runs on: http://localhost:8001

---

## API Contract

### POST /analyze

Accepts multipart/form-data:
- audio   : WAV/MP3/M4A audio file (required)
- user_id : string (optional, default: "demo_user")

Returns JSON:
{
  "session_id": "uuid string",
  "user_id": "string",
  "timestamp": "ISO8601 string",
  "processing_time_seconds": 12.4,
  "biomarkers": {
    "speech_rate": 146.99,
    "articulation_rate": 205.59,
    "pause_frequency": 26.85,
    "pause_duration_mean": 0.476,
    "filled_pause_rate": 0.0,
    "pitch_mean": 33.42,
    "pitch_range": 8.06,
    "jitter": 0.0517,
    "shimmer": 1.272,
    "HNR": 5.61,
    "lexical_diversity": 178.52,
    "semantic_coherence": 0.3447,
    "idea_density": 0.4201,
    "syntactic_complexity": 5.077
  },
  "anomaly_flags": [
    {
      "biomarker": "semantic_coherence",
      "severity": "mild",
      "current": 0.31,
      "baseline": 0.34,
      "deviation": 2.1
    }
  ],
  "risk_tier": "Green",
  "confidence_intervals": {
    "speech_rate": {
      "mean": 146.99,
      "std": 2.1,
      "lower_95": 142.8,
      "upper_95": 151.1
    }
  }
}

### GET /health

Returns: { "status": "ok" }

### POST /compare

Accepts two session JSON objects.
Returns diff of biomarker changes between sessions.

---

## Notes for Member 2

- Service runs on port 8001
- Call POST /analyze with multipart audio file
- user_id should match your users table
- First 2 sessions return empty anomaly_flags (need 3+ for baseline)
- Processing time: ~60-90s on CPU for a 3-minute audio file
- Add 35s timeout on your side as a safety net
```

---

### Step 4 — Final folder structure check

Your project should now look exactly like this:
```
CogniSafe/
├── api/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── test_audio.wav
├── pipeline/
│   ├── __init__.py
│   ├── acoustic.py
│   ├── anomaly.py
│   ├── nlp.py
│   ├── risk.py
│   └── transcription.py
├── tests/
│   ├── test_acoustic.py
│   ├── test_anomaly.py
│   ├── test_api.py
│   ├── test_full_pipeline.py
│   ├── test_nlp.py
│   └── test_transcription.py
├── opensmile/
├── venv/
├── requirements.txt
├── schema.json
├── start.bat
└── README.md