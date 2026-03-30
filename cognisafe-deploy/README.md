<div align="center">

<br/>

```
  ╔═══════════════════════════════════════════╗
  ║   🧠  C O G N I S A F E   A I / M L     ║
  ║        Voice Intelligence Pipeline        ║
  ╚═══════════════════════════════════════════╝
```

[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/Whisper_Base-412991?style=flat-square&logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace_Spaces-FF9D00?style=flat-square)](https://alamfarzann-cognisafe-ml.hf.space)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](#)

**A 5-stage voice analysis engine that extracts 14 cognitive biomarkers from a single audio recording.**

> 🔗 **Live API** → `https://alamfarzann-cognisafe-ml.hf.space`

</div>

---

## What it does

Upload any voice recording. In ~90 seconds, the pipeline returns:

- **14 biomarkers** — acoustic + linguistic signals extracted from speech
- **Anomaly flags** — deviations from your *personal* historical baseline (2-sigma)
- **Risk tier** — Green / Yellow / Orange / Red
- **95% Confidence intervals** — per biomarker, per user

---

## Architecture at a glance

```
Browser Audio (WebM/Opus)
        │
        ▼
 ┌─────────────┐     ffmpeg
 │  Stage 1    │ ──────────────→  16kHz mono WAV
 └─────────────┘
        │
        ▼
 ┌─────────────┐     Whisper base
 │  Stage 2    │ ──────────────→  transcript + word timestamps + pause events
 └─────────────┘
        │
   ┌────┴────┐
   ▼         ▼
 ┌──────┐  ┌──────┐
 │  S3  │  │  S4  │   librosa (acoustic)  +  spaCy + MiniLM (NLP)
 └──────┘  └──────┘
   │         │
   └────┬────┘
        ▼
 ┌─────────────┐     risk.py
 │  Stage 5    │ ──────────────→  merge → 14 biomarkers → anomaly → risk tier
 └─────────────┘
        │
        ▼
  JSON response
```

---

## The 14 Biomarkers

> Stages 3 & 4 extract 10 acoustic + 4 linguistic signals in parallel.

### 🎙️ Acoustic (librosa)

| # | Biomarker | Method | Normal Range |
|---|-----------|--------|-------------|
| 1 | `speech_rate` | words / total duration × 60 | 100–180 wpm |
| 2 | `articulation_rate` | words / speech-only duration × 60 | 130–220 wpm |
| 3 | `pause_frequency` | pause count / minute | 5–30 /min |
| 4 | `pause_duration_mean` | mean gap > 200ms | 0.3–0.8 s |
| 5 | `filled_pause_rate` | uh/um count / minute | 0–5 /min |
| 6 | `pitch_mean` | pYIN mean F0 (voiced frames) | person-relative |
| 7 | `pitch_range` | peak-to-peak F0 | person-relative |
| 8 | `jitter` | cycle-to-cycle pitch variation | < 0.05 |
| 9 | `shimmer` | cycle-to-cycle amplitude variation | < 0.15 |
| 10 | `HNR` | harmonics-to-noise ratio | > 10 dB |

### 📝 Linguistic (spaCy + MiniLM)

| # | Biomarker | Method | Normal Range |
|---|-----------|--------|-------------|
| 11 | `lexical_diversity` | MTLD (fwd + bwd avg) | 40–100+ |
| 12 | `semantic_coherence` | cosine similarity, consecutive sentences | 0.3–0.8 |
| 13 | `idea_density` | propositions / word count | 0.3–0.6 |
| 14 | `syntactic_complexity` | mean dependency parse tree depth | 2.0–8.0 |

---

## Pipeline stages in detail

<details>
<summary><strong>Stage 1 — ffmpeg conversion</strong></summary>

Converts browser-recorded WebM/Opus to 16kHz mono WAV (Whisper's optimal format). Skipped if input is already `.wav`.

```python
subprocess.run([
    'ffmpeg', '-y', '-i', raw_path,
    '-ar', '16000', '-ac', '1', '-f', 'wav', temp_path
], timeout=60)
```

**Supported formats:** `.wav` `.mp3` `.m4a` `.ogg` `.flac` `.webm` `.weba` `.opus`

</details>

<details>
<summary><strong>Stage 2 — Whisper transcription</strong></summary>

Runs `whisper-base` (CPU, ~60–90s for 3 min audio) to produce text + word-level timestamps. Pauses are detected as inter-word gaps > 200ms.

```python
for i in range(1, len(words)):
    gap = words[i]['start'] - words[i-1]['end']
    if gap > 0.2:
        pause_events.append({ ... })
```

Fallback: if Whisper fails, returns an empty structure — the pipeline continues rather than crashing.

</details>

<details>
<summary><strong>Stage 3 — Acoustic feature extraction</strong></summary>

Uses **librosa** (pure Python, no binary required). Key algorithms:

```python
# Pitch — probabilistic YIN
f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=2093)

# Jitter — cycle-to-cycle pitch period variation
periods = 1.0 / (voiced_f0 + 1e-9)
jitter  = np.mean(np.abs(np.diff(periods))) / np.mean(periods)

# HNR — harmonic-to-noise energy ratio
harmonics = librosa.effects.harmonic(y)
hnr = 10 * np.log10(np.sum(harmonics**2) / np.sum((y - harmonics)**2))
```

</details>

<details>
<summary><strong>Stage 4 — NLP analysis</strong></summary>

**Lexical diversity** — MTLD: tracks TTR degradation along the text.  
**Semantic coherence** — cosine similarity between consecutive sentence embeddings (MiniLM-L6-v2). Values below 0.3 suggest disjointed or tangential speech.  
**Idea density** — proposition-bearing POS tags (VERB, ADJ, ADV, ADP) / total words.  
**Syntactic complexity** — mean depth of spaCy dependency parse trees.

</details>

<details>
<summary><strong>Stage 5 — Anomaly detection & risk tier</strong></summary>

Compares each biomarker against the user's **own** historical baseline (requires ≥ 3 past sessions).

```python
deviation = abs(current - mean) / std  # z-score

severity = (
    'severe'   if deviation >= 3.0 else
    'moderate' if deviation >= 2.5 else
    'mild'     if deviation >= 2.0 else None
)
```

**Risk tier rules:**

| Tier | Condition |
|------|-----------|
| 🟢 Green | No flags |
| 🟡 Yellow | 2+ mild **or** 1 moderate |
| 🟠 Orange | 2+ moderate **or** 1 severe |
| 🔴 Red | 2+ severe **or** 3+ moderate |

</details>

---

## API Reference

Base URL: `https://alamfarzann-cognisafe-ml.hf.space`

### `GET /health`
Wake-up ping. Returns `{ "status": "ok" }`.

---

### `POST /analyze`

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `audio` | File | Any supported audio format |
| `user_id` | string | For longitudinal tracking (default: `"demo_user"`) |

**Response `200`:**
```json
{
  "session_id": "f47ac10b-...",
  "user_id": "42",
  "timestamp": "2026-03-29T10:30:00.123456",
  "processing_time_seconds": 74.2,
  "biomarkers": {
    "speech_rate": 146.99,
    "semantic_coherence": 0.3447,
    "lexical_diversity": 178.52
    // ...14 total
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
  "risk_tier": "Yellow",
  "confidence_intervals": {
    "speech_rate": { "mean": 146.99, "std": 2.10, "lower_95": 142.87, "upper_95": 151.11 }
  }
}
```

---

### `POST /compare`

Diff two session objects — returns direction + magnitude of change per biomarker.

```json
// Request
{ "session_a": { "biomarkers": { "speech_rate": 150.0 } },
  "session_b": { "biomarkers": { "speech_rate": 130.0 } } }

// Response
{ "diff": { "speech_rate": { "change": -20.0, "change_pct": -13.33, "direction": "down" } } }
```

---

## Database

SQLite at `data/sessions.db` — per-user session history for personalized anomaly detection.

```sql
CREATE TABLE sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    biomarkers    TEXT NOT NULL,   -- JSON, 14 values
    risk_tier     TEXT NOT NULL,
    anomaly_flags TEXT NOT NULL    -- JSON
);
```

> ⚠️ HuggingFace free tier has an ephemeral filesystem — the SQLite DB resets on Space restarts. The backend PostgreSQL database is the production source of truth; SQLite serves as a fast local cache for in-session anomaly detection.

---

## Project structure

```
cognisafe-deploy/
├── Dockerfile
├── requirements.txt
├── api/
│   └── main.py               # /analyze  /compare  /health
├── pipeline/
│   ├── acoustic.py           # librosa — 10 acoustic biomarkers
│   ├── transcription.py      # Whisper + pause detection
│   ├── nlp.py                # spaCy + MiniLM — 4 linguistic biomarkers
│   ├── anomaly.py            # 2-sigma detection + risk tier + SQLite
│   └── risk.py               # merge acoustic + NLP → 14 biomarkers
└── data/
    └── sessions.db
```

---

## Getting started locally

**Prerequisites:** Python 3.11+, ffmpeg in PATH

```bash
# 1. Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run
uvicorn api.main:app --host 0.0.0.0 --port 7860

# 3. Test
curl http://localhost:7860/health
curl -X POST http://localhost:7860/analyze \
     -F "audio=@test_audio.wav" -F "user_id=test_user"

# 4. Docs
open http://localhost:7860/docs
```

---

## Deployment (HuggingFace Spaces)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg git
COPY . /app && WORKDIR /app
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

**Cold start:** ~60–90 seconds (Whisper + spaCy + MiniLM load). The frontend pings `/health` on page load and sets a 480s request timeout.

Push to the HF Space repo → auto-rebuild.

---

## Design decisions

| Decision | Reason |
|----------|--------|
| **librosa over openSMILE** | Pure Python pip install — no binary config for Docker/Linux |
| **Whisper base over large** | 140 MB vs 3 GB; ~90s vs ~10 min on CPU |
| **SQLite over PostgreSQL** | Zero config inside Docker; PostgreSQL is production source of truth |
| **HF called directly from frontend** | Bypasses Render's 30s timeout; ML processing takes ~90s |
| **Personalized baseline over population norms** | Individual biomarker ranges vary too much for norms to be meaningful; personal delta is the signal |

---

<div align="center">

Built by **Farjan Alam** · Team FAIV 🤖

[![Frontend](https://img.shields.io/badge/Frontend-Vercel-000000?style=flat-square)](https://cogni-safe.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square)](https://github.com/SyntaxSaviour/CogniSafe)
[![Live API](https://img.shields.io/badge/Live_API-HuggingFace-FF9D00?style=flat-square)](https://alamfarzann-cognisafe-ml.hf.space/health)

</div>
