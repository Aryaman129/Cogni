<div align="center">

<img src="https://img.shields.io/badge/CogniSafe-AI%2FML%20Pipeline-0A1628?style=for-the-badge&logoColor=E8A020" />

# 🤖 CogniSafe AI/ML Pipeline
### *The technical heart of cognitive health monitoring.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper%20Base-412991?style=for-the-badge&logo=openai)](https://github.com/openai/whisper)
[![HuggingFace](https://img.shields.io/badge/🤗%20Deployed-HuggingFace%20Spaces-FF9D00?style=for-the-badge)](https://alamfarzann-cognisafe-ml.hf.space)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://docker.com)

> **Live API:** `https://alamfarzann-cognisafe-ml.hf.space`

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Pipeline Architecture](#-pipeline-architecture)
- [Stage 1 — Audio Conversion](#-stage-1--audio-conversion-ffmpeg)
- [Stage 2 — Whisper Transcription](#-stage-2--whisper-transcription)
- [Stage 3 — Acoustic Feature Extraction](#-stage-3--acoustic-feature-extraction)
- [Stage 4 — NLP Analysis](#-stage-4--nlp-analysis)
- [Stage 5 — Anomaly Detection & Risk Tier](#-stage-5--anomaly-detection--risk-tier)
- [The 14 Biomarkers](#-the-14-biomarkers)
- [API Reference](#-api-reference)
- [Database](#-database)
- [Getting Started Locally](#-getting-started-locally)
- [Deployment — HuggingFace Spaces](#-deployment--huggingface-spaces)
- [Design Decisions](#-design-decisions)

---

## 🧠 Overview

The CogniSafe AI/ML pipeline is a **5-stage voice analysis engine** deployed as a FastAPI service on HuggingFace Spaces. It accepts a raw audio file from the browser, processes it through a sequence of acoustic and linguistic analysis stages, and returns a complete cognitive health snapshot — **14 biomarkers, anomaly flags, a risk tier, and 95% confidence intervals** — in a single JSON response.

Every intelligent feature in CogniSafe flows through this service. The pipeline is designed to be:

- **Self-contained** — one Docker container, one command to start
- **Fault-tolerant** — every stage has a fallback that returns safe defaults instead of crashing
- **Platform-agnostic** — runs on CPU with no GPU requirement
- **Stateful per user** — longitudinal session history stored in SQLite, enabling personalized anomaly detection

---

## 📁 Project Structure

```
cognisafe-deploy/
│
├── Dockerfile                  # Docker build — Python 3.11 slim + ffmpeg + dependencies
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── api/
│   ├── __init__.py
│   └── main.py                 # FastAPI app — /analyze, /compare, /health endpoints
│
├── pipeline/
│   ├── __init__.py
│   ├── acoustic.py             # Acoustic feature extraction (librosa)
│   ├── transcription.py        # Whisper speech-to-text + pause detection
│   ├── nlp.py                  # NLP biomarkers (spaCy + sentence-transformers)
│   ├── anomaly.py              # Anomaly detection + risk tier + SQLite storage
│   └── risk.py                 # Biomarker merger (acoustic + NLP → 14 biomarkers)
│
└── data/
    └── sessions.db             # SQLite — longitudinal session history per user
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI + Uvicorn** | REST API server, async request handling |
| **OpenAI Whisper (base)** | Speech-to-text + word timestamps |
| **librosa** | Acoustic feature extraction (pitch, jitter, shimmer, HNR, pauses) |
| **spaCy (en_core_web_sm)** | POS tagging, dependency parsing, sentence segmentation |
| **sentence-transformers (MiniLM-L6-v2)** | Sentence embeddings for semantic coherence |
| **scikit-learn** | Cosine similarity computation |
| **numpy / scipy** | Statistical analysis, signal processing |
| **soundfile** | Audio file I/O |
| **SQLite** | Lightweight longitudinal session storage per user |
| **ffmpeg** | Audio format conversion (WebM → WAV) |
| **Docker** | Containerization for HuggingFace Spaces deployment |
| **python-multipart** | Multipart audio file upload handling |

---

## 🏗️ Pipeline Architecture

```
Browser Audio (WebM/Opus from MediaRecorder)
             │
             ▼
┌────────────────────────────────────────────────────────────┐
│                   POST /analyze                            │
│                   api/main.py                              │
└────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 1: ffmpeg Conversion │  WebM → 16kHz mono WAV
│  (api/main.py)              │  subprocess call, 60s timeout
└─────────────────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: Whisper           │  Speech → text + word timestamps
│  (pipeline/transcription.py)│  pause events (gaps > 200ms)
└─────────────────────────────┘
             │
             ├──────────────────────────────────┐
             ▼                                  ▼
┌─────────────────────────────┐    ┌──────────────────────────┐
│  Stage 3: Acoustic          │    │  Stage 4: NLP Analysis   │
│  (pipeline/acoustic.py)     │    │  (pipeline/nlp.py)       │
│                             │    │                          │
│  • Pitch mean + range       │    │  • Semantic coherence    │
│  • Jitter                   │    │  • Lexical diversity     │
│  • Shimmer                  │    │  • Idea density          │
│  • HNR                      │    │  • Syntactic complexity  │
│  • Speech rate              │    └──────────────────────────┘
│  • Articulation rate        │                │
│  • Pause frequency          │                │
│  • Pause duration mean      │                │
│  • Filled pause rate        │                │
└─────────────────────────────┘                │
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
             ┌──────────────────────────┐
             │  risk.py: merge_biomarkers│  Combines → 14 biomarker dict
             └──────────────────────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │  Stage 5: Anomaly        │  2-sigma deviation vs baseline
             │  (pipeline/anomaly.py)   │  Risk tier scoring
             │                          │  SQLite session storage
             └──────────────────────────┘
                            │
                            ▼
              JSON Response (14 biomarkers +
              risk tier + anomaly flags +
              confidence intervals)
```

---

## 🔄 Stage 1 — Audio Conversion (ffmpeg)

**File:** `api/main.py`

The browser records audio using `MediaRecorder` which produces **WebM/Opus** format. librosa and Whisper work best with WAV files. The pipeline converts all incoming audio to **16kHz mono WAV** before processing.

```python
subprocess.run([
    'ffmpeg', '-y',
    '-i', raw_path,       # input: WebM/MP3/M4A/OGG
    '-ar', '16000',       # 16kHz sample rate (Whisper optimal)
    '-ac', '1',           # mono channel
    '-f', 'wav',
    temp_path             # output: WAV
], timeout=60)
```

**Supported input formats:** `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.weba`, `.opus`

If the input is already `.wav`, the conversion step is skipped entirely.

---

## 🗣️ Stage 2 — Whisper Transcription

**File:** `pipeline/transcription.py`

Uses **OpenAI Whisper base model** (loaded once at module startup) to transcribe the audio. The base model is chosen for CPU performance — it runs in ~60-90 seconds on CPU for a 3-minute file.

### What it extracts

```python
{
    'text':         "In this picture I can see a peaceful park...",
    'words':        [
        { 'word': 'In',      'start': 0.0,  'end': 0.18 },
        { 'word': 'this',    'start': 0.18, 'end': 0.36 },
        ...
    ],
    'pause_events': [
        { 'after_word': 'park', 'before_word': 'the', 'duration': 0.42, 'start_time': 3.2 },
        ...
    ],
    'word_count':   219,
    'duration':     89.4
}
```

### Pause detection logic

A pause is defined as a gap between consecutive words longer than **200ms**:

```python
for i in range(1, len(words)):
    gap = words[i]['start'] - words[i - 1]['end']
    if gap > 0.2:   # 200ms threshold
        pause_events.append({ ... })
```

### Fallback

If Whisper fails for any reason (ffmpeg missing, corrupt audio, too short), `_fallback_transcript()` returns an empty structure — the pipeline continues with zero word-count based metrics rather than crashing.

---

## 🎵 Stage 3 — Acoustic Feature Extraction

**File:** `pipeline/acoustic.py`

Extracts **10 acoustic biomarkers** using **librosa** — a pure Python audio analysis library that works on any platform without external binaries.

> **Note:** The local development version used openSMILE (eGeMAPS). The deployed version uses librosa for cross-platform compatibility on HuggingFace Spaces.

### Pitch Features

Uses `librosa.pyin` (Probabilistic YIN algorithm) for robust fundamental frequency estimation:

```python
f0, voiced_flag, _ = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),   # ~65 Hz
    fmax=librosa.note_to_hz('C7')    # ~2093 Hz
)
voiced_f0   = f0[voiced_flag]
pitch_mean  = np.mean(voiced_f0)
pitch_range = np.ptp(voiced_f0)      # peak-to-peak range
```

### Jitter (Pitch Perturbation)

Cycle-to-cycle variation in pitch periods — elevated jitter indicates vocal cord irregularity:

```python
periods = 1.0 / (voiced_f0 + 1e-9)
jitter  = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
```

### Shimmer (Amplitude Perturbation)

Cycle-to-cycle variation in amplitude — elevated shimmer indicates breathiness:

```python
rms     = librosa.feature.rms(y=y)[0]
shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9)
```

### HNR (Harmonics-to-Noise Ratio)

Ratio of harmonic energy to noise energy — lower HNR indicates hoarser voice:

```python
harmonics = librosa.effects.harmonic(y)
noise     = y - harmonics
hnr       = 10 * np.log10(np.sum(harmonics**2) / np.sum(noise**2))
```

### Pause & Speech Rate

```python
# Non-silent intervals via Voice Activity Detection
intervals = librosa.effects.split(y, top_db=30)

# Pauses = gaps between non-silent intervals > 200ms
pauses = [gap for gap in gaps if gap > 0.2]

pause_frequency     = len(pauses) / (duration / 60)    # pauses per minute
pause_duration_mean = np.mean(pauses)                   # average pause length

speech_rate         = (word_count / duration) * 60      # words per minute (total)
articulation_rate   = (word_count / speech_duration) * 60  # words per minute (excl. pauses)
filled_pause_rate   = count("uh", "um") / (duration / 60)  # from transcript
```

---

## 📝 Stage 4 — NLP Analysis

**File:** `pipeline/nlp.py`

Extracts **4 linguistic biomarkers** from the Whisper transcript using spaCy and sentence-transformers. Both models are loaded once at module startup.

### Lexical Diversity — MTLD

**Measure of Textual Lexical Diversity** — measures vocabulary richness by tracking how quickly the type-token ratio degrades as you read through the text.

```python
def mtld_pass(tokens, threshold=0.72):
    factor_count, token_count = 0, 0
    types = set()
    for token in tokens:
        token_count += 1
        types.add(token)
        ttr = len(types) / token_count
        if ttr <= threshold:          # TTR dropped below threshold
            factor_count += 1         # count as one "factor"
            token_count, types = 0, set()  # reset
    # partial factor at end
    factor_count += (1 - ttr) / (1 - threshold)
    return len(tokens) / factor_count

# MTLD = average of forward and backward pass
mtld = (mtld_pass(tokens) + mtld_pass(reversed(tokens))) / 2
```

**Interpretation:** Higher = more diverse vocabulary = healthier. Typical range: 40–100+. Values below 40 suggest repetitive speech.

---

### Semantic Coherence

Measures sentence-to-sentence logical flow using **cosine similarity** between consecutive sentence embeddings from `all-MiniLM-L6-v2`:

```python
sentences  = [sent.text for sent in doc.sents]
embeddings = EMBEDDER.encode(sentences)

similarities = [
    cosine_similarity(embeddings[i-1], embeddings[i])
    for i in range(1, len(embeddings))
]
semantic_coherence = np.mean(similarities)   # 0.0 to 1.0
```

**Interpretation:** Higher = more logically connected speech. Values below 0.3 suggest disjointed or tangential speech — a known early indicator of cognitive decline.

---

### Idea Density

Propositions per word — measures how much information is packed into speech:

```python
proposition_pos   = {'VERB', 'ADJ', 'ADV', 'ADP'}
proposition_count = len([t for t in doc if t.pos_ in proposition_pos])
idea_density      = proposition_count / total_word_count   # 0.0 to 1.0
```

**Interpretation:** Typical healthy range: 0.3–0.6. Declining idea density correlates with reduced cognitive load capacity.

---

### Syntactic Complexity

Average depth of the dependency parse tree — measures sentence structural complexity:

```python
def get_depth(token, depth=0):
    children = list(token.children)
    if not children:
        return depth
    return max(get_depth(child, depth + 1) for child in children)

depths = [get_depth(root) for root in sentence_roots]
syntactic_complexity = np.mean(depths)   # typical: 2–8
```

**Interpretation:** Declining syntactic complexity (simpler sentences) is a marker of reduced executive function.

---

## 🚨 Stage 5 — Anomaly Detection & Risk Tier

**File:** `pipeline/anomaly.py`

The most clinically significant stage. Instead of comparing against a population average, CogniSafe compares **each user's current session against their own historical baseline** — making the system sensitive to personal change rather than population norms.

### Baseline Requirement

Anomaly detection requires **at least 3 past sessions** to compute a meaningful baseline. New users will always receive `Green` with empty `anomaly_flags` until 3 sessions are recorded.

### 2-Sigma Detection Algorithm

```python
for biomarker in BIOMARKERS:
    historical_values = [session[biomarker] for session in past_sessions]
    
    mean      = np.mean(historical_values)
    std       = np.std(historical_values)
    deviation = abs(current_value - mean) / std   # z-score

    if deviation >= 2.0:
        severity = (
            'severe'   if deviation >= 3.0 else
            'moderate' if deviation >= 2.5 else
            'mild'
        )
        anomaly_flags.append({
            'biomarker': biomarker,
            'severity':  severity,
            'current':   current_value,
            'baseline':  mean,
            'deviation': deviation
        })
```

### Risk Tier Rules

| Tier | Condition |
|---|---|
| 🟢 **Green** | No anomaly flags |
| 🟡 **Yellow** | 2+ mild flags **OR** 1 moderate flag |
| 🟠 **Orange** | 2+ moderate flags **OR** 1 severe flag |
| 🔴 **Red** | 2+ severe flags **OR** 3+ moderate flags |

### 95% Confidence Intervals

For each biomarker with sufficient history, the pipeline computes:

```python
intervals[biomarker] = {
    'mean':     np.mean(values),
    'std':      np.std(values),
    'lower_95': mean - 1.96 * std,
    'upper_95': mean + 1.96 * std,
}
```

These allow the frontend to show the user where their current value sits relative to their personal normal band.

---

## 📊 The 14 Biomarkers

| # | Biomarker | Module | Method | Healthy Range |
|---|---|---|---|---|
| 1 | `speech_rate` | acoustic | word count / total duration × 60 | 100–180 wpm |
| 2 | `articulation_rate` | acoustic | word count / speech duration × 60 | 130–220 wpm |
| 3 | `pause_frequency` | acoustic | pause count / duration in minutes | 5–30 /min |
| 4 | `pause_duration_mean` | acoustic | mean gap between words > 200ms | 0.3–0.8s |
| 5 | `filled_pause_rate` | acoustic | uh/um count / duration in minutes | 0–5 /min |
| 6 | `pitch_mean` | acoustic | librosa.pyin mean F0 (voiced frames) | varies |
| 7 | `pitch_range` | acoustic | peak-to-peak F0 range (voiced frames) | varies |
| 8 | `jitter` | acoustic | mean abs diff of pitch periods / mean period | < 0.05 |
| 9 | `shimmer` | acoustic | mean abs diff of RMS / mean RMS | < 0.15 |
| 10 | `HNR` | acoustic | 10 × log10(harmonic power / noise power) | > 10 dB |
| 11 | `lexical_diversity` | nlp | MTLD (forward + backward average) | 40–100+ |
| 12 | `semantic_coherence` | nlp | mean cosine similarity consecutive sentences | 0.3–0.8 |
| 13 | `idea_density` | nlp | propositions / total words | 0.3–0.6 |
| 14 | `syntactic_complexity` | nlp | mean dependency parse tree depth | 2.0–8.0 |

---

## 📡 API Reference

Base URL: `https://alamfarzann-cognisafe-ml.hf.space`

---

### `GET /health`

Health check — use this to wake up the HuggingFace Space before a session.

**Response `200`:**
```json
{
  "status": "ok",
  "service": "CogniSafe AI Pipeline",
  "timestamp": "2026-03-29T10:00:00.000000"
}
```

---

### `POST /analyze`

Main analysis endpoint. Accepts audio, runs the full 5-stage pipeline.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `audio` | File | ✅ | Audio file (WAV, MP3, M4A, WebM, OGG, FLAC, OPUS) |
| `user_id` | string | ❌ | User identifier for longitudinal tracking (default: `"demo_user"`) |

**Response `200`:**
```json
{
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "user_id": "42",
  "timestamp": "2026-03-29T10:30:00.123456",
  "processing_time_seconds": 74.2,
  "biomarkers": {
    "speech_rate":         146.99,
    "articulation_rate":   205.59,
    "pause_frequency":     26.85,
    "pause_duration_mean": 0.476,
    "filled_pause_rate":   0.0,
    "pitch_mean":          33.42,
    "pitch_range":         8.06,
    "jitter":              0.051739,
    "shimmer":             1.2724,
    "HNR":                 5.6108,
    "lexical_diversity":   178.52,
    "semantic_coherence":  0.3447,
    "idea_density":        0.4201,
    "syntactic_complexity": 5.077
  },
  "anomaly_flags": [
    {
      "biomarker": "semantic_coherence",
      "severity":  "mild",
      "current":   0.31,
      "baseline":  0.34,
      "deviation": 2.1
    }
  ],
  "risk_tier": "Yellow",
  "confidence_intervals": {
    "speech_rate": {
      "mean":     146.99,
      "std":      2.10,
      "lower_95": 142.87,
      "upper_95": 151.11
    }
  }
}
```

**Errors:**
| Code | Reason |
|---|---|
| `400` | Unsupported audio format |
| `500` | Internal pipeline error (check logs) |

---

### `POST /compare`

Diff two session JSON objects. Returns change direction and magnitude per biomarker.

**Request Body:**
```json
{
  "session_a": { "biomarkers": { "speech_rate": 150.0, ... }, "timestamp": "..." },
  "session_b": { "biomarkers": { "speech_rate": 130.0, ... }, "timestamp": "..." }
}
```

**Response `200`:**
```json
{
  "timestamp_a": "2026-02-01T...",
  "timestamp_b": "2026-03-29T...",
  "diff": {
    "speech_rate": {
      "session_a":  150.0,
      "session_b":  130.0,
      "change":     -20.0,
      "change_pct": -13.33,
      "direction":  "down"
    }
  }
}
```

---

## 🗄️ Database

**File:** `pipeline/anomaly.py` — SQLite via Python's built-in `sqlite3`

The pipeline maintains a local SQLite database at `data/sessions.db` inside the container. This enables longitudinal anomaly detection — comparing each session against the user's personal history.

### Schema

```sql
CREATE TABLE sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,       -- ISO8601 UTC
    biomarkers    TEXT    NOT NULL,       -- JSON string of 14 biomarker values
    risk_tier     TEXT    NOT NULL,       -- Green / Yellow / Orange / Red
    anomaly_flags TEXT    NOT NULL        -- JSON string of anomaly flag objects
);
```

### Session lifecycle

```
1. load_sessions(user_id)        ← read past sessions for baseline
2. detect_anomalies(...)         ← compare current vs baseline
3. compute_risk_tier(flags)      ← aggregate severity → tier
4. compute_confidence_intervals  ← 95% CI per biomarker
5. save_session(...)             ← write current session to DB
```

> ⚠️ **HuggingFace Spaces note:** The free tier has an ephemeral filesystem — the SQLite DB resets when the Space restarts. For persistent longitudinal tracking, the backend PostgreSQL database (via the `/api/sessions` endpoint) is the source of truth. The ML pipeline's SQLite is used for in-session anomaly detection during the active deployment cycle.

---

## 🚀 Getting Started Locally

### Prerequisites

- Python 3.11+
- ffmpeg installed and in PATH

**Install ffmpeg on Windows:**
```bash
# Using winget
winget install Gyan.FFmpeg

# Using chocolatey
choco install ffmpeg
```

### 1. Create virtual environment

```bash
cd cognisafe-deploy
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Start the server

```bash
cd api
python main.py
# Runs on http://localhost:7860
```

Or from the root:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

### 4. Test the health check

```bash
curl http://localhost:7860/health
```

### 5. Test the analyze endpoint

```bash
curl -X POST http://localhost:7860/analyze \
  -F "audio=@path/to/test_audio.wav" \
  -F "user_id=test_user"
```

### 6. View auto-generated API docs

```
http://localhost:7860/docs       # Swagger UI
http://localhost:7860/redoc      # ReDoc
```

---

## 🐳 Deployment — HuggingFace Spaces

The pipeline is containerized with Docker and deployed on **HuggingFace Spaces** (Docker SDK).

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN mkdir -p data

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### HuggingFace Space config (`README.md` front matter)

```yaml
title: Cognisafe ML
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
```

### Cold start behaviour

HuggingFace free tier Spaces sleep after ~15 minutes of inactivity. On wake:
- Docker container starts
- Whisper base model loads (~5-10 seconds)
- spaCy + sentence-transformers load (~10-15 seconds)
- Total cold start: ~60-90 seconds

The frontend handles this with:
1. Health ping on app load (`App.jsx`)
2. Health ping when session page opens (`Session.jsx`)
3. 480-second request timeout (`sessionService.js`)

### Redeploying

Push changes directly to the HuggingFace Space repository, or update files via the HF web editor. The Space rebuilds automatically on each push.

---

## 🧩 Design Decisions

### Why librosa instead of openSMILE?

The original local pipeline used **openSMILE** (a dedicated acoustic analysis binary) for eGeMAPS feature extraction. For HuggingFace Spaces deployment, openSMILE was replaced with **librosa** because:

- librosa is a pure Python pip package — no binary download required
- Works identically on Linux (Docker) and Windows without path configuration
- Covers all required biomarkers with comparable accuracy
- Simplifies the Dockerfile to a single `pip install`

### Why Whisper base instead of large?

- `base` model: ~140MB, ~60s on CPU for 3 min audio ✅
- `large-v3` model: ~3GB, ~8-12 min on CPU ❌

The base model achieves sufficient transcription accuracy for biomarker extraction. Word-level timestamps (used for pause detection) are reliable even at base quality.

### Why SQLite instead of PostgreSQL?

The ML pipeline uses **SQLite** for longitudinal session storage because:
- Zero configuration — no database server required
- Runs inside the Docker container
- Sufficient for per-user session history lookups
- The backend PostgreSQL is the production source of truth — SQLite serves as a fast local cache for anomaly detection within the active deployment

### Why call HF directly from the frontend?

The backend (Render free tier) has a **30-second request timeout** which is insufficient for ML processing (~90 seconds). The frontend calls HF directly to bypass this limit, then saves the result to the backend separately.

### Why personalized baseline instead of population norms?

Cognitive biomarkers vary enormously between individuals — a speech rate of 120 wpm might be perfectly normal for one person and significantly reduced for another. By comparing each user against their **own historical baseline**, CogniSafe detects meaningful personal change rather than flagging natural individual variation. This approach requires 3+ sessions before anomaly detection activates.

---

<div align="center">

**CogniSafe AI/ML Pipeline** — Built by **Farjan Alam** · Team FAIV 🤖

*Part of the CogniSafe cognitive health monitoring platform.*

[![Frontend](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge)](https://cogni-safe.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge)](https://github.com/SyntaxSaviour/CogniSafe)
[![Live API](https://img.shields.io/badge/Live%20API-HuggingFace-FF9D00?style=for-the-badge)](https://alamfarzann-cognisafe-ml.hf.space/health)

</div>
