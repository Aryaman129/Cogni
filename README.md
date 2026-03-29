<div align="center">

<img src="https://img.shields.io/badge/CogniSafe-Cognitive%20Health%20AI-0A1628?style=for-the-badge&logo=brain&logoColor=E8A020" />

# 🧠 CogniSafe
### *Hear the Difference. Before It's Too Late.*

> **AI-powered voice biomarker analysis for early cognitive health monitoring.**
> Detect subtle changes in speech patterns that may indicate early cognitive decline —
> using just your voice.

<br/>

[![Live Demo](https://img.shields.io/badge/🌐%20Live%20Demo-cogni--safe.vercel.app-E8A020?style=for-the-badge)](https://cogni-safe.vercel.app)
[![ML API](https://img.shields.io/badge/🤗%20ML%20API-HuggingFace%20Spaces-FF9D00?style=for-the-badge)](https://alamfarzann-cognisafe-ml.hf.space/health)
[![License](https://img.shields.io/badge/License-MIT-0A1628?style=for-the-badge)](LICENSE)
[![Built With](https://img.shields.io/badge/Built%20With-❤️%20at%20Hackathon-red?style=for-the-badge)]()

<br/>

```
Every 3 seconds, someone in the world develops dementia.
50 million people are living with it today.
Most are diagnosed years too late.
CogniSafe gives families the one thing they desperately want — time.
```

<br/>

</div>

---

## 📖 Table of Contents

- [🧠 What is CogniSafe?](#-what-is-cognisafe)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🤖 AI/ML Pipeline](#-aiml-pipeline)
- [🖥️ Frontend](#️-frontend)
- [⚙️ Backend](#️-backend)
- [🚀 Getting Started](#-getting-started)
- [📡 API Reference](#-api-reference)
- [👥 Team](#-team)
- [🗺️ Roadmap](#️-roadmap)

---

## 🧠 What is CogniSafe?

CogniSafe is a **non-invasive, voice-first cognitive health monitoring platform**. Instead of expensive MRI scans or clinical neuropsychological tests, CogniSafe asks you to speak for 2 minutes — and lets AI do the rest.

By analyzing **14 voice biomarkers** across acoustic, linguistic, and cognitive dimensions, CogniSafe builds a longitudinal picture of your cognitive health over time. It detects **subtle drift** in speech patterns — the kind of changes that are invisible to the human ear but statistically significant to machine learning models trained on cognitive health data.

### 🎯 Who is it for?

| Persona | Use Case |
|---|---|
| 👨‍👩‍👧 **Remote Caregivers** | Monitor an aging parent's cognitive health from across the country |
| 🏈 **Retired Athletes** | Track potential CTE-related cognitive changes over time |
| 👫 **Concerned Spouses** | Get early, objective signals before a clinical diagnosis |
| 🏥 **Neurologists** | Supplement clinical visits with longitudinal voice data |

---

## ✨ Key Features

### 🎙️ Voice Session Recording
- **Daily 2-minute sessions** with a picture description task (clinical standard)
- **Live voice orb** that pulses in real-time with your speech amplitude
- **WebRTC audio capture** directly in the browser — no app download needed
- Automatic conversion and processing of browser-recorded audio

### 📊 14-Biomarker Analysis Engine
- **10 Acoustic Biomarkers** extracted via openSMILE eGeMAPS
- **4 NLP Biomarkers** extracted via spaCy + sentence-transformers
- Full analysis returned in under 90 seconds on CPU

### 🔴 Risk Tier System
| Tier | Meaning |
|---|---|
| 🟢 **Green** | No anomalies detected — baseline stable |
| 🟡 **Yellow** | Mild deviations — worth monitoring |
| 🟠 **Orange** | Moderate anomalies — recommend follow-up |
| 🔴 **Red** | Significant deviation — seek professional advice |

### 📈 Longitudinal Trend Dashboard
- **14 biomarker cards** each with 90-day sparkline charts
- **Personal baseline** — compared against YOUR own history, not population averages
- **2-sigma anomaly detection** — statistically rigorous deviation flagging
- Confidence intervals showing model certainty

### 🧬 3D Semantic Drift Sphere
- Interactive Three.js visualization of your **vocabulary over time**
- Word nodes sized by frequency, colored by recency
- **Time slider** — drag to see how your vocabulary has changed over 6 months
- The most powerful visual in cognitive health tech

### 👨‍👩‍👧 Caregiver Dashboard
- Plain-language status: *"Dad is doing WELL this week"*
- 30-day calendar heat map
- Zero medical jargon — designed for family members, not clinicians
- PDF report download for sharing with a neurologist

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER BROWSER                         │
│                                                             │
│   ┌─────────────┐    Records Audio (WebM)                  │
│   │  React.js   │ ──────────────────────────────────┐      │
│   │  Frontend   │                                   │      │
│   │  (Vercel)   │ ◄── Session Results + Biomarkers  │      │
│   └─────────────┘                                   │      │
│          │                                          │      │
└──────────│──────────────────────────────────────────│──────┘
           │ REST API calls                           │
           │ (auth, sessions, history)                │ Direct ML call
           ▼                                          ▼
┌─────────────────────┐              ┌──────────────────────────┐
│   FastAPI Backend   │              │   AI/ML Pipeline         │
│   Python (Render)   │              │   FastAPI + Whisper      │
│                     │              │   (HuggingFace Spaces)   │
│  • JWT Auth         │              │                          │
│  • Session Storage  │              │  • Whisper Transcription │
│  • PostgreSQL DB    │              │  • openSMILE Acoustics   │
│  • Report Gen       │              │  • spaCy NLP             │
│  • User Management  │              │  • Anomaly Detection     │
└─────────────────────┘              │  • Risk Tier Scoring     │
           │                         └──────────────────────────┘
           ▼
┌─────────────────────┐
│     PostgreSQL      │
│   (Session Data,    │
│   Biomarker History)│
└─────────────────────┘
```

---

## 🤖 AI/ML Pipeline

> **Deployed at:** `https://alamfarzann-cognisafe-ml.hf.space`

The AI pipeline is the heart of CogniSafe. It accepts a raw audio file and returns a complete cognitive health snapshot in under 90 seconds.

### Pipeline Stages

```
Audio File (WebM/WAV/MP3)
        │
        ▼
┌───────────────────┐
│  Stage 1: ffmpeg  │  Convert WebM → 16kHz mono WAV
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stage 2: Whisper │  Speech-to-text + word timestamps + pause detection
│  (base model)     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stage 3:         │  eGeMAPS 88-feature extraction
│  openSMILE        │  → pitch, jitter, shimmer, HNR, speech rate...
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stage 4: NLP     │  semantic coherence (sentence-transformers)
│  spaCy +          │  lexical diversity (MTLD)
│  MiniLM-L6-v2     │  idea density + syntactic complexity
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Stage 5: Anomaly │  2-sigma deviation vs personal baseline
│  Detection        │  SQLite longitudinal tracker
│                   │  Green / Yellow / Orange / Red risk tier
└───────────────────┘
        │
        ▼
    JSON Response
  (14 biomarkers + risk tier + anomaly flags + confidence intervals)
```

### The 14 Biomarkers

| # | Biomarker | Type | What it measures |
|---|---|---|---|
| 1 | `speech_rate` | Acoustic | Words per minute |
| 2 | `articulation_rate` | Acoustic | Words per minute excluding pauses |
| 3 | `pause_frequency` | Acoustic | Pauses per minute |
| 4 | `pause_duration_mean` | Acoustic | Average pause length (seconds) |
| 5 | `filled_pause_rate` | Acoustic | Uh/um frequency per minute |
| 6 | `pitch_mean` | Acoustic | Average fundamental frequency |
| 7 | `pitch_range` | Acoustic | Pitch variability |
| 8 | `jitter` | Acoustic | Cycle-to-cycle pitch variation |
| 9 | `shimmer` | Acoustic | Amplitude variation |
| 10 | `HNR` | Acoustic | Harmonics-to-noise ratio |
| 11 | `lexical_diversity` | NLP | MTLD vocabulary richness score |
| 12 | `semantic_coherence` | NLP | Sentence-to-sentence logical flow |
| 13 | `idea_density` | NLP | Propositions per word |
| 14 | `syntactic_complexity` | NLP | Average parse tree depth |

### Tech Stack — ML

```
Python 3.11          FastAPI + Uvicorn
OpenAI Whisper       openSMILE (eGeMAPS)
spaCy                sentence-transformers (MiniLM-L6-v2)
scikit-learn         librosa + soundfile
SQLite               ffmpeg
HuggingFace Spaces   Docker
```

---

## 🖥️ Frontend

> **Live at:** `https://cogni-safe.vercel.app`

Built with React + Vite, the frontend is designed to be **demo-first** — every screen is optimized for clarity, emotional impact, and judge impressiveness.

### Pages

| Page | Route | Description |
|---|---|---|
| 🔐 Auth | `/auth` | Login / Register with JWT |
| 📊 Dashboard | `/dashboard` | 14 biomarker cards + risk tier + trends |
| 🎙️ Session | `/session` | Voice recording with live orb animation |
| 🧬 Brain | `/brain` | 3D Semantic Drift Sphere (Three.js) |
| 📄 AR Report | `/ar-report` | AI-generated cognitive health report |

### Tech Stack — Frontend

```
React 18             Vite
React Router v6      TailwindCSS
Three.js             Framer Motion
Recharts             Web Audio API
MediaRecorder API    Vercel
```

---

## ⚙️ Backend

> Built with FastAPI (Python), deployed on Render.

Handles authentication, session persistence, trend APIs, and report generation.

### Key Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth/register` | Register new user |
| `POST` | `/api/auth/login` | Login + JWT token |
| `GET` | `/api/sessions/today` | Check if user recorded today |
| `POST` | `/api/sessions` | Save ML results to DB |
| `GET` | `/api/sessions/history` | Get all past sessions |
| `GET` | `/api/sessions/trends` | Get biomarker trend data |
| `GET` | `/api/reports/generate` | Generate AI health report |

### Tech Stack — Backend

```
Python 3.11          FastAPI
PostgreSQL           SQLAlchemy
JWT Auth             bcrypt
Render               Docker
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- ffmpeg installed
- PostgreSQL running

### 1. Clone the repo

```bash
git clone https://github.com/WTC-Group-3/wtc-round-2-group-3-faiv.git
cd CogniSafe
```

### 2. Start the ML Pipeline

```bash
cd cognisafe-deploy
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd api
python main.py
# Runs on http://localhost:7860
```

### 3. Start the Backend

```bash
cd cognisafe-backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# Runs on http://localhost:8000
```

### 4. Start the Frontend

```bash
cd cognisafe-frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

### 5. One-command startup (Windows)

```bash
start_all.bat
```

This launches all three services in separate terminal windows automatically.

### Environment Variables

Create a `.env` file in `cognisafe-backend/`:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/cognisafe
SECRET_KEY=your-jwt-secret-key
ML_SERVICE_URL=https://alamfarzann-cognisafe-ml.hf.space
```

Create a `.env` file in `cognisafe-frontend/`:

```env
VITE_API_URL=http://localhost:8000
```

---

## 📡 API Reference

### ML Pipeline — `POST /analyze`

**URL:** `https://alamfarzann-cognisafe-ml.hf.space/analyze`

**Request:**
```
Content-Type: multipart/form-data
audio    : <audio file> (WAV, MP3, M4A, WebM, OGG)
user_id  : string
```

**Response:**
```json
{
  "session_id": "uuid",
  "user_id": "demo_user",
  "timestamp": "2026-03-29T00:00:00",
  "processing_time_seconds": 72.4,
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
  "risk_tier": "Yellow",
  "confidence_intervals": {
    "speech_rate": {
      "mean": 146.99,
      "std": 2.1,
      "lower_95": 142.8,
      "upper_95": 151.1
    }
  }
}
```

### ML Pipeline — `GET /health`

```json
{ "status": "ok", "service": "CogniSafe AI Pipeline", "timestamp": "..." }
```

### ML Pipeline — `POST /compare`

Accepts two session JSON objects, returns a biomarker diff showing direction and magnitude of change between sessions.

---

## 👥 Team

> **Team FAIV** — Built at Watch The Code Hackathon 2026

| Member | Role | Responsibilities |
|---|---|---|
| **Farjan Alam** | 🤖 AI/ML Engineer | Voice biomarker pipeline, Whisper, openSMILE, NLP, anomaly detection, FastAPI, HuggingFace deployment |
| **Vansh Singh** | ⚙️ Backend Engineer | FastAPI backend, PostgreSQL, JWT auth, session APIs, report generation |
| **Ishika Rawat** | 🎨 Frontend Engineer | React UI, Three.js 3D sphere, voice orb, dashboard, caregiver view |
| **Akansha Parley** | 📋 Product & Demo Lead | Demo scripting, integration QA, business docs, presentation |

---

## 🗺️ Roadmap

```
2026 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━►

 Q1-Q2          Q3              Q4           2027        2028-2030
   │             │               │             │              │
   ▼             ▼               ▼             ▼              ▼
[MVP Launch] [iOS/Android]  [Insurance   [FDA Breakthrough [Pharma
B2C beta     React Native   B2B2C        Device            Clinical
launch       mobile app     partnerships  Designation]     Trial API]
```

| Phase | Timeline | Milestone |
|---|---|---|
| 🚀 **MVP** | 2026 Q1 | B2C launch, web platform |
| 📱 **Mobile** | 2026 Q3 | React Native iOS + Android app |
| 🏥 **Insurance** | 2026 Q4 | B2B2C insurance partnerships |
| 📋 **FDA** | 2027 | Breakthrough Device Designation pathway |
| 💊 **Pharma** | 2028-2030 | Clinical trial API licensing |

---

## 🧪 Why Voice?

Research shows that **speech and language changes precede clinical dementia diagnosis by 5-10 years.**

Key findings that inspired CogniSafe:

- 📉 **Lexical diversity** declines years before MCI diagnosis
- ⏸️ **Pause patterns** change with cognitive load
- 🔄 **Semantic coherence** drops as working memory degrades
- 🎵 **Pitch variability** correlates with neurological health

> *"The voice is a window into the brain. We just needed the right lens."*

---

## 🏆 Competitive Landscape

| Solution | Cost | Invasiveness | Frequency | Longitudinal |
|---|---|---|---|---|
| 🧲 MRI Scan | $1,000–$3,000 | High | Yearly | ❌ |
| 🧩 Neuropsych Tests | $500–$2,000 | Medium | Yearly | ❌ |
| ⌚ Wearables | $200–$500 | Low | Continuous | ✅ |
| 📱 Mood Apps | Free–$20/mo | None | Daily | Partial |
| 🧠 **CogniSafe** | **$15/mo** | **None** | **Daily** | **✅** |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 💙 by Team SyntaxSaviour**

*"We are not building an app.*
*We are building the world's largest longitudinal cognitive dataset —*
*and using it to give families the one thing they desperately want: time."*

<br/>

[![CogniSafe](https://img.shields.io/badge/CogniSafe-Hear%20the%20Difference-E8A020?style=for-the-badge)](https://cogni-safe.vercel.app)

</div>
