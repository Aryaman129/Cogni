<div align="center">

<img src="https://img.shields.io/badge/CogniSafe-Backend%20API-0A1628?style=for-the-badge&logoColor=E8A020" />

# ⚙️ CogniSafe Backend
### *The nervous system of cognitive health monitoring.*

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Latest-4169E1?style=for-the-badge&logo=postgresql)](https://postgresql.org)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-D71F00?style=for-the-badge)](https://sqlalchemy.org)
[![JWT](https://img.shields.io/badge/Auth-JWT%20Bearer-000000?style=for-the-badge&logo=jsonwebtokens)](https://jwt.io)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Database Schema](#-database-schema)
- [API Reference](#-api-reference)
- [Authentication](#-authentication)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Demo Data Seeding](#-demo-data-seeding)
- [ML Pipeline Integration](#-ml-pipeline-integration)

---

## 🧠 Overview

The CogniSafe backend is a **FastAPI REST API** that acts as the connective layer between the React frontend and the AI/ML pipeline. It handles:

- 🔐 **User authentication** — JWT-based register/login
- 💾 **Session persistence** — stores all 14 biomarkers per session in PostgreSQL
- 📊 **Trend & history APIs** — powers the dashboard sparklines and calendar heatmap
- 📄 **Weekly report generation** — computes narrative insights from biomarker averages
- 📈 **Trajectory scoring** — monthly cognitive pulse score over time
- 🤖 **ML proxy** — optional proxy route to the HuggingFace ML service

---

## 📁 Project Structure

```
cognisafe-backend/
│
├── main.py                  # FastAPI app entry point, CORS, startup hooks
├── database.py              # SQLAlchemy engine + session factory
├── auth.py                  # Password hashing, JWT creation + decoding
├── dependencies.py          # get_current_user dependency injection
├── seed.py                  # Demo data seeder (6 months of sessions)
├── requirements.txt         # Python dependencies
│
├── models/
│   ├── __init__.py          # Exports User, Session
│   ├── user.py              # User SQLAlchemy model
│   └── session.py           # Session SQLAlchemy model (14 biomarkers)
│
├── schemas/
│   ├── __init__.py          # Exports all Pydantic schemas
│   ├── auth.py              # RegisterRequest, LoginRequest, TokenResponse
│   ├── session.py           # SessionCreate, SessionResponse, HistoryItem, TodayResponse
│   └── user.py              # UserResponse
│
└── routes/
    ├── __init__.py          # Exports all routers
    ├── auth.py              # POST /api/auth/register, /api/auth/login
    ├── sessions.py          # POST/GET /api/sessions/*
    ├── reports.py           # GET /api/reports/weekly, /api/reports/trajectory
    ├── users.py             # GET /api/users/me
    └── ml.py                # POST /api/ml/analyze, GET /api/ml/warmup
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **FastAPI** | Web framework — async, fast, auto-docs |
| **SQLAlchemy** | ORM for PostgreSQL / SQLite |
| **PostgreSQL** | Production database |
| **SQLite** | Local development fallback (zero config) |
| **python-jose** | JWT token creation and decoding |
| **passlib + bcrypt** | Password hashing |
| **python-dotenv** | Environment variable management |
| **httpx** | Async HTTP client for ML proxy calls |
| **Pydantic v2** | Request/response schema validation |
| **Uvicorn** | ASGI server |

---

## 🗄️ Database Schema

### `users` table

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment user ID |
| `name` | String | Full name |
| `email` | String (unique) | Login email |
| `password_hash` | String | bcrypt hashed password |
| `dob` | String | Date of birth (YYYY-MM-DD) |
| `created_at` | DateTime | Account creation timestamp |

### `sessions` table

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment session ID |
| `user_id` | FK → users.id | Owner of this session |
| `risk_tier` | String | Green / Yellow / Orange / Red |
| `recorded_at` | DateTime | When the session was recorded |
| `semantic_coherence` | Float | Sentence-to-sentence logical flow (0-1) |
| `lexical_diversity` | Float | MTLD vocabulary richness score |
| `idea_density` | Float | Propositions per word (0-1) |
| `speech_rate` | Float | Words per minute |
| `pause_frequency` | Float | Pauses per minute |
| `pause_duration` | Float | Mean pause duration (seconds) |
| `pitch_mean` | Float | Average fundamental frequency |
| `pitch_range` | Float | Pitch variability |
| `jitter` | Float | Cycle-to-cycle pitch variation |
| `shimmer` | Float | Amplitude variation |
| `hnr` | Float | Harmonics-to-noise ratio |
| `syntactic_complexity` | Float | Average parse tree depth |
| `articulation_rate` | Float | Words per minute excluding pauses |
| `emotional_entropy` | Float | Emotional variability index |
| `has_anomaly` | Boolean | True if any biomarker flagged |
| `anomaly_flags` | String | JSON string of anomaly flag objects |

### Entity Relationship

```
┌─────────────┐         ┌──────────────────────┐
│    users    │         │       sessions        │
│─────────────│         │──────────────────────│
│ id (PK)     │◄────────│ user_id (FK)          │
│ name        │  1  : N │ id (PK)               │
│ email       │         │ risk_tier             │
│ password    │         │ recorded_at           │
│ dob         │         │ [14 biomarker columns]│
│ created_at  │         │ has_anomaly           │
└─────────────┘         │ anomaly_flags         │
                        └──────────────────────┘
```

---

## 📡 API Reference

Base URL: `http://localhost:8000`

### 🔐 Auth Routes — `/api/auth`

---

#### `POST /api/auth/register`

Register a new user account.

**Request Body:**
```json
{
  "name": "Arjun Sharma",
  "email": "arjun@example.com",
  "password": "securepass123",
  "dob": "1968-05-14"
}
```

**Response `200`:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user_id": 1,
  "name": "Arjun Sharma",
  "email": "arjun@example.com"
}
```

**Errors:**
- `400` — Email already registered

---

#### `POST /api/auth/login`

Login with email and password.

**Request Body:**
```json
{
  "email": "arjun@example.com",
  "password": "securepass123"
}
```

**Response `200`:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user_id": 1,
  "name": "Arjun Sharma",
  "email": "arjun@example.com"
}
```

**Errors:**
- `401` — Invalid email or password

---

### 🎙️ Session Routes — `/api/sessions`

> All session routes require `Authorization: Bearer <token>` header.

---

#### `POST /api/sessions`

Save ML analysis results to the database.

**Request Body:**
```json
{
  "risk_tier": "Yellow",
  "semantic_coherence": 0.3447,
  "lexical_diversity": 178.52,
  "idea_density": 0.4201,
  "speech_rate": 146.99,
  "pause_frequency": 26.85,
  "pause_duration": 0.476,
  "pitch_mean": 33.42,
  "pitch_range": 8.06,
  "jitter": 0.0517,
  "shimmer": 1.272,
  "hnr": 5.61,
  "syntactic_complexity": 5.077,
  "articulation_rate": 205.59,
  "emotional_entropy": null,
  "has_anomaly": true,
  "anomaly_flags": "[{\"biomarker\": \"semantic_coherence\", \"severity\": \"mild\"}]"
}
```

**Response `200`:**
```json
{
  "id": 42,
  "risk_tier": "Yellow",
  "recorded_at": "2026-03-29T10:30:00",
  "semantic_coherence": 0.3447,
  ...all 14 biomarkers...
  "has_anomaly": true,
  "anomaly_flags": "[...]"
}
```

---

#### `GET /api/sessions/today`

Check if the current user has already recorded a session today.

**Response `200` (recorded):**
```json
{
  "recorded": true,
  "risk_tier": "Green",
  "session_id": 42
}
```

**Response `200` (not recorded):**
```json
{
  "recorded": false,
  "risk_tier": null,
  "session_id": null
}
```

---

#### `GET /api/sessions/latest`

Get the most recent session for the current user.

**Response `200`:**
```json
{
  "id": 42,
  "risk_tier": "Green",
  "recorded_at": "2026-03-29T10:30:00",
  ...all 14 biomarkers...
}
```

**Errors:**
- `404` — No sessions found

---

#### `GET /api/sessions/history?months=1`

Get session history for the past N months.

**Query Params:**

| Param | Type | Default | Description |
|---|---|---|---|
| `months` | int | `1` | Number of months to look back |

**Response `200`:**
```json
[
  {
    "date": "2026-03-29T10:30:00",
    "status": "good",
    "risk_tier": "Green",
    "session_id": 42,
    "semantic_coherence": 0.3447,
    "speech_rate": 146.99,
    "pause_frequency": 26.85
  }
]
```

> `status` is a frontend-friendly string: `"good"` (Green), `"warn"` (Yellow), `"bad"` (Red)

---

### 📄 Report Routes — `/api/reports`

> All report routes require `Authorization: Bearer <token>` header.

---

#### `GET /api/reports/weekly`

Generate a weekly cognitive health report from the past 7 days of sessions.

**Response `200`:**
```json
{
  "narrative": "This week you completed 5 sessions. Your semantic coherence averaged 0.74, above your baseline. Overall cognitive health is green.",
  "insights": [
    {
      "color": "success",
      "text": "Semantic coherence averaged 0.74 — above your personal baseline of 0.76."
    },
    {
      "color": "success",
      "text": "Pause frequency 3.2/min is within your healthy range."
    },
    {
      "color": "indigo",
      "text": "You recorded 5 out of 7 days. Consistency improves baseline accuracy."
    }
  ],
  "avg_semantic_coherence": 0.74,
  "avg_speech_rate": 142.3,
  "avg_pause_frequency": 3.2,
  "sessions_this_week": 5,
  "risk_tier": "Green"
}
```

**Insight Colors:**

| Color | Meaning |
|---|---|
| `success` | Positive signal — within healthy range |
| `warn` | Mild concern — worth monitoring |
| `indigo` | Informational — consistency metrics |

---

#### `GET /api/reports/trajectory?months=6`

Get monthly cognitive pulse scores for trend visualization.

**Query Params:**

| Param | Type | Default | Description |
|---|---|---|---|
| `months` | int | `6` | Number of months to look back |

**Scoring Formula:**
```
score = (semantic_coherence × 40) +
        (min(speech_rate / 150 × 20, 20)) +
        (max(0, 20 - pause_frequency × 3)) +
        (min(hnr / 25 × 20, 20))
        
Range: 0–100
```

**Response `200`:**
```json
[
  { "month": "Oct", "score": 78.4, "session_count": 18 },
  { "month": "Nov", "score": 76.1, "session_count": 21 },
  { "month": "Dec", "score": 74.8, "session_count": 19 },
  { "month": "Jan", "score": 73.2, "session_count": 20 },
  { "month": "Feb", "score": 71.5, "session_count": 17 },
  { "month": "Mar", "score": 69.8, "session_count": 15 }
]
```

---

### 🤖 ML Proxy Routes — `/api/ml`

Optional proxy routes to the HuggingFace ML service.

---

#### `POST /api/ml/analyze`

Proxy audio to the ML pipeline (alternative to calling HF directly).

**Request:** `multipart/form-data`
- `audio` — audio file
- `user_id` — string

**Response:** Passes through the full ML response JSON (14 biomarkers + risk tier).

**Errors:**
- `504` — ML service timed out (120s timeout)
- `502` — ML service unreachable

---

#### `GET /api/ml/warmup`

Ping the HuggingFace Space to wake it up before a session.

**Response `200` (warmed):**
```json
{
  "status": "warmed",
  "hf": { "status": "ok", "service": "CogniSafe AI Pipeline" }
}
```

**Response `200` (still warming):**
```json
{
  "status": "warming",
  "detail": "Connection timeout"
}
```

---

### 🏥 Health Check

#### `GET /health`

```json
{ "status": "ok" }
```

#### `GET /`

```json
{ "status": "CogniSafe API running", "version": "1.0.0" }
```

---

## 🔐 Authentication

CogniSafe uses **JWT Bearer token** authentication.

### Flow

```
1. POST /api/auth/register  →  returns access_token
2. POST /api/auth/login     →  returns access_token
3. All protected routes require:
   Authorization: Bearer <access_token>
```

### Token Details

| Setting | Value |
|---|---|
| Algorithm | `HS256` |
| Expiry | `10080 minutes` (7 days) |
| Payload | `{ "sub": "<user_id>", "exp": <timestamp> }` |

### Dependency Injection

All protected routes use the `get_current_user` dependency:

```python
def get_current_user(
    token: str       = Depends(oauth2_scheme),
    db:    DBSession = Depends(get_db)
) -> User:
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    return db.query(User).filter(User.id == int(payload["sub"])).first()
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
cd cognisafe-backend
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Run locally (SQLite — zero config)

```bash
uvicorn main:app --reload --port 8000
```

The app auto-creates all tables on startup and creates the demo user `demo@cognisafe.app`.

### 4. Seed demo data

```bash
python seed.py
```

This creates 6 months of realistic sessions for the demo account.

### 5. View auto-generated API docs

```
http://localhost:8000/docs       # Swagger UI
http://localhost:8000/redoc      # ReDoc
```

---

## 🌿 Environment Variables

Create a `.env` file in `cognisafe-backend/`:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cognisafe
# Leave empty to use SQLite locally:
# DATABASE_URL=sqlite:///./cognisafe.db

# JWT
SECRET_KEY=your-super-secret-jwt-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# ML Service
ML_SERVICE_URL=https://alamfarzann-cognisafe-ml.hf.space
```

> ⚠️ **Never commit `.env` to git.** It's already in `.gitignore`.

---

## 🌱 Demo Data Seeding

Run `seed.py` once to populate the database with a realistic demo account:

```bash
python seed.py
```

**What it creates:**

| Field | Value |
|---|---|
| Name | Arjun Sharma |
| Email | `demo@cognisafe.app` |
| Password | `demo1234` |
| DOB | 1968-05-14 |
| Sessions | ~130 sessions over 6 months |

**Session pattern:**
- **Months 1-3:** Stable Green — healthy baseline
- **Month 4:** Gradual Yellow drift
- **Month 5-6:** Mild decline — some Orange sessions

This gives the dashboard a realistic-looking trend that clearly shows cognitive drift over time — perfect for demo day.

---

## 🤖 ML Pipeline Integration

The frontend calls the ML pipeline **directly** (bypassing this backend) to avoid timeout issues on Render's free tier. The flow is:

```
Frontend
   │
   ├─── POST https://alamfarzann-cognisafe-ml.hf.space/analyze
   │         (audio file + user_id)
   │         ↓
   │    ML returns 14 biomarkers + risk tier
   │
   └─── POST http://localhost:8000/api/sessions
             (saves ML results to PostgreSQL)
```

The backend's `/api/ml/analyze` proxy route exists as a fallback but is not used in production to keep latency low.

---

## 📦 Requirements

```
fastapi
uvicorn
sqlalchemy
psycopg2-binary
python-jose[cryptography]
passlib[bcrypt]
python-dotenv
python-multipart
httpx
pydantic
```

---

<div align="center">

**CogniSafe Backend** — Built with FastAPI 🚀

*Part of the CogniSafe cognitive health monitoring platform.*

[![ML Pipeline](https://img.shields.io/badge/ML%20Pipeline-HuggingFace-FF9D00?style=for-the-badge)](https://alamfarzann-cognisafe-ml.hf.space)
[![Frontend](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge)](https://cogni-safe.vercel.app)

</div>
