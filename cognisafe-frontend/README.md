<div align="center">

<img src="https://img.shields.io/badge/CogniSafe-Frontend-0A1628?style=for-the-badge&logoColor=E8A020" />

# 🎨 CogniSafe Frontend
### *Everything the judges see, touch, and feel.*

[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=for-the-badge&logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=for-the-badge&logo=vite)](https://vitejs.dev)
[![Vercel](https://img.shields.io/badge/Deployed-Vercel-000000?style=for-the-badge&logo=vercel)](https://cogni-safe.vercel.app)
[![Live](https://img.shields.io/badge/🌐%20Live-cogni--safe.vercel.app-E8A020?style=for-the-badge)](https://cogni-safe.vercel.app)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Pages & Features](#-pages--features)
- [Design System](#-design-system)
- [Services Layer](#-services-layer)
- [Authentication Flow](#-authentication-flow)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Routing](#-routing)
- [ML Integration](#-ml-integration)

---

## 🧠 Overview

The CogniSafe frontend is a **React 18 + Vite** single-page application deployed on Vercel. It is the face of the entire platform — every screen has been designed to be **demo-first**, with pixel-perfect visuals and smooth interactions that make the product come alive for judges and users alike.

Key design principles:
- **Demo screens first** — the Session orb, Brain map, and Dashboard are built for stage impact
- **Dark/light mode** — fully themed with CSS variables, toggled per page and persisted in localStorage
- **No third-party UI library** — all components are custom-built with raw CSS for full control
- **Canvas-heavy** — brain visualization, sparklines, QR codes, and auth background are all drawn with the HTML5 Canvas API

---

## 📁 Project Structure

```
cognisafe-frontend/
│
├── index.html                        # Vite entry point
├── vite.config.js                    # Vite + React plugin config
├── vercel.json                       # SPA rewrite rules for Vercel
├── package.json                      # Dependencies
│
└── src/
    ├── main.jsx                      # ReactDOM render root
    ├── App.jsx                       # Router + AuthProvider wrapper
    │
    ├── context/
    │   └── AuthContext.jsx           # Global auth state (token, user, logout)
    │
    ├── components/
    │   └── common/
    │       ├── Navbar.jsx            # Top navigation bar
    │       └── ProtectedRoute.jsx    # JWT guard — redirects to /auth if not logged in
    │
    ├── pages/
    │   ├── Auth.jsx                  # Login / Register page
    │   ├── Dashboard.jsx             # 14 biomarker cards + calendar + weekly report
    │   ├── Session.jsx               # Voice recording + ML analysis flow
    │   ├── Brain.jsx                 # Brain region map + cognitive trajectory chart
    │   └── ARReport.jsx              # Weekly PDF report generator
    │
    ├── services/
    │   ├── authService.js            # loginUser, registerUser API calls
    │   ├── sessionService.js         # analyzeAudio (HF), saveSession, checkToday
    │   └── dashboardService.js       # getProfile, getSessionHistory, getWeeklyReport, getTrajectory
    │
    └── styles/
        ├── global.css                # CSS variables, resets, shared utilities
        ├── auth.css                  # Auth page styles
        ├── session.css               # Session recording page styles
        ├── dashboard.css             # Dashboard styles
        ├── brain.css                 # Brain map page styles
        └── arreport.css              # AR Report page styles
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| **React** | 18.3 | UI framework |
| **Vite** | 5.4 | Build tool + dev server |
| **React Router DOM** | 6.26 | Client-side routing |
| **jsPDF** | 4.2 | PDF report generation in-browser |
| **Web Audio API** | Native | Live audio amplitude visualization |
| **MediaRecorder API** | Native | Browser audio recording (WebM/Opus) |
| **Speech Recognition API** | Native | Live transcript display during recording |
| **Canvas 2D API** | Native | Brain map, sparklines, QR code, neural background |
| **Vercel** | — | Hosting + SPA rewrites |

> ⚡ No external UI component library — all components are custom built for full design control.

---

## 🖥️ Pages & Features

### 🔐 `/auth` — Auth Page

The entry point for all users. Features a **live animated neural network** drawn on HTML Canvas as the background — nodes float and connect with glowing green edges in real time.

**Features:**
- Toggle between **Login** and **Register** tabs
- Password strength bar (color-coded: red → amber → green)
- Real-time form validation
- JWT token stored in `AuthContext` on successful login/register
- Demo credentials shown for easy judge access: `demo@cognisafe.app / demo1234`
- Social login UI (Google, GitHub, LinkedIn) — visual only
- Fully responsive

---

### 📊 `/dashboard` — Main Dashboard

The primary screen for daily cognitive health monitoring. Designed to give a complete picture at a glance.

**Features:**

#### 14 Biomarker Cards
- One card per biomarker in a responsive grid
- Each card shows: current value, trend percentage, direction arrow (↑ ↓ →), status dot (green/amber/red)
- **Sparkline chart** drawn on Canvas — 14-point trend line per biomarker
- Status colors: `good` (green), `warn` (amber), `crit` (red)

#### Session Calendar
- Full month calendar with navigation (← →)
- Each day with a session shows a colored dot: green (good), amber (watch), red (alert)
- Click any day to see that session's biomarker snapshot in a bottom panel
- Pulls real data from `GET /api/sessions/history`

#### Weekly Report Card
- AI-generated narrative from the backend report endpoint
- 3 insight pills with color-coded indicators
- Session count and average biomarker values for the week

#### Risk Tier Badge
- Prominent `Green / Yellow / Orange / Red` badge in the header
- Derived from the most recent session's risk tier

#### Dark / Light Mode
- Toggle button in the top right
- Persisted in `localStorage` as `cog_dark`

---

### 🎙️ `/session` — Voice Recording Session

The most important screen. Judges interact with this live on stage.

**States the page moves through:**

```
checking → idle → recording → processing → result (donegood / donewarn / donebad)
```

| State | What the user sees |
|---|---|
| `checking` | Spinner while checking if already recorded today |
| `idle` | Prompt + image + "Start Recording" button |
| `recording` | Live countdown timer (3:00 → 0:00) + progress ring + live transcript |
| `processing` | Pulsing animation while ML pipeline runs |
| `donegood` | Green card — "Excellent session!" |
| `donewarn` | Amber card — "Pause frequency elevated" |
| `donebad` | Red card — "Changes detected, try again tomorrow" |

**Key technical features:**

- **5 randomized clinical prompts** — picture description tasks (mountain lake, city, forest) + memory recall + navigation tasks, matching standard clinical assessment protocols
- **WebRTC audio capture** — `navigator.mediaDevices.getUserMedia({ audio: true })`
- **MediaRecorder** — records in `audio/webm;codecs=opus` (best quality) with 250ms chunks
- **Live transcript** — uses `window.SpeechRecognition` to show words appearing in real time during recording
- **Countdown ring** — SVG circle with `stroke-dashoffset` animation, color shifts gold → red in the last 45 seconds
- **HF Space warmup** — pings `GET /health` on page mount to wake the HuggingFace Space before the user finishes recording
- **AbortController** — 480 second timeout on the ML fetch with clean cancellation
- **Auto-stop** — recording stops automatically when timer hits 0

**After recording:**
1. Audio blob sent directly to `POST https://alamfarzann-cognisafe-ml.hf.space/analyze`
2. ML result normalized via `normalizeAIResult()`
3. Result saved to backend via `POST /api/sessions`
4. Risk tier result card shown to user

---

### 🧬 `/brain` — Brain Region Map

The visually striking screen that makes judges take out their phones.

**Features:**

#### Interactive Brain Canvas
- **Animated brain outline** drawn on Canvas — glowing ellipse with subtle pulse
- **6 brain regions** mapped to biomarkers and plotted as interactive nodes:
  - Prefrontal Cortex → `semantic_coherence`, `idea_density`, `syntactic_complexity`
  - Temporal Lobe → `lexical_diversity`
  - Parietal Lobe → `semantic_coherence`
  - Broca's Area → `speech_rate`, `pause_frequency`
  - Wernicke's Area → `lexical_diversity`
  - Cerebellum → `pause_duration`, `pitch_mean`, `articulation_rate`
- Nodes colored by status: green (ok), amber (warn), red (bad)
- **Hover** to see region description tooltip
- **Click** to select and highlight a region

#### Cognitive Trajectory Chart
- Line chart showing monthly cognitive pulse score (0–100) over 6 months
- Pulls real data from `GET /api/reports/trajectory`
- Falls back to static demo data if API unavailable

#### Biomarker Anomaly List
- 6 key biomarkers listed with current value and status badge
- Mapped from the latest session result

---

### 📄 `/ar-report` — Weekly Report & PDF Export

The caregiver-friendly report view.

**Features:**

- **Weekly narrative** — plain-language AI summary from `GET /api/reports/weekly`
- **3 insight pills** — color-coded (success/warn/indigo)
- **Risk tier banner** — large, prominent, plain-language status
- **Biomarker summary table** — key metrics with status indicators
- **Session streak counter**
- **QR code** — drawn on Canvas, decorative (links to platform)
- **PDF Export** — uses `jsPDF` to generate a professional A4 report entirely in-browser:
  - Dark green header with CogniSafe branding
  - Risk tier badge
  - Biomarker summary table
  - Weekly narrative
  - Page footer with timestamp

---

## 🎨 Design System

### Color Tokens

```css
/* Primary palette */
--navy:      #0A1628    /* primary background */
--amber:     #E8A020    /* accent / highlight */
--off-white: #F4F6FA    /* light background */

/* Status colors */
--good:  #10B981   /* green  — healthy */
--warn:  #F59E0B   /* amber  — watch */
--crit:  #EF4444   /* red    — alert */

/* Dark mode variants */
--good-dark: #6EE7B7
--warn-dark: #FDE68A
--crit-dark: #FCA5A5
```

### Status System

Every piece of data in the UI maps to one of three statuses:

| Status | Color | Meaning |
|---|---|---|
| `good` | 🟢 Green | Within healthy range |
| `warn` | 🟡 Amber | Slightly elevated — monitor |
| `bad` / `crit` | 🔴 Red | Anomaly detected — attention needed |

Risk tier from the ML pipeline maps as:
```javascript
{ "Green": "good", "Yellow": "warn", "Red": "bad" }
```

### Typography

Pure CSS — no external fonts loaded. Uses system font stack for performance and reliability.

### Dark Mode

Each page has its own dark/light CSS variable set. Toggled via `localStorage.setItem("cog_dark", "true")` and read on mount.

---

## 🔌 Services Layer

### `authService.js`

```javascript
loginUser(email, password)     → { access_token, user_id, name, email }
registerUser(name, email, password, dob) → { access_token, user_id, name, email }
```

### `sessionService.js`

```javascript
checkToday(token)              → { recorded, risk_tier, session_id }
analyzeAudio(audioBlob, userId)→ normalizedAIResult
saveSession(token, aiResult)   → saved session object
normalizeAIResult(raw)         → { risk_tier, biomarkers, anomaly_flags, ... }
```

**`analyzeAudio` flow:**
1. Creates `FormData` with `recording.webm` file and `user_id`
2. Calls `POST https://alamfarzann-cognisafe-ml.hf.space/analyze` directly
3. 480 second timeout via `AbortController`
4. Normalizes the ML response to a consistent internal shape

### `dashboardService.js`

```javascript
getProfile(token)              → user profile
getSessionHistory(token, months) → [ HistoryItem, ... ]
getLatestSession(token)        → SessionResponse
getWeeklyReport(token)         → WeeklyReport
getTrajectory(token, months)   → [ TrajectoryPoint, ... ]
```

---

## 🔐 Authentication Flow

```
User opens app
      │
      ▼
Redirected to /auth
      │
      ├── Login / Register
      │         │
      │         ▼
      │   POST /api/auth/login or /register
      │         │
      │         ▼
      │   JWT token stored in AuthContext (React state)
      │
      ▼
Protected routes check AuthContext
      │
      ├── Token present → render page
      └── Token missing → redirect to /auth
```

`AuthContext` provides:
```javascript
{ token, user, login(token, user), logout() }
```

`ProtectedRoute` wraps all authenticated pages:
```jsx
<ProtectedRoute>
  <Dashboard />
</ProtectedRoute>
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
cd cognisafe-frontend
npm install
```

### 2. Create environment file

```bash
cp .env.example .env
```

### 3. Start dev server

```bash
npm run dev
# Runs on http://localhost:5173
```

### 4. Build for production

```bash
npm run build
```

### 5. Preview production build

```bash
npm run preview
```

---

## 🌿 Environment Variables

Create `.env` in `cognisafe-frontend/`:

```env
VITE_API_URL=http://localhost:8000
```

For production (set in Vercel dashboard):
```env
VITE_API_URL=https://your-backend-url.onrender.com
```

> The ML pipeline URL (`https://alamfarzann-cognisafe-ml.hf.space`) is hardcoded in `sessionService.js` since it's a public endpoint.

---

## 🗺️ Routing

| Route | Component | Protected | Description |
|---|---|---|---|
| `/` | — | — | Redirects to `/auth` |
| `/auth` | `Auth.jsx` | ❌ | Login / Register |
| `/dashboard` | `Dashboard.jsx` | ✅ | Main biomarker dashboard |
| `/session` | `Session.jsx` | ✅ | Voice recording session |
| `/brain` | `Brain.jsx` | ✅ | Brain region map + trajectory |
| `/ar-report` | `ARReport.jsx` | ✅ | Weekly report + PDF export |

### Vercel SPA Rewrites

The `vercel.json` file ensures all routes return `index.html` so React Router handles navigation client-side:

```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/" }
  ]
}
```

Without this, hard refreshing on `/dashboard` would return a 404 from Vercel.

---

## 🤖 ML Integration

The frontend calls the ML pipeline **directly** — bypassing the backend to avoid Render's 30-second timeout limit.

```
User records audio (WebM/Opus)
          │
          ▼
POST https://alamfarzann-cognisafe-ml.hf.space/analyze
  body: FormData { audio: recording.webm, user_id: "123" }
  timeout: 480 seconds
          │
          ▼
ML returns 14 biomarkers + risk tier + anomaly flags
          │
          ▼
normalizeAIResult() maps ML field names → internal field names
  (e.g. HNR → hnr, pause_duration_mean → pause_duration)
          │
          ▼
POST http://localhost:8000/api/sessions
  saves result to PostgreSQL via backend
          │
          ▼
Result card shown to user (Green / Yellow / Red)
```

### HuggingFace Cold Start Handling

HF Spaces on the free tier sleep after inactivity. Two strategies used:

1. **App-level warmup** — `App.jsx` pings `/health` on mount
2. **Session-level warmup** — `Session.jsx` pings `/health` when the recording page opens

This gives the Space 1-3 minutes to wake up while the user reads the prompt and prepares to record.

---

<div align="center">

**CogniSafe Frontend** — Built by **Ishika Rawat** · Team FAIV 🎨

*Part of the CogniSafe cognitive health monitoring platform.*

[![ML Pipeline](https://img.shields.io/badge/ML%20Pipeline-HuggingFace-FF9D00?style=for-the-badge)](https://alamfarzann-cognisafe-ml.hf.space)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge)](https://github.com/SyntaxSaviour/CogniSafe)

</div>
