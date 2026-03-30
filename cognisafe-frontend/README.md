<div align="center">

<img src="https://img.shields.io/badge/CogniSafe-Frontend-0A1628?style=for-the-badge&logoColor=E8A020" />

# 🎨 CogniSafe Frontend
### *Everything the judges see, touch, and feel.*

[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF?style=flat-square&logo=vite)](https://vitejs.dev)
[![Vercel](https://img.shields.io/badge/Deployed-Vercel-000?style=flat-square&logo=vercel)](https://cogni-safe.vercel.app)
[![Live](https://img.shields.io/badge/🌐_Live-cogni--safe.vercel.app-E8A020?style=flat-square)](https://cogni-safe.vercel.app)
[![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)](../LICENSE)

<br/>

> A **React 18 + Vite** SPA — zero third-party UI libraries, Canvas-heavy, built demo-first.
> Every screen was designed for stage impact: judges pick it up, tap it, and remember it.

<br/>

```
/auth  ──►  /dashboard  ──►  /session  ──►  /brain  ──►  /ar-report
 🔐            📊               🎙️             🧬              📄
Login      Biomarkers       Record Voice    Brain Map      PDF Export
```

</div>

---

## 📖 Contents

[Overview](#-overview) · [Structure](#-project-structure) · [Tech Stack](#-tech-stack) · [Pages](#-pages--features) · [Design System](#-design-system) · [Services](#-services-layer) · [Auth Flow](#-authentication-flow) · [Getting Started](#-getting-started) · [Routing](#-routing) · [ML Integration](#-ml-integration)

---

## 🧠 Overview

| Principle | Detail |
|---|---|
| 🎭 **Demo-first** | Session orb, Brain map, and Dashboard are built for stage impact |
| 🌗 **Dark/Light mode** | CSS variables per page, persisted in `localStorage` |
| 🎨 **No UI lib** | Every component custom-built in raw CSS for full design control |
| 🖼️ **Canvas-heavy** | Brain viz, sparklines, QR codes, auth background — all HTML5 Canvas |

---

## 📁 Project Structure

```
cognisafe-frontend/
├── index.html                   # Vite entry point
├── vite.config.js               # Vite + React config
├── vercel.json                  # SPA rewrite rules
│
└── src/
    ├── App.jsx                  # Router + AuthProvider
    ├── context/
    │   └── AuthContext.jsx      # Global token/user state
    ├── components/common/
    │   ├── Navbar.jsx
    │   └── ProtectedRoute.jsx   # JWT guard → /auth redirect
    ├── pages/
    │   ├── Auth.jsx             # Login / Register
    │   ├── Dashboard.jsx        # 14 biomarker cards + calendar
    │   ├── Session.jsx          # Voice recording + ML flow
    │   ├── Brain.jsx            # Brain region map + trajectory
    │   └── ARReport.jsx         # Weekly PDF report
    ├── services/
    │   ├── authService.js
    │   ├── sessionService.js
    │   └── dashboardService.js
    └── styles/
        ├── global.css           # Variables, resets, utilities
        ├── auth.css
        ├── session.css
        ├── dashboard.css
        ├── brain.css
        └── arreport.css
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| React | 18.3 | UI framework |
| Vite | 5.4 | Build tool + dev server |
| React Router DOM | 6.26 | Client-side routing |
| jsPDF | 4.2 | In-browser PDF generation |
| Web Audio API | Native | Live amplitude visualization |
| MediaRecorder API | Native | WebM/Opus audio capture |
| Speech Recognition API | Native | Live transcript during recording |
| Canvas 2D API | Native | Brain map, sparklines, QR, neural BG |
| Vercel | — | Hosting + SPA rewrites |

---

## 🖥️ Pages & Features

### 🔐 `/auth` — Login & Register

Live **animated neural network** on Canvas — glowing nodes float and connect in real time.

| Feature | Detail |
|---|---|
| Tabs | Toggle between Login / Register |
| Password strength | Color bar: red → amber → green |
| Demo access | `demo@cognisafe.app` / `demo1234` shown for judges |
| Social UI | Google, GitHub, LinkedIn (visual only) |

---

### 📊 `/dashboard` — Biomarker Dashboard

**14 biomarker cards** in a responsive grid. Each card has a live sparkline (Canvas), trend arrow (↑ ↓ →), and status dot. Below the grid: a **session calendar** with colored dots per day, a **weekly AI narrative**, and a **risk tier badge**.

> Dark/Light toggle persisted as `cog_dark` in localStorage.

---

### 🎙️ `/session` — Voice Recording

The live demo screen. Moves through these states:

```
checking → idle → recording → processing → donegood / donewarn / donebad
```

| State | UI |
|---|---|
| `idle` | Clinical prompt + "Start Recording" |
| `recording` | Countdown ring · live transcript · auto-stops at 0:00 |
| `processing` | Pulsing orb animation (~90s) |
| `done*` | Colored result card (Green / Amber / Red) |

**Key tech:** WebRTC mic capture → MediaRecorder (Opus) → direct HF Space POST → `normalizeAIResult()` → backend save.

---

### 🧬 `/brain` — Brain Region Map

Interactive Canvas brain with **6 clickable region nodes**, each mapped to biomarkers:

| Region | Biomarkers |
|---|---|
| Prefrontal Cortex | `semantic_coherence`, `idea_density`, `syntactic_complexity` |
| Broca's Area | `speech_rate`, `pause_frequency` |
| Wernicke's Area | `lexical_diversity` |
| Temporal Lobe | `lexical_diversity` |
| Parietal Lobe | `semantic_coherence` |
| Cerebellum | `pause_duration`, `pitch_mean`, `articulation_rate` |

Nodes colored green/amber/red by status. Hover = tooltip. Click = highlight + detail panel. Includes a **6-month cognitive trajectory line chart**.

---

### 📄 `/ar-report` — Weekly Report & PDF

AI-generated weekly narrative, risk tier banner, biomarker summary table, session streak, and decorative QR code. One-click **PDF export** via jsPDF — full A4 with branding, badge, table, and footer timestamp.

---

## 🎨 Design System

### Colors

```css
--navy:      #0A1628   /* backgrounds      */
--amber:     #E8A020   /* accent / CTA     */
--off-white: #F4F6FA   /* light surfaces   */
--good:      #10B981   /* healthy          */
--warn:      #F59E0B   /* watch            */
--crit:      #EF4444   /* alert            */
```

### Status Mapping

```
ML risk tier  →  UI status
──────────────────────────
"Green"       →  good   🟢
"Yellow"      →  warn   🟡
"Red"         →  bad    🔴
```

> System font stack throughout — no external fonts, maximum reliability.

---

## 🔌 Services Layer

### `authService.js`
```js
loginUser(email, password)
registerUser(name, email, password, dob)
// → { access_token, user_id, name, email }
```

### `sessionService.js`
```js
checkToday(token)               // → { recorded, risk_tier, session_id }
analyzeAudio(audioBlob, userId) // → normalizedAIResult  (480s timeout)
saveSession(token, aiResult)    // → saved session
normalizeAIResult(raw)          // → { risk_tier, biomarkers, anomaly_flags }
```

### `dashboardService.js`
```js
getProfile(token)
getSessionHistory(token, months)  // → [ HistoryItem ]
getLatestSession(token)
getWeeklyReport(token)
getTrajectory(token, months)      // → [ TrajectoryPoint ]
```

---

## 🔐 Authentication Flow

```
Open app → /auth → Login/Register
                       │
                  POST /api/auth/login
                       │
              JWT stored in AuthContext
                       │
          ┌────────────┴────────────┐
      Token present             Token missing
          │                         │
     Render page              Redirect → /auth
```

`ProtectedRoute` wraps every authenticated page:
```jsx
<ProtectedRoute><Dashboard /></ProtectedRoute>
```

---

## 🚀 Getting Started

```bash
# 1. Install
cd cognisafe-frontend && npm install

# 2. Environment
cp .env.example .env          # set VITE_API_URL

# 3. Dev
npm run dev                   # → http://localhost:5173

# 4. Production build
npm run build && npm run preview
```

### Environment Variables

```env
# .env (local)
VITE_API_URL=http://localhost:8000

# Vercel dashboard (production)
VITE_API_URL=https://your-backend.onrender.com
```

> ML URL (`https://alamfarzann-cognisafe-ml.hf.space`) is hardcoded in `sessionService.js` — it's a public endpoint.

---

## 🗺️ Routing

| Route | Component | Auth | Description |
|---|---|---|---|
| `/` | — | — | Redirect → `/auth` |
| `/auth` | `Auth.jsx` | ❌ | Login / Register |
| `/dashboard` | `Dashboard.jsx` | ✅ | Biomarker overview |
| `/session` | `Session.jsx` | ✅ | Voice recording |
| `/brain` | `Brain.jsx` | ✅ | Brain map + trajectory |
| `/ar-report` | `ARReport.jsx` | ✅ | Report + PDF export |

`vercel.json` rewrites all routes → `index.html` (prevents 404 on hard refresh):
```json
{ "rewrites": [{ "source": "/(.*)", "destination": "/" }] }
```

---

## 🤖 ML Integration

```
MediaRecorder (WebM/Opus)
        │
        ▼  POST /analyze  (FormData: audio + user_id)
HuggingFace Space                    timeout: 480s
        │
        ▼
14 biomarkers + risk tier + anomaly flags
        │
        ▼  normalizeAIResult()
POST /api/sessions  →  PostgreSQL
        │
        ▼
Result card  (🟢 Green / 🟡 Yellow / 🔴 Red)
```

**Cold start strategy** — HF free tier sleeps after inactivity:

| Warmup | Where | Effect |
|---|---|---|
| App-level | `App.jsx` pings `/health` on mount | Wakes Space on every visit |
| Session-level | `Session.jsx` pings `/health` on page open | 1–3 min warmup while user reads prompt |

---

<div align="center">

Built by **Ishika Rawat** · Team FAIV 🎨

[![ML Pipeline](https://img.shields.io/badge/ML_Pipeline-HuggingFace-FF9D00?style=flat-square)](https://alamfarzann-cognisafe-ml.hf.space)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square)](https://github.com/WTC-Group-3/wtc-round-2-group-3-faiv)
[![Root README](https://img.shields.io/badge/Root-README-0A1628?style=flat-square)](../README.md)

</div>
