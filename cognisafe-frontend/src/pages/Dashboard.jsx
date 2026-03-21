import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { getProfile, getSessionHistory, getLatestSession, getWeeklyReport } from "../services/dashboardService";
import "../styles/dashboard.css";

// ── TOKENS ──
const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];

const BIOMARKERS_EMPTY = [
  { name: "Semantic\ncoherence",   val: "—", tr: "—", dir: "flat", st: "good", key: "semantic_coherence"   },
  { name: "Lexical\ndiversity",    val: "—", tr: "—", dir: "flat", st: "warn", key: "lexical_diversity"    },
  { name: "Idea\ndensity",         val: "—", tr: "—", dir: "flat", st: "good", key: "idea_density"         },
  { name: "Speech\nrate",          val: "—",  tr: "—", dir: "flat", st: "good", key: "speech_rate"          },
  { name: "Pause\nfreq.",          val: "—",  tr: "—", dir: "flat", st: "warn", key: "pause_frequency"      },
  { name: "Pause\ndur.",           val: "—", tr: "—", dir: "flat", st: "good", key: "pause_duration"       },
  { name: "Pitch\nmean",           val: "—",  tr: "—", dir: "flat", st: "good", key: "pitch_mean"           },
  { name: "Pitch\nrange",          val: "—",   tr: "—", dir: "flat", st: "warn", key: "pitch_range"          },
  { name: "Jitter",                val: "—",tr: "—", dir: "flat", st: "good", key: "jitter"               },
  { name: "Shimmer",               val: "—", tr: "—", dir: "flat", st: "good", key: "shimmer"              },
  { name: "HNR",                   val: "—", tr: "—", dir: "flat", st: "good", key: "hnr"                  },
  { name: "Syntactic",             val: "—", tr: "—", dir: "flat", st: "good", key: "syntactic_complexity" },
  { name: "Artic.\nrate",          val: "—",  tr: "—", dir: "flat", st: "good", key: "articulation_rate"    },
  { name: "Emotional",             val: "—", tr: "—", dir: "flat", st: "good", key: "emotional_entropy"    },
];

const BIOMARKERS_STATIC = [
  { name: "Semantic\ncoherence",   val: "0.81", tr: "+2.1%", dir: "up",   st: "good", key: "semantic_coherence"   },
  { name: "Lexical\ndiversity",    val: "0.74", tr: "-0.8%", dir: "down", st: "warn", key: "lexical_diversity"    },
  { name: "Idea\ndensity",         val: "0.61", tr: "+1.4%", dir: "up",   st: "good", key: "idea_density"         },
  { name: "Speech\nrate",          val: "118",  tr: "0.0%",  dir: "flat", st: "good", key: "speech_rate"          },
  { name: "Pause\nfreq.",          val: "3.2",  tr: "+5.2%", dir: "down", st: "warn", key: "pause_frequency"      },
  { name: "Pause\ndur.",           val: "0.42", tr: "-1.1%", dir: "up",   st: "good", key: "pause_duration"       },
  { name: "Pitch\nmean",           val: "182",  tr: "+0.6%", dir: "up",   st: "good", key: "pitch_mean"           },
  { name: "Pitch\nrange",          val: "64",   tr: "-2.3%", dir: "down", st: "warn", key: "pitch_range"          },
  { name: "Jitter",                val: "0.012",tr: "-0.4%", dir: "up",   st: "good", key: "jitter"               },
  { name: "Shimmer",               val: "0.08", tr: "+0.2%", dir: "flat", st: "good", key: "shimmer"              },
  { name: "HNR",                   val: "18.4", tr: "+1.8%", dir: "up",   st: "good", key: "hnr"                  },
  { name: "Syntactic",             val: "0.70", tr: "+0.9%", dir: "up",   st: "good", key: "syntactic_complexity" },
  { name: "Artic.\nrate",          val: "4.8",  tr: "-0.5%", dir: "flat", st: "good", key: "articulation_rate"    },
  { name: "Emotional",             val: "0.65", tr: "-1.2%", dir: "up",   st: "good", key: "emotional_entropy"    },
];

const MOCK_HISTORY = [
  { date: new Date(new Date().setDate(new Date().getDate() - 2)).toISOString(), risk_tier: "Green", semantic_coherence: 0.82, speech_rate: 115, pause_frequency: 3.1 },
  { date: new Date(new Date().setDate(new Date().getDate() - 4)).toISOString(), risk_tier: "Green", semantic_coherence: 0.81, speech_rate: 118, pause_frequency: 3.2 },
  { date: new Date(new Date().setDate(new Date().getDate() - 7)).toISOString(), risk_tier: "Yellow", semantic_coherence: 0.76, speech_rate: 105, pause_frequency: 4.8 },
  { date: new Date(new Date().setDate(new Date().getDate() - 8)).toISOString(), risk_tier: "Green", semantic_coherence: 0.80, speech_rate: 112, pause_frequency: 3.5 },
  { date: new Date(new Date().setDate(new Date().getDate() - 11)).toISOString(), risk_tier: "Green", semantic_coherence: 0.83, speech_rate: 119, pause_frequency: 3.0 },
];

// ── SPARKLINE ──
const LC = { good:"#10B981", warn:"#F59E0B", crit:"#EF4444" };
const DC = { good:"#6EE7B7", warn:"#FDE68A", crit:"#FCA5A5" };

const Sparkline = ({ status, dark }) => {
  const ref = useRef(null);
  const draw = useCallback(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const W = c.offsetWidth || 80, H = 22;
    c.width = W; c.height = H;
    let pts = [], v = 0.5 + Math.random() * 0.2;
    for (let i = 0; i < 14; i++) {
      if (status === "static") {
        v += (Math.random() - 0.48) * 0.07;
        v = Math.max(0.1, Math.min(0.9, v));
        pts.push(v);
      } else {
        pts.push(0.5); // Flat line if real historical data is not passed
      }
    }
    const mn = Math.min(...pts), mx = Math.max(...pts), rng = mx - mn || 0.01;
    const col = dark ? DC : LC;
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = (i / (pts.length - 1)) * W;
      const y = H - ((p - mn) / rng) * (H - 4) - 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = col[status] || col.good;
    ctx.lineWidth = 1.5; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.stroke();
  }, [status, dark]);
  useEffect(() => { draw(); }, [draw]);
  return <canvas ref={ref} className="sparkline" />;
};

// ── LOGO ──
const LogoIcon = () => (
  <svg width="14" height="14" viewBox="0 0 18 18" fill="none">
    <circle cx="9" cy="9" r="5" stroke="#fff" strokeWidth="1.6"/>
    <circle cx="9" cy="9" r="1.8" fill="#fff"/>
    <line x1="9" y1="2"  x2="9"  y2="4"  stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
    <line x1="9" y1="14" x2="9"  y2="16" stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
    <line x1="2" y1="9"  x2="4"  y2="9"  stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
    <line x1="14" y1="9" x2="16" y2="9"  stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
  </svg>
);

// ── CALENDAR ──
const Calendar = ({ dark, history = [] }) => {
  const today = new Date();
  const [year, setYear] = useState(2026);
  const [month, setMonth] = useState(2); // March
  const [selected, setSelected] = useState(20);

  const firstDay = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const isCurMonth = year === today.getFullYear() && month === today.getMonth();

  // Build sessionMap from real API history
  const sessionMap = {};
  if (history.length > 0) {
    history.forEach(item => {
      const d = new Date(item.date);
      const dayNum = d.getDate();
      const monthMatch = d.getMonth() === month && d.getFullYear() === year;
      if (monthMatch) {
        const riskToStatus = { Green: "good", Yellow: "warn", Red: "bad" };
        sessionMap[dayNum] = {
          status: riskToStatus[item.risk_tier] || "good",
          coherence: item.semantic_coherence != null ? String(item.semantic_coherence) : "—",
          speechRate: item.speech_rate != null ? String(item.speech_rate) : "—",
          pauseFreq: item.pause_frequency != null ? String(item.pause_frequency) : "—",
          risk: item.risk_tier || "Green",
        };
      }
    });
  }

  const SESSION_DATA = sessionMap;
  const RISK_CLS = { Green:"good", Yellow:"warn", Red:"bad" };
  const selectedSess = selected && SESSION_DATA[selected];

  return (
    <div className="calendar-card">
      <div className="cal-header">
        <span className="cal-title">Session history</span>
        <div className="cal-nav-group">
          <button className="cal-nav-btn" onClick={() => {
            if (month === 0) { setMonth(11); setYear(y => y-1); } else setMonth(m => m-1);
            setSelected(null);
          }}>‹</button>
          <span className="cal-month-label">{MONTHS[month]} {year}</span>
          <button className="cal-nav-btn" onClick={() => {
            if (month === 11) { setMonth(0); setYear(y => y+1); } else setMonth(m => m+1);
            setSelected(null);
          }}>›</button>
        </div>
      </div>

      <div className="day-labels-row">
        {["Su","Mo","Tu","We","Th","Fr","Sa"].map(d => (
          <div key={d} className="day-label">{d}</div>
        ))}
      </div>

      <div className="day-grid">
        {Array(firstDay).fill(null).map((_, i) => (
          <div key={`e${i}`} className="day-cell empty" />
        ))}
        {Array(daysInMonth).fill(null).map((_, i) => {
          const d = i + 1;
          const sess = SESSION_DATA[d];
          const isFuture = isCurMonth && d > today.getDate();
          const cls = isFuture ? "empty" : sess ? (sess.status === "bad" ? "bad" : sess.status) : "empty";
          return (
            <div key={d}
              className={`day-cell ${cls} ${isCurMonth && d === today.getDate() ? "today" : ""} ${selected === d ? "selected" : ""}`}
              onClick={() => { if (sess && !isFuture) setSelected(d); }}
            >
              <span className="day-num">{d}</span>
              {sess && !isFuture && <span className="day-dot" />}
            </div>
          );
        })}
      </div>

      <div className="cal-legend">
        <div className="leg-item"><div className="leg-dot" style={{background:"rgba(16,185,129,0.5)"}} />Good</div>
        <div className="leg-item"><div className="leg-dot" style={{background:"rgba(245,158,11,0.5)"}} />Watch</div>
        <div className="leg-item"><div className="leg-dot" style={{background:"rgba(239,68,68,0.45)"}} />Alert</div>
        <div className="leg-item"><div className="leg-dot" style={{background:"var(--card2)",border:"1px solid var(--border)"}} />None</div>
      </div>

      {selectedSess && (
        <div className="day-detail-panel">
          <div className="dd-date">
            {new Date(year, month, selected).toLocaleDateString("en-US",{weekday:"long",month:"long",day:"numeric",year:"numeric"})}
          </div>
          <div className={`dd-risk ${RISK_CLS[selectedSess.risk] || "good"}`}>
            ● Risk tier: {selectedSess.risk}
          </div>
          <div className="dd-metrics">
            <div className="dd-metric">
              <div className="dd-metric-label">Semantic coherence</div>
              <div className="dd-metric-value">{selectedSess.coherence}</div>
              <div className="dd-metric-trend" style={{color:`var(--${selectedSess.status === "bad" ? "danger" : "success"})`}}>
                {selectedSess.status === "bad" ? "▼ Low" : "▲ Normal"}
              </div>
            </div>
            <div className="dd-metric">
              <div className="dd-metric-label">Speech rate</div>
              <div className="dd-metric-value">{selectedSess.speechRate}</div>
              <div className="dd-metric-trend" style={{color:"var(--success)"}}>▲ Normal</div>
            </div>
            <div className="dd-metric">
              <div className="dd-metric-label">Pause freq.</div>
              <div className="dd-metric-value">{selectedSess.pauseFreq}</div>
              <div className="dd-metric-trend" style={{color:parseFloat(selectedSess.pauseFreq)>4?"var(--danger)":"var(--success)"}}>
                {parseFloat(selectedSess.pauseFreq) > 4 ? "▼ High" : "▲ Normal"}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ── SKELETON ──
const Skeleton = ({ w = "100%", h = 16, radius = 6, style = {} }) => (
  <div style={{
    width: w, height: h, borderRadius: radius,
    background: "linear-gradient(90deg, var(--border) 25%, var(--card2) 50%, var(--border) 75%)",
    backgroundSize: "200% 100%",
    animation: "shimmer 1.4s infinite",
    ...style
  }} />
);

// ── MAIN ──
const Dashboard = () => {
  const navigate = useNavigate();
  const { token, user, logout } = useAuth();

  const [dark, setDark] = useState(
    () => localStorage.getItem("cog_dark") === "true"
  );

  const [profile,  setProfile]  = useState(null);
  const [history,  setHistory]  = useState([]);
  const [latest,   setLatest]   = useState(null);
  const [report,   setReport]   = useState(null);
  const [fetching, setFetching] = useState(true);

  useEffect(() => {
    if (!token) return;
    const fetchAll = async () => {
      try {
        const [prof, hist, lat, rep] = await Promise.all([
          getProfile(token),
          getSessionHistory(token, 1),
          getLatestSession(token),
          getWeeklyReport(token),
        ]);
        setProfile(prof);
        setHistory(hist || []);
        setLatest(lat);
        setReport(rep);
      } catch (err) {
        console.error("Dashboard fetch error:", err);
        if (err.message?.includes("401") || err.message?.includes("Unauthorized")) {
          logout();
        }
      } finally {
        setFetching(false);
      }
    };
    fetchAll();
  }, [token]);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    localStorage.setItem("cog_dark", String(next));
  };

  const getBmVal = (bm) => {
    if (latest?.[bm.key] != null) {
      const val = latest[bm.key];
      return typeof val === "number" ? (val < 10 ? val.toFixed(3) : Math.round(val).toString()) : String(val);
    }
    return "—";
  };

  const getBmStatus = (bm) => {
    if (latest?.anomaly_flags) {
      try {
        const flags = Array.isArray(latest.anomaly_flags)
          ? latest.anomaly_flags
          : JSON.parse(latest.anomaly_flags || "[]");
        if (flags.includes(bm.key)) return "warn";
      } catch {}
    }
    return "—";
  };

  const isDemo = user?.email?.includes("demo");

  const firstName = profile?.name?.split(" ")[0] || user?.name?.split(" ")[0] || "there";
  const cogAge = isDemo ? 52 : (profile?.sessions_total > 0 && profile?.cognitive_age != null ? profile.cognitive_age : "—");
  const bioAge = isDemo ? 58 : ((profile?.dob || user?.dob) ? (new Date().getFullYear() - new Date(profile?.dob || user?.dob).getFullYear()) : "—");
  const ageDiff = isDemo ? 6 : ((bioAge !== "—" && cogAge !== "—") ? bioAge - cogAge : null);
  const streak = isDemo ? 14 : (profile?.streak ?? "—");
  const sessionsMonth = isDemo ? 12 : (profile?.sessions_this_month ?? "—");
  const lastSessionHours = profile?.last_session_hours_ago;
  const lastSessionStr = isDemo ? "2h ago" : (lastSessionHours != null ? `${lastSessionHours}h ago` : "—");
  const narrative = isDemo ? "Your word choice diversity and phonation metrics remain perfectly stable..." : (report?.narrative ?? null);
  const riskTier = isDemo ? "Green" : (profile?.risk_tier || latest?.risk_tier || "Green");
  
  const BIOMARKERS_DISPLAY = isDemo ? BIOMARKERS_STATIC : BIOMARKERS_EMPTY;
  const RISK_COLORS = { Green: "success", Yellow: "warn", Red: "danger" };
  const riskColor = RISK_COLORS[riskTier] || "success";

  return (
    <div className={`dash-root ${dark ? "dm" : "lm"}`}>
      <style>{`@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }`}</style>

      {/* NAV */}
      <nav className="dash-nav">
        <a href="/dashboard" className="nav-logo">
          <div className="nav-logo-box"><LogoIcon /></div>
          <span className="nav-logo-name">CogniSafe</span>
        </a>
        <div className="nav-links">
          {[
            { label:"Dashboard", path:"/dashboard" },
            { label:"Session",   path:"/session"   },
            { label:"Brain",     path:"/brain"      },
            { label:"Report",    path:"/ar-report"  },
          ].map(l => (
            <button key={l.label}
              className={`nav-link ${window.location.pathname === l.path ? "active" : ""}`}
              onClick={() => navigate(l.path)}>{l.label}</button>
          ))}
        </div>
        <div className="nav-right">
          <button className="nav-mode-btn" onClick={toggleDark}>
            {dark ? "☀️ Light" : "🌙 Dark"}
          </button>
          <button className="nav-session-btn" onClick={() => navigate("/session")}>
            + Start session
          </button>
          <div className="nav-avatar"
            style={{cursor:"pointer"}}
            title="Click to logout"
            onClick={logout}
          >
            {user?.name?.charAt(0)?.toUpperCase() || "U"}
          </div>
        </div>
      </nav>

      {/* BODY */}
      <div className="dash-body">
        <div className="dash-greeting-sub">
          Good morning · {new Date().toLocaleDateString("en-US",{weekday:"long",month:"long",day:"numeric"})}
        </div>
        <div className="dash-greeting-title">
          Welcome back, <em>{firstName}</em> — your cognitive pulse
        </div>

        {/* HERO */}
        <div className="dash-hero">

          {/* Risk card */}
          <div className="d-card">
            <div className="card-label">Current risk tier</div>
            {fetching ? (
              <>
                <Skeleton w="160px" h={28} style={{marginBottom:8}} />
                <Skeleton h={14} style={{marginBottom:4}} />
                <Skeleton w="70%" h={14} />
              </>
            ) : (
              <>
                <div className="risk-pill">
                  <span className="risk-dot" style={{background:`var(--${riskColor})`}} />
                  <span style={{color:`var(--${riskColor})`}}>
                    Cognitive health: {riskTier === "Green" ? "Good" : riskTier === "Yellow" ? "Watch" : "Alert"}
                  </span>
                </div>
                <p className="risk-desc">
                  {riskTier === "Green"
                    ? "Your voice patterns are stable. No significant changes detected over the last 30 days."
                    : "Your voice patterns are showing some variation. Continue monitoring daily."}
                </p>
                <p className="risk-upd">Updated today</p>
              </>
            )}
          </div>

          {/* Cognitive age card */}
          <div className="d-card-mint">
            <div className="card-label-w">Cognitive age</div>
            {fetching ? (
              <Skeleton w="80px" h={48} style={{marginBottom:8,opacity:0.5}} />
            ) : (
              <>
                <div className="cog-num">{profile?.sessions_total > 0 ? cogAge : "—"}</div>
                <div className="cog-sub">
                  {bioAge ? `Biological age: ${bioAge} — ` : ""}
                  {(isDemo || (profile?.sessions_total > 0 && ageDiff != null && ageDiff > 0))
                    ? <b>{ageDiff} yrs younger ✦</b>
                    : (profile?.sessions_total > 0 ? "calculating…" : "record to calculate")}
                </div>
                <div className="cog-bar">
                  <div className="cog-bar-fill" style={{
                    width: (isDemo || profile?.sessions_total > 0) && cogAge !== "—" ? `${Math.min(100, (Number(cogAge) / 80) * 100)}%` : "0%"
                  }} />
                </div>
              </>
            )}
          </div>

          {/* Stats card */}
          <div className="d-card-indigo">
            <div className="stat-row">
              <div className="stat-row-label">Sessions this month</div>
              <div className="stat-row-value">{fetching ? "…" : sessionsMonth}</div>
              <div className="stat-row-sub">
                {!fetching && (report?.sessions_this_week != null ? `${report.sessions_this_week} this week` : "0 this week")}
              </div>
            </div>
            <div className="stat-row">
              <div className="stat-row-label">Current streak</div>
              <div className="stat-row-value">{fetching ? "…" : `${streak}d 🔥`}</div>
              <div className="stat-row-sub">
                {!fetching && (profile?.best_streak ? `Personal best: ${profile.best_streak}d` : "Keep it up!")}
              </div>
            </div>
            <div className="stat-row">
              <div className="stat-row-label">Last session</div>
              <div className="stat-row-value">{fetching ? "…" : lastSessionStr}</div>
              <div className="stat-row-sub">
                {!fetching && lastSessionHours != null && lastSessionHours < 12 ? "this morning" : ""}
              </div>
            </div>
          </div>
        </div>

        {/* BIOMARKERS */}
        <div className="section-header">
          <span className="section-title">Biomarker dashboard</span>
          <span className="section-sub">14 metrics · live</span>
        </div>
        <div className="chips-row">
          <span className="chip good">
            {fetching ? "…" : `${BIOMARKERS_DISPLAY.filter(b => getBmStatus(b) === "good").length} stable`}
          </span>
          <span className="chip warn">
            {fetching ? "…" : `${BIOMARKERS_DISPLAY.filter(b => getBmStatus(b) === "warn").length} watch`}
          </span>
          <span className="chip crit">{fetching ? "…" : "0 critical"}</span>
        </div>
        <div className="bm-grid">
          {BIOMARKERS_DISPLAY.map(bm => {
            const arrow = isDemo ? bm.tr.charAt(0) : "—";
            const trCls = isDemo ? bm.dir : "flat";
            const currentStatus = isDemo ? bm.st : getBmStatus(bm);
            const currentVal = isDemo ? bm.val : getBmVal(bm);
            return (
              <div key={bm.name} className="bm-card">
                <div className="bm-top">
                  <span className="bm-name" style={{whiteSpace:"pre-line"}}>{bm.name}</span>
                  <span className={`bm-dot ${currentStatus}`} />
                </div>
                {fetching
                  ? <Skeleton h={20} style={{marginBottom:4}} />
                  : <div className="bm-value">{currentVal}</div>
                }
                <div className={`bm-trend ${trCls}`}>{arrow} {bm.tr}</div>
                <Sparkline status={isDemo ? "static" : currentStatus} dark={dark} />
              </div>
            );
          })}
        </div>

        {/* BOTTOM */}
        <div className="dash-bottom">

          {/* Weekly report card */}
          <div className="report-card">
            <div className="report-accent" />
            <div className="report-label">This week, your voice told us…</div>
            {fetching ? (
              <>
                <Skeleton h={14} style={{marginBottom:6}} />
                <Skeleton h={14} style={{marginBottom:6}} />
                <Skeleton w="60%" h={14} style={{marginBottom:16}} />
                <Skeleton w="120px" h={34} radius={8} />
              </>
            ) : (
              <>
                <p className="report-text">
                  {narrative ? narrative : (
                     "No sessions recorded yet."
                  )}
                </p>
                <div className="report-actions">
                  <button className="report-btn-primary" onClick={() => navigate("/ar-report")}>View full report</button>
                  <button className="report-btn-secondary">Share with doctor</button>
                </div>
              </>
            )}
          </div>

          {/* Calendar */}
          <Calendar dark={dark} history={isDemo ? MOCK_HISTORY : history} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
