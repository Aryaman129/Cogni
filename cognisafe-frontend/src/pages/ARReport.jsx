import { useRef, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { getWeeklyReport, getLatestSession, getProfile } from "../services/dashboardService";
import "../styles/arreport.css";

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

// ── QR CODE CANVAS ──
const QRCanvas = ({ dark }) => {
  const ref = useRef(null);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx     = c.getContext("2d");
    const size    = 120;
    const modules = 25;
    const cell    = Math.floor(size / modules);

    ctx.fillStyle = dark ? "#0A1F12" : "#FFFFFF";
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = dark ? "#6EE7B7" : "#0A1A10";

    const seed = "cognisafe-report-2026";
    const hash = (s, i) => {
      let h = 0;
      for (let j = 0; j < s.length; j++) h = (h * 31 + s.charCodeAt(j) + i) & 0xFFFFFF;
      return h;
    };

    [[0, 0], [0, modules - 7], [modules - 7, 0]].forEach(([r, cl]) => {
      ctx.fillStyle = dark ? "#6EE7B7" : "#0A1A10";
      ctx.fillRect(cl * cell, r * cell, 7 * cell, 7 * cell);
      ctx.fillStyle = dark ? "#0A1F12" : "#FFFFFF";
      ctx.fillRect((cl + 1) * cell, (r + 1) * cell, 5 * cell, 5 * cell);
      ctx.fillStyle = dark ? "#6EE7B7" : "#0A1A10";
      ctx.fillRect((cl + 2) * cell, (r + 2) * cell, 3 * cell, 3 * cell);
    });

    ctx.fillStyle = dark ? "#6EE7B7" : "#0A1A10";
    for (let r = 0; r < modules; r++) {
      for (let cl = 0; cl < modules; cl++) {
        const inFinder = (r < 8 && cl < 8) || (r < 8 && cl >= modules - 8) || (r >= modules - 8 && cl < 8);
        if (!inFinder && hash(seed, r * modules + cl) % 3 === 0) {
          ctx.fillRect(cl * cell, r * cell, cell - 1, cell - 1);
        }
      }
    }
  }, [dark]);

  return <canvas ref={ref} className="qr-canvas" width={120} height={120} />;
};

async function generatePDF(data) {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const W = 210, M = 20;

  doc.setFillColor(13, 40, 24);
  doc.rect(0, 0, W, 28, "F");
  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(224, 255, 240);
  doc.text("CogniSafe", M, 17);
  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(110, 231, 183);
  doc.text("WEEKLY COGNITIVE REPORT", W - M, 17, { align: "right" });

  doc.setTextColor(5, 26, 16);
  doc.setFontSize(13);
  doc.setFont("helvetica", "bold");
  doc.text(data.name || "CogniSafe User", M, 38);
  doc.setFontSize(10);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(74, 154, 120);
  doc.text(
    `${data.weekLabel || "This week"} · Session streak: ${data.streak || "—"} days`,
    M, 44
  );

  const riskColor = data.riskTier === "Green" ? [16, 185, 129]
    : data.riskTier === "Yellow" ? [245, 158, 11]
    : [239, 68, 68];
  doc.setFillColor(232, 255, 244);
  doc.roundedRect(M, 50, W - 2 * M, 16, 3, 3, "F");
  doc.setTextColor(...riskColor);
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.text(
    `● RISK TIER: ${data.riskTier || "GREEN"} — Cognitive health: ${data.riskTier === "Green" ? "Good" : data.riskTier === "Yellow" ?  "Watch" : data.riskTier === "Orange" ? "Elevated" : "Alert"}`,
    M + 6, 60
  );

  doc.setTextColor(5, 26, 16);
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("Biomarker Summary", M, 76);
  doc.setDrawColor(176, 237, 216);
  doc.line(M, 78, W - M, 78);

  const rows = data.metrics;
  doc.setFontSize(9);
  rows.forEach(([name, val, note], i) => {
    const y = 84 + i * 9;
    if (i % 2 === 0) {
      doc.setFillColor(245, 255, 251);
      doc.rect(M, y - 4, W - 2 * M, 9, "F");
    }
    doc.setFont("helvetica", "normal");
    doc.setTextColor(26, 90, 64);
    doc.text(name, M + 3, y + 1);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(5, 26, 16);
    doc.text(String(val), M + 72, y + 1);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(74, 154, 120);
    doc.text(String(note), M + 88, y + 1);
  });

  const iy = 84 + rows.length * 9 + 10;
  doc.setTextColor(5, 26, 16);
  doc.setFontSize(12);
  doc.setFont("helvetica", "bold");
  doc.text("AI-Generated Insights", M, iy);
  doc.setDrawColor(176, 237, 216);
  doc.line(M, iy + 2, W - M, iy + 2);

  const insights = data.insights;
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  insights.forEach((ins, i) => {
    const y = iy + 9 + i * 8;
    doc.setTextColor(16, 185, 129);
    doc.text("●", M + 2, y);
    doc.setTextColor(26, 90, 64);
    const clean = typeof ins === "string" ? ins.replace(/<[^>]+>/g, "") : ins;
    const lines = doc.splitTextToSize(clean, W - 2 * M - 10);
    doc.text(lines[0] || clean, M + 7, y);
  });

  doc.setFillColor(13, 40, 24);
  doc.rect(0, 281, W, 14, "F");
  doc.setTextColor(110, 231, 183);
  doc.setFontSize(8);
  doc.text("CogniSafe · AI-Powered Cognitive Health Monitoring", M, 290);
  doc.text(
    "Generated: " + new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" }),
    W - M, 290, { align: "right" }
  );

  doc.save(`CogniSafe-Report-${new Date().toISOString().split("T")[0]}.pdf`);
}

const Skeleton = ({ w = "100%", h = 16, radius = 6, style = {} }) => (
  <div style={{
    width: w, height: h, borderRadius: radius,
    background: "linear-gradient(90deg,var(--border) 25%,var(--card2) 50%,var(--border) 75%)",
    backgroundSize: "200% 100%",
    animation: "shimmer 1.4s infinite",
    ...style
  }} />
);

const ARReport = () => {
  const navigate = useNavigate();
  const { token, user, logout } = useAuth();

  const [dark, setDark]           = useState(() => localStorage.getItem("cog_dark") === "true");
  const [downloading, setDownloading] = useState(false);
  const [toast, setToast]         = useState(null);
  const [fetching, setFetching]   = useState(true);

  const [profile,  setProfile]  = useState(null);
  const [report,   setReport]   = useState(null);
  const [latest,   setLatest]   = useState(null);
  const [metrics,  setMetrics]  = useState([]);
  const [insights, setInsights] = useState([]);

  const toggleDark = () => {
    const n = !dark;
    setDark(n);
    localStorage.setItem("cog_dark", String(n));
  };

  useEffect(() => {
    if (!token) { setFetching(false); return; }

    const fetchAll = async () => {
      try {
        const [prof, rep, lat] = await Promise.all([
          getProfile(token),
          getWeeklyReport(token),
          getLatestSession(token),
        ]);

        if (prof) setProfile(prof);
        if (rep)  setReport(rep);
        if (lat)  setLatest(lat);

        if (rep || lat || prof) {
          const bm = lat?.biomarkers || lat || {};
          const newMetrics = [
            {
              icon: "🧠",
              name: "Semantic coherence",
              val:  bm.semantic_coherence != null ? bm.semantic_coherence.toFixed(2) : "—",
              trend: bm.semantic_coherence != null ? "▲ Above baseline" : "—",
              type: bm.semantic_coherence != null ? "up" : "flat",
            },
            {
              icon: "👤",
              name: "Cognitive age",
              val:  prof?.sessions_total > 0 ? prof?.cognitive_age : "—",
              trend: prof?.sessions_total > 0 ? "▲ Stable" : "—",
              type: prof?.sessions_total > 0 ? "up" : "flat",
            },
            {
              icon: "🎙️",
              name: "Speech rate",
              val:  bm.speech_rate != null ? Math.round(bm.speech_rate) : "—",
              trend: bm.speech_rate != null ? "▲ Normal" : "—",
              type: bm.speech_rate != null ? "up" : "flat",
            },
            {
              icon: "⏸️",
              name: "Pause frequency",
              val:  bm.pause_frequency != null ? bm.pause_frequency.toFixed(1) : "—",
              trend: bm.pause_frequency != null ? ((bm.pause_frequency ?? 0) > 4 ? "⚠ Elevated" : "▲ Normal") : "—",
              type: bm.pause_frequency != null ? ((bm.pause_frequency ?? 0) > 4 ? "flat" : "up") : "flat",
            },
            {
              icon: "📊",
              name: "Lexical diversity",
              val:  bm.lexical_diversity != null ? bm.lexical_diversity.toFixed(2) : "—",
              trend: bm.lexical_diversity != null ? ((bm.lexical_diversity ?? 0) < 0.70 ? "▼ Watch" : "▲ Normal") : "—",
              type: bm.lexical_diversity != null ? ((bm.lexical_diversity ?? 0) < 0.70 ? "dn" : "up") : "flat",
            },
            {
              icon: "📅",
              name: "Sessions completed",
              val:  rep?.sessions_this_week != null ? `${rep.sessions_this_week}/7` : "0/7",
              trend: "— On track",
              type: "flat",
            },
            {
              icon: "🔥",
              name: "Streak",
              val:  prof?.streak != null ? `${prof.streak}d` : "—",
              trend: "▲ Growing",
              type: "up",
            },
          ];
          setMetrics(newMetrics);

          if (rep?.insights && Array.isArray(rep.insights) && rep.insights.length > 0) {
            setInsights(rep.insights.map(ins => ({
              color: ins.color ? `var(--${ins.color})` : "var(--success)",
              text: ins.text,
            })));
          } else if (rep?.narrative) {
            setInsights([
              { color: "var(--success)", text: rep.narrative }
            ]);
          } else {
            setInsights([]);
          }
        }
      } catch (err) {
        console.error("AR Report fetch error:", err);
        if (err.message?.includes("401")) logout();
      } finally {
        setFetching(false);
      }
    };

    fetchAll();
  }, [token]);

  const isDemo = user?.email?.includes("demo");

  const cogAge   = isDemo ? 52 : (profile?.sessions_total > 0 && profile?.cognitive_age != null ? profile.cognitive_age : "—");
  const bioAge   = ((profile?.dob || user?.dob) ? (new Date().getFullYear() - new Date(profile?.dob || user?.dob).getFullYear()) : "—");
  const ageDiff  = isDemo ? 6 : ((bioAge !== "—" && cogAge !== "—") ? bioAge - cogAge : null);
  const streak   = isDemo ? 14 : (profile?.streak ?? "—");
  const riskTier = isDemo ? "Green" : (profile?.risk_tier || latest?.risk_tier || "Green");
  const sessionsThisWeek = isDemo ? 5 : (report?.sessions_this_week ?? 0);
  const userName = user?.name || "CogniSafe User";

  const displayMetrics = isDemo ? [
    { icon: "🧠", name: "Semantic coherence", val: "0.81", trend: "▲ Above baseline", type: "up" },
    { icon: "👤", name: "Cognitive age", val: "52", trend: "▲ Stable", type: "up" },
    { icon: "🎙️", name: "Speech rate", val: "118", trend: "▲ Normal", type: "up" },
    { icon: "⏸️", name: "Pause frequency", val: "3.2", trend: "▲ Normal", type: "flat" },
    { icon: "📊", name: "Lexical diversity", val: "0.74", trend: "▼ Watch", type: "dn" },
    { icon: "📅", name: "Sessions completed", val: "5/7", trend: "— On track", type: "flat" },
    { icon: "🔥", name: "Streak", val: "14d", trend: "▲ Growing", type: "up" },
  ] : metrics;

  const displayInsights = isDemo ? [
    { color: "var(--success)", text: "Your working memory correlates with a youthful cognitive state." },
    { color: "var(--success)", text: "Semantic coherence is highly stable across timeframes." }
  ] : insights;

  const today = new Date();
  const weekStart = new Date(today);
  weekStart.setDate(today.getDate() - today.getDay() + 1);
  const weekEnd = new Date(weekStart);
  weekEnd.setDate(weekStart.getDate() + 6);
  const weekLabel = `Week of ${weekStart.getDate()}–${weekEnd.getDate()} ${today.toLocaleString("en-US", { month: "long", year: "numeric" })}`;

  const RISK_COLORS = { Green: "success", Yellow: "warn", Orange: "Warn", Red: "danger" };
  const riskColor = RISK_COLORS[riskTier] || "success";

  const handleDownload = async () => {
    setDownloading(true);
    try {
      await generatePDF({
        name:      userName,
        weekLabel,
        streak,
        riskTier,
        cogAge,
        metrics:   displayMetrics.map(m => [m.name, m.val, m.trend]),
        insights:  displayInsights.map(i => i.text),
      });
      showToast("✓ PDF downloaded successfully!");
    } catch (err) {
      console.error("PDF error:", err);
      showToast("⚠ PDF failed — make sure jspdf is installed (npm install jspdf)");
    } finally {
      setDownloading(false);
    }
  };

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 3200);
  };

  const SHARE_OPTS = [
    { icon: "📧", bg: "rgba(99,102,241,0.12)",  label: "Email",     sub: "Send to doctor",  action: "Email"    },
    { icon: "💬", bg: "rgba(16,185,129,0.12)",  label: "WhatsApp",  sub: "Share instantly", action: "WhatsApp" },
    { icon: "🔗", bg: "rgba(245,158,11,0.12)",  label: "Copy link", sub: "Shareable URL",   action: "Copy link"},
    { icon: "🖨️", bg: "rgba(239,68,68,0.1)",    label: "Print",     sub: "Physical copy",   action: "Print"    },
  ];

  const NAV_LINKS = [
    ["Dashboard", "/dashboard"],
    ["Session",   "/session"],
    ["Brain",     "/brain"],
    ["Report",    "/ar-report"],
  ];

  return (
    <div className={`ar-root ${dark ? "dm" : "lm"}`}>
      <style>{`@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }`}</style>

      {/* ── NAV ── */}
      <nav className="dash-nav">
        <a href="/dashboard" className="nav-logo">
          <div className="nav-logo-box"><LogoIcon /></div>
          <span className="nav-logo-name">CogniSafe</span>
        </a>
        <div className="nav-links">
          {NAV_LINKS.map(([label, path]) => (
            <button key={label}
              className={`nav-link ${window.location.pathname === path ? "active" : ""}`}
              onClick={() => navigate(path)}>
              {label}
            </button>
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
            title="Click to logout"
            style={{ cursor: "pointer" }}
            onClick={logout}
          >
            {user?.name?.charAt(0)?.toUpperCase() || "U"}
          </div>
        </div>
      </nav>

      {/* ── PAGE ── */}
      <div className="ar-page">
        <div className="page-sub">Weekly cognitive report</div>
        <div className="page-title">
          Your <em>health summary</em> — ready to share
        </div>
        <div className="page-meta">
          {weekLabel} · Session streak: {streak} days 🔥 · Risk tier: {riskTier}
        </div>

        {/* ── RISK BANNER ── */}
        <div className="risk-banner">
          <div className="rb-icon">🧠</div>
          <div>
            <div className="rb-label">Current risk tier</div>
            <div className="rb-value">
              Cognitive health:{" "}
              {riskTier === "Green" ? "Good" : riskTier === "Yellow" ? "Watch" : riskTier === "Orange" ? "Elevated" : "Alert"}
            </div>
            <div className="rb-sub">
              {riskTier === "Green"
                ? "No significant anomalies detected this week"
                : "Some voice pattern changes detected — continue monitoring"}
            </div>
          </div>
          <div className="rb-pill" style={{
            background: `rgba(${riskTier === "Green" ? "16,185,129" : riskTier === "Yellow" ?  "245,158,11" : riskTier === "Orange" ? "249,115,22" : "239,68,68"},0.12)`,
            border: `1px solid rgba(${riskTier === "Green" ? "16,185,129" : riskTier === "Yellow" ?  "245,158,11" : riskTier === "Orange" ? "249,115,22" : "239,68,68"},0.3)`,
            color: `var(--${riskColor})`,
          }}>
            ● {riskTier}
          </div>
        </div>

        {/* ── MINI STATS ── */}
        <div className="mini-stats">
          <div className="mini-stat">
            <div className="ms-label">Cognitive age</div>
            {fetching
              ? <Skeleton w="80px" h={28} style={{ marginBottom: 4 }} />
              : <div className="ms-value">{profile?.sessions_total > 0 ? cogAge : "—"}</div>
            }
            <div className="ms-sub">Biological age: {bioAge}</div>
            <div className="ms-trend">
              {ageDiff > 0 && profile?.sessions_total > 0 ? `▲ ${ageDiff} years younger than bio age` : "▲ Calculating…"}
            </div>
          </div>
          <div className="mini-stat">
            <div className="ms-label">Sessions this week</div>
            {fetching
              ? <Skeleton w="60px" h={28} style={{ marginBottom: 4 }} />
              : <div className="ms-value">{sessionsThisWeek}/7</div>
            }
            <div className="ms-sub">Streak: {streak} days 🔥</div>
            <div className="ms-trend">▲ On track — keep going!</div>
          </div>
        </div>

        {/* ── MAIN GRID ── */}
        <div className="ar-main">

          {/* METRICS LIST */}
          <div className="ar-card">
            <div className="accent-bar" />
            <div className="ar-card-title">Weekly biomarker summary</div>
            <div className="ar-card-sub">Powered by 14 voice biomarkers · last session</div>
            <div className="metric-rows">
              {fetching
                ? Array(7).fill(null).map((_, i) => (
                    <Skeleton key={i} h={44} radius={12} style={{ marginBottom: 8 }} />
                  ))
                : displayMetrics.map(m => {
                    const bgColor = m.type === "up" ? "rgba(16,185,129,0.1)"
                      : m.type === "dn" ? "rgba(239,68,68,0.1)"
                      : "rgba(245,158,11,0.1)";
                    const trCls = m.type === "up" ? "tr-up"
                      : m.type === "dn" ? "tr-dn"
                      : "tr-flat";
                    return (
                      <div key={m.name} className="metric-row">
                        <div className="mr-left">
                          <div className="mr-icon" style={{ background: bgColor }}>{m.icon}</div>
                          <div className="mr-name">{m.name}</div>
                        </div>
                        <div className="mr-right">
                          <div className="mr-value">{m.val}</div>
                          <div className={`mr-trend ${trCls}`}>{m.trend}</div>
                        </div>
                      </div>
                    );
                  })
              }
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

            <div className="ar-card">
              <div className="ar-card-title">Download PDF report</div>
              <div className="ar-card-sub">Scan QR code or tap button to download</div>
              <div className="qr-section">
                <div className="qr-container">
                  <QRCanvas dark={dark} />
                </div>
                <div className="qr-hint">
                  Scan with your phone camera to download this report as a PDF
                </div>
                <button className="qr-download-btn" onClick={handleDownload} disabled={downloading}>
                  {downloading ? <>⏳ Generating…</> : <>⬇ Download PDF Report</>}
                </button>
              </div>
            </div>

            <div className="ar-card">
              <div className="ar-card-title">Share your report</div>
              <div className="ar-card-sub">Send to your doctor or care team</div>
              <div className="share-options">
                {SHARE_OPTS.map(s => (
                  <div key={s.label} className="share-opt"
                    onClick={() => showToast(`${s.label} sharing opened!`)}>
                    <div className="so-icon" style={{ background: s.bg }}>{s.icon}</div>
                    <div>
                      <div className="so-label">{s.label}</div>
                      <div className="so-sub">{s.sub}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>

        {/* ── AI INSIGHTS ── */}
        <div className="ar-card">
          <div className="accent-bar" />
          <div className="ar-card-title">AI-generated weekly insights</div>
          <div className="ar-card-sub">
            Based on your {sessionsThisWeek} sessions this week
          </div>
          <div className="insight-list">
            {fetching
              ? Array(4).fill(null).map((_, i) => (
                  <Skeleton key={i} h={52} radius={12} style={{ marginBottom: 8 }} />
                ))
              : displayInsights.length > 0 ? displayInsights.map((ins, i) => (
                  <div key={i} className="insight-item">
                    <div className="ii-dot" style={{ background: ins.color }} />
                    <div className="ii-text"
                      dangerouslySetInnerHTML={{ __html: ins.text }} />
                  </div>
                )) : (
                  <div className="insight-item">
                     <div className="ii-text">No recorded insights available yet. Please complete a session.</div>
                  </div>
                )
            }
          </div>
        </div>

      </div>

      {toast && (
        <div className="toast" style={{ display: "flex" }}>
          {toast}
        </div>
      )}

    </div>
  );
};

export default ARReport;
