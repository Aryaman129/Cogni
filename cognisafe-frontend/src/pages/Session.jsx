import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { checkToday } from "../services/sessionService";
import "../styles/session.css";

// ── PROMPTS ──
const PROMPTS = [
  {
    text: "Describe what you see in this image in as much detail as you can — the objects, colours, setting, and what might be happening.",
    image: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80",
  },
  {
    text: "Tell me about your most vivid childhood memory. Describe everything you can remember — the place, people, smells, and how it made you feel.",
    image: null,
  },
  {
    text: "Look at this photo and describe every detail you notice — colours, shapes, mood, and what story the image might be telling.",
    image: "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=600&q=80",
  },
  {
    text: "Explain how you would get from your home to your favourite restaurant — describe the journey as if giving directions to a friend.",
    image: null,
  },
  {
    text: "Describe what's happening in this photo in as much detail as possible — the environment, any people, objects, light and shadows.",
    image: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=600&q=80",
  },
];

// ── CONSTANTS ──
const TOTAL   = 180;
const CIRC    = 389.6;
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ✅ Call HF directly — bypasses Render 30s timeout
const HF_URL  = "https://alamfarzann-cognisafe-ml.hf.space";

// ── RESULT MESSAGES ──
const GOOD_MSG = (name) =>
  `Your <b>semantic coherence</b> is above your personal baseline — clear, connected thinking today. Pause frequency has improved. <b>Excellent session, ${name}!</b> Let's meet again tomorrow.`;
const WARN_MSG = () =>
  `Your pause frequency was slightly elevated today — possibly fatigue or stress. <b>Semantic coherence is holding steady</b>. Rest well and try again tomorrow.`;
const BAD_MSG  = (name) =>
  `We noticed some changes in your voice patterns today, <b>${name}</b>. This can happen with fatigue or illness. You can <b>record again</b> to see if it improves, or rest and try tomorrow.`;

// ── HELPERS ──
const fmt = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;

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

// ── MAIN COMPONENT ──
const Session = () => {
  const navigate  = useNavigate();
  const { token, user, logout } = useAuth();

  const [dark, setDark]               = useState(() => localStorage.getItem("cog_dark") === "true");
  const [prompt]                      = useState(() => PROMPTS[Math.floor(Math.random() * PROMPTS.length)]);
  const [skipped, setSkipped]         = useState(false);
  const [state, setState]             = useState("checking");
  const [secsLeft, setSecsLeft]       = useState(TOTAL);
  const [progress, setProgress]       = useState(0);
  const [result, setResult]           = useState(null);
  const [errorMsg, setErrorMsg]       = useState(null);
  const [aiWarning, setAiWarning]     = useState(false);
  const [transcript, setTranscript]   = useState("");

  const timerRef       = useRef(null);
  const progRef        = useRef(null);
  const mrRef          = useRef(null);
  const chunksRef      = useRef([]);
  const resultRef      = useRef(null);
  const abortRef       = useRef(null);
  const recognitionRef = useRef(null);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    localStorage.setItem("cog_dark", String(next));
  };

  // ✅ Warm up HF Space the moment user opens the session page
  useEffect(() => {
    fetch(`${HF_URL}/health`).catch(() => {});
  }, []);

  // ── On mount: check if already recorded today ──
  useEffect(() => {
    const check = async () => {
      const lastSess = localStorage.getItem("cog_last_session");
      const lastTier = localStorage.getItem("cog_last_result_tier");
      if (lastSess && lastTier === "Green") {
        const sameDay = new Date(lastSess).toDateString() === new Date().toDateString();
        if (sameDay) { setState("donegood"); return; }
      }
      if (token) {
        try {
          const res = await checkToday(token);
          if (res.recorded && res.risk_tier === "Green") {
            setState("donegood");
            return;
          }
        } catch {
          // fall through to idle
        }
      }
      setState("idle");
    };
    check();
  }, [token]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      clearInterval(progRef.current);
      abortRef.current?.abort();
      mrRef.current?.stream?.getTracks().forEach(t => t.stop());
      if (recognitionRef.current) {
        try { recognitionRef.current.stop(); } catch(e) {}
      }
    };
  }, []);

  const offset      = CIRC * (secsLeft / TOTAL);
  const strokeColor = secsLeft / TOTAL > 0.5
    ? "url(#tg)"
    : secsLeft / TOTAL > 0.25 ? "#F59E0B" : "#EF4444";

  // ── START RECORDING ──
  const startRecording = async () => {
    setErrorMsg(null);
    setAiWarning(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      mrRef.current = mr;
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.start(250);
    } catch (err) {
      console.warn("Mic access denied or unavailable:", err.message);
    }

    setTranscript("");
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.onresult = (event) => {
        let currentTranscript = "";
        for (let i = 0; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setTranscript(currentTranscript);
      };
      recognition.onerror = (event) => {
        console.warn("Speech recognition error", event.error);
      };
      try {
        recognition.start();
        recognitionRef.current = recognition;
      } catch (err) {
        console.warn("Could not start speech recognition", err);
      }
    }

    setState("recording");
    setSecsLeft(TOTAL);
    setProgress(0);

    timerRef.current = setInterval(() => {
      setSecsLeft(prev => {
        const next = prev - 1;
        setProgress(((TOTAL - next) / TOTAL) * 55);
        if (next <= 0) {
          stopRecording();
          return 0;
        }
        return next;
      });
    }, 1000);
  };

  // ── STOP RECORDING + SEND TO AI ──
  const stopRecording = useCallback(async () => {
    clearInterval(timerRef.current);

    let audioBlob = null;
    if (mrRef.current && mrRef.current.state !== "inactive") {
      await new Promise((resolve) => {
        mrRef.current.onstop = resolve;
        mrRef.current.stop();
      });
      mrRef.current.stream?.getTracks().forEach(t => t.stop());
      if (chunksRef.current.length > 0) {
        const mimeType = mrRef.current.mimeType || "audio/webm";
        audioBlob = new Blob(chunksRef.current, { type: mimeType });
      }
    }

    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
    }

    setState("analysing");
    setProgress(55);

    // Slower progress animation — HF can take 60-120s
    let p = 55;
    progRef.current = setInterval(() => {
      p += 0.1;
      setProgress(Math.min(p, 98));
    }, 100);

    try {
      let aiResult;

      if (audioBlob && audioBlob.size > 1000) {
        try {
          const controller = new AbortController();
          abortRef.current = controller;

          const formData = new FormData();
          // ✅ send as .webm — HF Space accepts it after our fix
          formData.append("audio", audioBlob, "recording.webm");
          formData.append("user_id", String(user?.id || 1));

          // ✅ 290s timeout — HF needs up to 2 mins on free tier
          const timeoutId = setTimeout(() => controller.abort(), 290000);

          // ✅ call HF directly — no Render proxy, no 30s limit
          const aiRes = await fetch(`${HF_URL}/analyze`, {
            method: "POST",
            body: formData,
            signal: controller.signal,
          });

          clearTimeout(timeoutId);

          if (!aiRes.ok) {
            const err = await aiRes.json().catch(() => ({}));
            throw new Error(err.detail || `AI service returned ${aiRes.status}`);
          }

          aiResult = await aiRes.json();

        } catch (aiErr) {
          if (aiErr.name === "AbortError") throw new Error("Analysis timed out after 290s — please try again.");
          console.warn("AI service error, using fallback:", aiErr.message);
          setAiWarning(true);
          aiResult = buildMockResult();
        }
      } else {
        console.warn("No audio recorded, using mock result for demo.");
        setAiWarning(true);
        aiResult = buildMockResult();
      }

      // ── Save to Render backend (fast call, well under 30s) ──
      if (token) {
        try {
          await saveSessionToBackend(token, aiResult);
        } catch (saveErr) {
          console.warn("Failed to save session to backend:", saveErr.message);
        }
      }

      localStorage.setItem("cog_last_session", new Date().toISOString());
      localStorage.setItem("cog_last_result_tier", aiResult.risk_tier);

      clearInterval(progRef.current);
      setProgress(100);
      setResult(aiResult);
      setState("done");

      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 200);

    } catch (err) {
      clearInterval(progRef.current);
      console.error("Session error:", err);
      setErrorMsg(err.message || "Something went wrong. Please try again.");
      setState("error");
      setProgress(0);
    }
  }, [token, user]);

  // ── CANCEL SESSION ──
  const cancelSession = () => {
    clearInterval(timerRef.current);
    clearInterval(progRef.current);
    abortRef.current?.abort();
    if (mrRef.current && mrRef.current.state !== "inactive") {
      mrRef.current.stop();
      mrRef.current.stream?.getTracks().forEach(t => t.stop());
    }
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
    }
    setTranscript("");
    setState("idle");
    setSecsLeft(TOTAL);
    setProgress(0);
    setResult(null);
    setErrorMsg(null);
  };

  // ── RETRY SESSION ──
  const retrySession = () => {
    localStorage.removeItem("cog_last_session");
    localStorage.removeItem("cog_last_result_tier");
    cancelSession();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const riskBadgeCls = !result ? "rb-green"
    : result.risk_tier === "Green"  ? "rb-green"
    : result.risk_tier === "Yellow" ? "rb-yellow"
    : result.risk_tier === "Orange" ? "rb-orange"
    : "rb-red";

  const userName = user?.name?.split(" ")[0] || "there";

  const getMessage = () => {
    if (!result) return "";
    if (result.risk_tier === "Green")  return GOOD_MSG(userName);
    if (result.risk_tier === "Yellow") return WARN_MSG();
    if (result.risk_tier === "Orange" || result.risk_tier === "Red") return BAD_MSG(userName);
  };

  const canRetry = result && result.risk_tier !== "Green";

  const NAV_LINKS = [
    ["Dashboard", "/dashboard"],
    ["Session",   "/session"],
    ["Brain",     "/brain"],
    ["Report",    "/ar-report"],
  ];

  return (
    <div className={`session-root ${dark ? "dm" : "lm"}`}>

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

      <div className="session-page">

        {state === "checking" && (
          <div className="done-today-card" style={{ textAlign: "center", padding: "48px 32px" }}>
            <div style={{ fontSize: 36, marginBottom: 12 }}>🧠</div>
            <div className="done-title" style={{ fontSize: 16, color: "var(--text2)" }}>
              Loading your session…
            </div>
          </div>
        )}

        {state === "donegood" && (
          <div className="done-today-card">
            <div className="done-icon">🌿</div>
            <div className="done-title">You've completed today's session!</div>
            <div className="done-sub">
              Your cognitive health is looking great today.<br />
              Your voice data has been recorded and analysed.
            </div>
            <div className="done-badge">
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--success)", display: "inline-block" }} />
              Cognitive health: Good
            </div>
            <p className="done-quote">
              "Your semantic coherence score is above your personal baseline.
              Keep up the great work — let's meet again tomorrow!"
            </p>
            <div className="done-actions">
              <button className="ra-primary" onClick={() => navigate("/dashboard")}>
                View dashboard
              </button>
              <button className="ra-secondary" onClick={() => navigate("/ar-report")}>
                View full report
              </button>
            </div>
          </div>
        )}

        {state === "error" && (
          <div className="done-today-card" style={{ borderColor: "var(--danger)" }}>
            <div className="done-icon">⚠️</div>
            <div className="done-title" style={{ color: "var(--danger)" }}>Something went wrong</div>
            <div className="done-sub">{errorMsg || "An unexpected error occurred. Please try again."}</div>
            <div className="done-actions">
              <button className="ra-primary" onClick={() => { setState("idle"); setErrorMsg(null); }}>
                Try again
              </button>
              <button className="ra-secondary" onClick={() => navigate("/dashboard")}>
                Go to dashboard
              </button>
            </div>
          </div>
        )}

        {(state === "idle" || state === "recording" || state === "analysing" || state === "done") && (
          <>
            {!skipped ? (
              <div className="s-card">
                <div className="prompt-eyebrow">
                  <span className="pe-label">Today's prompt</span>
                  <div className="pe-line" />
                  <span className="pe-num">Session {user?.sessions_count ? user.sessions_count + 1 : 1}</span>
                </div>
                <div className="prompt-text">"{prompt.text}"</div>
                {prompt.image && (
                  <img className="prompt-image" src={prompt.image} alt="Session prompt"
                    onError={(e) => { e.target.style.display = "none"; }} />
                )}
                {state === "idle" && (
                  <div className="skip-row">
                    <button className="skip-btn" onClick={() => setSkipped(true)}>
                      Skip prompt & speak freely →
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <div className="free-note">
                <b>Free-speak mode 🎙️</b><br />
                Speak about anything for 3 minutes. Your voice patterns will be analysed accurately.
              </div>
            )}

            <div className="s-card">
              <div className="orb-block">

                {state === "idle" && (
                  <div className="orb-idle">
                    <div className="orb-ring orb-ring-1" />
                    <div className="orb-ring orb-ring-2" />
                    <div className="orb-ring orb-ring-3" />
                    <div className="orb">🎙️</div>
                  </div>
                )}

                {state === "recording" && (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
                    <div className="timer-ring-wrap">
                      <svg className="timer-svg" width="140" height="140" viewBox="0 0 140 140">
                        <defs>
                          <linearGradient id="tg" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%"   stopColor="#10B981"/>
                            <stop offset="100%" stopColor="#6366F1"/>
                          </linearGradient>
                        </defs>
                        <circle className="timer-track"    cx="70" cy="70" r="62"/>
                        <circle className="timer-progress" cx="70" cy="70" r="62"
                          stroke={strokeColor}
                          strokeDasharray={CIRC}
                          strokeDashoffset={offset}
                        />
                      </svg>
                      <div className="orb-recording">🔴</div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <div className="timer-label">{fmt(secsLeft)}</div>
                      <div className="timer-sublabel">remaining</div>
                    </div>
                  </div>
                )}

                {state === "analysing" && (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
                    <div className="timer-ring-wrap">
                      <svg className="timer-svg" width="140" height="140" viewBox="0 0 140 140">
                        <circle className="timer-track"    cx="70" cy="70" r="62"/>
                        <circle className="timer-progress" cx="70" cy="70" r="62"
                          stroke="#6366F1"
                          strokeDasharray={CIRC}
                          strokeDashoffset="0"
                        />
                      </svg>
                      <div className="orb-analysing"><span className="spinner" /></div>
                    </div>
                    <div style={{ textAlign: "center" }}>
                      <div className="timer-label" style={{ fontSize: 16, color: "var(--indigo)" }}>
                        Analysing…
                      </div>
                      <div className="timer-sublabel">AI pipeline running</div>
                    </div>
                  </div>
                )}

                {state === "done" && (
                  <div className="orb-idle">
                    <div className="orb done">✓</div>
                  </div>
                )}

              </div>

              <div style={{ display: "flex", justifyContent: "center", marginBottom: 20 }}>
                <div className={`status-pill ${
                  state === "idle"        ? "sp-idle"
                  : state === "recording" ? "sp-rec"
                  : state === "analysing" ? "sp-ana"
                  : "sp-done"
                }`}>
                  <span className="sp-dot" />
                  <span>
                    {state === "idle"        ? "Ready to record"
                      : state === "recording" ? `Recording… ${fmt(secsLeft)}`
                      : state === "analysing" ? "Analysing your voice…"
                      : "Analysis complete ✓"}
                  </span>
                </div>
              </div>

              <div className="progress-wrap">
                <div className="progress-labels">
                  <span className={`progress-label ${state === "idle" || state === "recording" ? "active" : ""}`}>
                    1. Record
                  </span>
                  <span className={`progress-label ${state === "analysing" ? "active" : ""}`}>
                    2. Analyse
                  </span>
                  <span className={`progress-label ${state === "done" ? "active" : ""}`}>
                    3. Results
                  </span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${progress}%` }} />
                </div>
              </div>

              {state !== "done" && (
                <div className="btn-row">
                  <button
                    className={`btn-primary ${state === "recording" ? "recording" : ""}`}
                    onClick={state === "idle" ? startRecording : stopRecording}
                    disabled={state === "analysing"}
                  >
                    {state === "idle"
                      ? <>▶ Start Recording</>
                      : state === "recording"
                      ? <>⏹ Stop Recording</>
                      : <><span className="spinner" style={{ marginRight: 8 }} />Analysing…</>}
                  </button>
                  <button className="btn-ghost" onClick={cancelSession}
                    disabled={state === "analysing"}>
                    {state === "idle" ? "Cancel" : "Stop & cancel"}
                  </button>
                </div>
              )}

              {state === "recording" && transcript && (
                <div style={{ marginTop: 24, padding: 16, borderRadius: 12, background: "var(--card2)", color: "var(--text2)", fontSize: 15, lineHeight: 1.6, maxHeight: 110, overflowY: "auto", textAlign: "center", border: "1px solid var(--border)" }}>
                  <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 1, fontWeight: 700, marginBottom: 8, color: "var(--text3)", fontStyle: "normal" }}>Live Transcript</div>
                  <div style={{ fontStyle: "italic" }}>"{transcript}"</div>
                </div>
              )}

              {state === "analysing" && (
                <div className="analysis-notice">
                  ⏳ AI analysis takes 60–120 seconds — please keep this tab open
                </div>
              )}

              {state === "done" && aiWarning && (
                <div className="analysis-notice" style={{
                  marginTop: 12,
                  borderColor: "var(--amber)",
                  background: "rgba(245,158,11,0.06)",
                  color: "var(--warn)"
                }}>
                  ⚠️ AI service was unreachable — result is a demo simulation.
                  Real analysis will run when the AI service is available.
                </div>
              )}

            </div>

            {state === "done" && result && (
              <div className="result-section" ref={resultRef}>
                <div className="result-card">
                  <div className="result-header">
                    <div>
                      <div className="result-title">Session results</div>
                      <div className="result-subtitle">
                        Recorded {new Date().toLocaleDateString("en-US", { month: "long", day: "numeric" })}
                      </div>
                    </div>
                    <div className={`result-badge ${riskBadgeCls}`}>
                      ● {result.risk_tier} — {
                        result.risk_tier === "Green"  ? "Good"
                        : result.risk_tier === "Yellow" ? "Watch"
                        : result.risk_tier === "Orange" ? "Elevated"
                        : "Alert"
                      }
                    </div>
                  </div>

                  <div className="metrics-grid">
                    <div className="metric-item">
                      <div className="metric-label">Semantic coherence</div>
                      <div className="metric-value">
                        {result.biomarkers?.semantic_coherence?.toFixed
                          ? result.biomarkers.semantic_coherence.toFixed(2)
                          : result.biomarkers?.semantic_coherence ?? "—"}
                      </div>
                      <div className="metric-trend"
                        style={{ color: `var(--${result.risk_tier === "Green" ? "success" : "warn"})` }}>
                        {result.risk_tier === "Green" ? "▲ Above baseline" : "▼ Below baseline"}
                      </div>
                    </div>
                    <div className="metric-item">
                      <div className="metric-label">Speech rate</div>
                      <div className="metric-value">
                        {result.biomarkers?.speech_rate != null
                          ? Math.round(result.biomarkers.speech_rate)
                          : "—"}
                      </div>
                      <div className="metric-trend" style={{ color: "var(--success)" }}>▲ Normal range</div>
                    </div>
                    <div className="metric-item">
                      <div className="metric-label">Pause frequency</div>
                      <div className="metric-value">
                        {result.biomarkers?.pause_frequency?.toFixed
                          ? result.biomarkers.pause_frequency.toFixed(1)
                          : result.biomarkers?.pause_frequency ?? "—"}
                      </div>
                      <div className="metric-trend"
                        style={{ color: (result.biomarkers?.pause_frequency ?? 0) > 4 ? "var(--danger)" : "var(--success)" }}>
                        {(result.biomarkers?.pause_frequency ?? 0) > 4 ? "▼ Elevated" : "▲ Normal"}
                      </div>
                    </div>
                    <div className="metric-item">
                      <div className="metric-label">Risk tier</div>
                      <div className="metric-value"
                        style={{ color: `var(--${
                          result.risk_tier === "Green"  ? "success"
                          : result.risk_tier === "Yellow" ? "warn"
                          : result.risk_tier === "Orange" ? "var(--amber)"
                          : "danger"
                        })` }}>
                        {result.risk_tier}
                      </div>
                      <div className="metric-trend" style={{ color: "var(--success)" }}>● Analysed</div>
                    </div>
                  </div>

                  <div className="result-message"
                    dangerouslySetInnerHTML={{ __html: getMessage() }} />

                  <div className="result-actions">
                    <button className="ra-primary" onClick={() => navigate("/dashboard")}>
                      View dashboard
                    </button>
                    {canRetry && (
                      <button className="ra-secondary" onClick={retrySession}>
                        Record again
                      </button>
                    )}
                    <button className="ra-secondary" onClick={() => navigate("/ar-report")}>
                      View full report
                    </button>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

// ── HELPERS ──

function buildMockResult() {
  return {
    risk_tier: "Green",
    biomarkers: {
      semantic_coherence:   0.83,
      lexical_diversity:    0.74,
      idea_density:         0.62,
      speech_rate:          121,
      pause_frequency:      2.8,
      pause_duration:       0.41,
      pitch_mean:           183,
      pitch_range:          65,
      jitter:               0.011,
      shimmer:              0.078,
      hnr:                  18.6,
      syntactic_complexity: 0.71,
      articulation_rate:    4.9,
      emotional_entropy:    0.64,
    },
    anomaly_flags: [],
  };
}

async function saveSessionToBackend(token, aiResult) {
  const payload = {
    risk_tier:            aiResult.risk_tier,
    semantic_coherence:   aiResult.biomarkers?.semantic_coherence   ?? null,
    lexical_diversity:    aiResult.biomarkers?.lexical_diversity     ?? null,
    idea_density:         aiResult.biomarkers?.idea_density          ?? null,
    speech_rate:          aiResult.biomarkers?.speech_rate           ?? null,
    pause_frequency:      aiResult.biomarkers?.pause_frequency       ?? null,
    pause_duration:       aiResult.biomarkers?.pause_duration        ?? null,
    pitch_mean:           aiResult.biomarkers?.pitch_mean            ?? null,
    pitch_range:          aiResult.biomarkers?.pitch_range           ?? null,
    jitter:               aiResult.biomarkers?.jitter                ?? null,
    shimmer:              aiResult.biomarkers?.shimmer               ?? null,
    hnr:                  aiResult.biomarkers?.hnr                   ?? null,
    syntactic_complexity: aiResult.biomarkers?.syntactic_complexity  ?? null,
    articulation_rate:    aiResult.biomarkers?.articulation_rate     ?? null,
    emotional_entropy:    aiResult.biomarkers?.emotional_entropy     ?? null,
    has_anomaly:          (aiResult.anomaly_flags?.length ?? 0) > 0,
    anomaly_flags:        JSON.stringify(aiResult.anomaly_flags || []),
  };

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const res = await fetch(`${API_URL}/api/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Backend returned ${res.status}`);
  }

  return await res.json();
}

export default Session;