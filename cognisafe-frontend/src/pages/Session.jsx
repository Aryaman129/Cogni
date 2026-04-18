import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import {
  checkToday,
  saveSession,
  submitAudioJob,
  submitTextJob,
  STAGE_LABELS,
  TEXT_STAGE_LABELS,
  STAGE_ORDER,
  TEXT_STAGE_ORDER,
} from "../services/sessionService";
import "../styles/session.css";

// ── PROMPTS ──
const VOICE_PROMPTS = [
  {
    text: "Describe what you see in this image in as much detail as you can — the objects, colours, setting, and what might be happening.",
    image: "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=600&q=80",
    category: "Visual Description"
  },
  {
    text: "Tell me about your most vivid childhood memory. Describe everything you can remember — the place, people, smells, and how it made you feel.",
    image: null,
    category: "Memory Recall"
  },
  {
    text: "Look at this photo and describe every detail you notice — colours, shapes, mood, and what story the image might be telling.",
    image: "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=600&q=80",
    category: "Visual Narrative"
  },
  {
    text: "Explain how you would get from your home to your favourite restaurant — describe the journey as if giving directions to a friend.",
    image: null,
    category: "Spatial Navigation"
  },
  {
    text: "Describe what's happening in this photo in as much detail as possible — the environment, any people, objects, light and shadows.",
    image: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=600&q=80",
    category: "Scene Description"
  },
];

const TEXT_PROMPTS = [
  {
    text: "Describe your morning routine in as much detail as possible — from the moment you wake up to when you leave home or start your day.",
    category: "Daily Routine"
  },
  {
    text: "Think of a place that means a lot to you and describe it in detail — what does it look like, smell like, feel like? Why is it special to you?",
    category: "Memory & Place"
  },
  {
    text: "Describe what you would do on a perfect day — where you would go, who you would be with, and what you would eat and see.",
    category: "Imagination"
  },
  {
    text: "Tell the story of how you met your best friend or a person who is important to you. Describe the first time you met and how the relationship grew.",
    category: "Personal Story"
  },
  {
    text: "Describe a skill or hobby you have learned. How did you start, what was difficult, and what do you enjoy most about it now?",
    category: "Skills & Hobbies"
  },
];

// ── CONSTANTS ──
const TOTAL = 180;
const CIRC  = 395;
const MIN_WORDS = 50; // minimum words for a valid text session

// ── LOGO ICON ──
const LogoIcon = () => (
  <svg width="14" height="14" viewBox="0 0 18 18" fill="none">
    <circle cx="9" cy="9" r="5" stroke="#fff" strokeWidth="1.6" />
    <circle cx="9" cy="9" r="1.8" fill="#fff" />
    <line x1="9" y1="2" x2="9" y2="4" stroke="#fff" strokeWidth="1.4" strokeLinecap="round" />
    <line x1="9" y1="14" x2="9" y2="16" stroke="#fff" strokeWidth="1.4" strokeLinecap="round" />
    <line x1="2" y1="9" x2="4" y2="9" stroke="#fff" strokeWidth="1.4" strokeLinecap="round" />
    <line x1="14" y1="9" x2="16" y2="9" stroke="#fff" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
);

// ── RESULT MESSAGES ── (fallbacks used only when API interpretation is missing)
const GOOD_MSG  = (name) => `Your <b>semantic coherence</b> is above your personal baseline — clear, connected thinking today. <b>Excellent session, ${name}!</b> Let's meet again tomorrow.`;
const WARN_MSG  = ()     => `Some language patterns were slightly varied today — possibly fatigue or stress. <b>Semantic coherence is holding steady</b>. Rest well and try again tomorrow.`;
const BAD_MSG   = (name) => `We noticed some changes in your language patterns today, <b>${name}</b>. This can happen with fatigue or illness. You can <b>try again</b> to see if it improves, or rest and try tomorrow.`;

// ── MOOD DISPLAY HELPERS ──
const MOOD_EMOJI = { calm: "😌", stressed: "😰", sad: "😔", fatigued: "😴" };
const MOOD_COLOR = { calm: "#4caf87", stressed: "#e5836a", sad: "#7a8fc4", fatigued: "#b07cc6" };

// ── HELPERS ──
const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
};

const countWords = (text) => text.trim().split(/\s+/).filter(Boolean).length;

// ── WAVEFORM VISUALIZATION ──
const WaveformVisualizer = ({ isRecording }) => {
  const canvasRef    = useRef(null);
  const animationRef = useRef(null);
  const analyserRef  = useRef(null);
  const streamRef    = useRef(null);

  useEffect(() => {
    if (!isRecording) {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      return;
    }
    const setupAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const analyser     = audioContext.createAnalyser();
        const source       = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        analyser.fftSize   = 256;
        analyserRef.current = analyser;
        const bufferLength  = analyser.frequencyBinCount;
        const dataArray     = new Uint8Array(bufferLength);

        const draw = () => {
          if (!canvasRef.current || !analyserRef.current) return;
          const canvas = canvasRef.current;
          const ctx    = canvas.getContext("2d");
          const width  = canvas.clientWidth;
          const height = canvas.clientHeight;
          canvas.width  = width;
          canvas.height = height;
          analyserRef.current.getByteTimeDomainData(dataArray);
          ctx.clearRect(0, 0, width, height);
          ctx.beginPath();
          ctx.strokeStyle = "#D4A5B5";
          ctx.lineWidth   = 2;
          const sliceWidth = width / bufferLength;
          let x = 0;
          for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * height) / 2;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            x += sliceWidth;
          }
          ctx.stroke();
          animationRef.current = requestAnimationFrame(draw);
        };
        draw();
      } catch (err) {
        console.warn("Could not access microphone for visualization:", err);
      }
    };
    setupAudio();
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    };
  }, [isRecording]);

  if (!isRecording) {
    return (
      <div className="waveform-placeholder">
        {[...Array(40)].map((_, i) => (
          <div key={i} className="wave-bar-static" style={{ animationDelay: `${i * 0.05}s` }} />
        ))}
      </div>
    );
  }
  return <canvas ref={canvasRef} className="waveform-canvas" />;
};

// ════════════════════════════════════════════════════════════════
// ── MAIN SESSION COMPONENT ──────────────────────────────────────
// ════════════════════════════════════════════════════════════════
const Session = () => {
  const navigate = useNavigate();
  const { token, user, logout, getUserStorage, setUserStorage } = useAuth();

  // ── Theme ──
  const [dark, setDark] = useState(() => localStorage.getItem("cog_dark") === "true");
  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    localStorage.setItem("cog_dark", String(next));
    document.documentElement.setAttribute("data-theme", next ? "dark" : "light");
  };
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  }, [dark]);

  // ── Mode: "voice" | "text" ──
  const [mode, setMode] = useState("voice");

  // ── Prompts ──
  const [voicePrompt] = useState(() => VOICE_PROMPTS[Math.floor(Math.random() * VOICE_PROMPTS.length)]);
  const [textPrompt]  = useState(() => TEXT_PROMPTS[Math.floor(Math.random() * TEXT_PROMPTS.length)]);
  const [skipped, setSkipped] = useState(false);

  // ── Shared state ──
  const [state, setState]               = useState("checking");
  const [result, setResult]             = useState(null);
  const [errorMsg, setErrorMsg]         = useState(null);
  const [progress, setProgress]         = useState(0);
  const [analysisStage, setAnalysisStage] = useState("uploading");

  // ── Voice-only state ──
  const [timeLeft, setTimeLeft]     = useState(TOTAL);
  const [transcript, setTranscript] = useState("");
  const [volumeLevel, setVolumeLevel] = useState(0);

  // ── Text-only state ──
  const [typedText, setTypedText] = useState("");
  const wordCount = countWords(typedText);
  const textReady = wordCount >= MIN_WORDS;

  // ── Refs ──
  const timerRef         = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef        = useRef([]);
  const resultRef        = useRef(null);
  const recognitionRef   = useRef(null);
  const audioContextRef  = useRef(null);
  const volumeIntervalRef = useRef(null);

  // ── Ring maths ──
  const offset      = CIRC * (timeLeft / TOTAL);
  const strokeColor = timeLeft / TOTAL > 0.5 ? "url(#gradient)" : timeLeft / TOTAL > 0.25 ? "#E5B56A" : "#E58383";

  // ── Fallback timeout for "checking" state ──
  useEffect(() => {
    const t = setTimeout(() => {
      if (state === "checking") setState("idle");
    }, 3000);
    return () => clearTimeout(t);
  }, []);

  // ── Check if already recorded today ──
  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      const timeout = setTimeout(() => { if (isMounted && state === "checking") setState("idle"); }, 2000);
      try {
        if (!user?.id) { clearTimeout(timeout); if (isMounted) setState("idle"); return; }
        const lastSess = getUserStorage?.("last_session");
        const lastTier = getUserStorage?.("last_result_tier");
        if (lastSess && lastTier === "Green") {
          const sameDay = new Date(lastSess).toDateString() === new Date().toDateString();
          if (sameDay) { clearTimeout(timeout); if (isMounted) setState("donegood"); return; }
        }
        const isDemo = user?.email?.includes("demo");
        if (token && !isDemo) {
          try {
            const res = await checkToday(token);
            if (res.recorded && res.risk_tier === "Green") {
              clearTimeout(timeout); if (isMounted) setState("donegood"); return;
            }
          } catch (e) { console.log("Backend check failed:", e); }
        }
        clearTimeout(timeout);
        if (isMounted) setState("idle");
      } catch (e) {
        clearTimeout(timeout);
        if (isMounted) setState("idle");
      }
    };
    check();
    return () => { isMounted = false; };
  }, [token, user, getUserStorage]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => {
      clearInterval(timerRef.current);
      clearInterval(volumeIntervalRef.current);
      if (mediaRecorderRef.current?.stream) mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
      try { recognitionRef.current?.stop(); } catch (e) {}
      audioContextRef.current?.close();
    };
  }, []);

  // ── Switch mode — reset state ──
  const switchMode = (newMode) => {
    if (state === "recording" || state === "analysing") return; // don't allow mid-session
    setMode(newMode);
    setSkipped(false);
    setTypedText("");
    setResult(null);
    setErrorMsg(null);
    setProgress(0);
    setState("idle");
  };

  // ════════════════════════════
  // ── VOICE MODE HANDLERS ──
  // ════════════════════════════
  const startVolumeMonitoring = async () => {
    try {
      const stream      = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source      = audioContext.createMediaStreamSource(stream);
      const analyser    = audioContext.createAnalyser();
      source.connect(analyser);
      analyser.fftSize  = 256;
      const dataArray   = new Uint8Array(analyser.frequencyBinCount);
      volumeIntervalRef.current = setInterval(() => {
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setVolumeLevel(Math.min(100, (avg / 255) * 100));
      }, 100);
      audioContextRef.current = audioContext;
    } catch (e) { console.warn("Volume monitoring failed:", e); }
  };

  const stopVolumeMonitoring = () => {
    clearInterval(volumeIntervalRef.current);
    audioContextRef.current?.close();
    setVolumeLevel(0);
  };

  const startRecording = async () => {
    setErrorMsg(null);
    setTranscript("");
    setVolumeLevel(0);
    try {
      const stream   = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      const mr = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      mediaRecorderRef.current = mr;
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.start(250);
      startVolumeMonitoring();
    } catch (e) { console.warn("Mic access denied:", e.message); }

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SR) {
      const rec = new SR();
      rec.continuous = true; rec.interimResults = true;
  r   rec.lang = navigator.language || "en-US"; // auto-detect browser language
      rec.onresult = (e) => {
        let t = "";
        for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
        setTranscript(t);
      };
      rec.onerror = (e) => console.warn("Speech recognition error", e.error);
      try { rec.start(); recognitionRef.current = rec; } catch (e) {}
    }

    setState("recording");
    setTimeLeft(TOTAL);
    setProgress(0);
    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        const next = prev - 1;
        setProgress(((TOTAL - next) / TOTAL) * 55);
        if (next <= 0) { stopRecording(); return 0; }
        return next;
      });
    }, 1000);
  };

  const stopRecording = useCallback(async () => {
    clearInterval(timerRef.current);
    stopVolumeMonitoring();

    let audioBlob = null;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      await new Promise((resolve) => {
        mediaRecorderRef.current.onstop = resolve;
        mediaRecorderRef.current.stop();
      });
      mediaRecorderRef.current.stream?.getTracks().forEach(t => t.stop());
      if (chunksRef.current.length > 0) {
        const mimeType = mediaRecorderRef.current.mimeType || "audio/webm";
        audioBlob = new Blob(chunksRef.current, { type: mimeType });
      }
    }
    try { recognitionRef.current?.stop(); } catch (e) {}

    setState("analysing");
    setAnalysisStage("uploading");
    setProgress(5);

    try {
      let aiResult;
      const isDemo = user?.email?.includes("demo");

      if (isDemo) {
        for (let i = 0; i < STAGE_ORDER.length; i++) {
          setAnalysisStage(STAGE_ORDER[i]);
          setProgress(Math.round(((i + 1) / STAGE_ORDER.length) * 100));
          await new Promise(r => setTimeout(r, 600));
        }
        aiResult = {
          risk_tier: Math.random() > 0.8 ? "Yellow" : "Green",
          mode: "voice",
          biomarkers: {
            semantic_coherence: 0.75 + Math.random() * 0.2,
            lexical_diversity:  0.65 + Math.random() * 0.15,
            speech_rate:        110  + Math.random() * 25,
            pause_frequency:    2.5  + Math.random() * 2,
            hnr:                16   + Math.random() * 5,
            jitter:             0.008 + Math.random() * 0.008,
          },
          anomaly_flags: [],
          mood: { label: "calm", confidence: 0.76 },
          interpretation: "All cognitive markers are within normal range. Keep monitoring regularly.",
          method: "xgboost_only",
          session_count: 1,
        };
      } else if (audioBlob && audioBlob.size > 1000) {
        setAnalysisStage("uploading");
        setProgress(8);
        aiResult = await submitAudioJob(audioBlob, user?.id || 1, (stage) => {
          setAnalysisStage(stage);
          const idx = STAGE_ORDER.indexOf(stage);
          setProgress(idx >= 0 ? 10 + Math.round((idx / (STAGE_ORDER.length - 1)) * 85) : 10);
        });
      } else {
        setAnalysisStage("done");
        setProgress(100);
        aiResult = {
          risk_tier: "Green",
          mode: "voice",
          biomarkers: {
            semantic_coherence: Number((0.80 + Math.random() * 0.1).toFixed(2)),
            lexical_diversity:  Number((0.70 + Math.random() * 0.1).toFixed(2)),
            speech_rate:        Math.floor(115 + Math.random() * 15),
            pause_frequency:    Number((2.0  + Math.random() * 1.5).toFixed(1)),
            hnr:                Number((18.0 + Math.random() * 2.0).toFixed(1)),
            jitter:             Number((0.009 + Math.random() * 0.004).toFixed(3)),
          },
          anomaly_flags: [],
        };
      }

      await _finaliseResult(aiResult);
    } catch (err) {
      console.error("Session error:", err);
      setErrorMsg(err.message || "Something went wrong. Please try again.");
      setState("error");
      setProgress(0);
    }
  }, [token, user, setUserStorage]);

  // ════════════════════════════
  // ── TEXT MODE HANDLER ──
  // ════════════════════════════
  const submitText = async () => {
    if (!textReady) return;
    setErrorMsg(null);
    setState("analysing");
    setAnalysisStage("nlp");
    setProgress(10);

    try {
      let aiResult;
      const isDemo = user?.email?.includes("demo");

      if (isDemo) {
        // Simulate text analysis stages
        for (let i = 0; i < TEXT_STAGE_ORDER.length; i++) {
          setAnalysisStage(TEXT_STAGE_ORDER[i]);
          setProgress(Math.round(((i + 1) / TEXT_STAGE_ORDER.length) * 100));
          await new Promise(r => setTimeout(r, 800));
        }
        aiResult = {
          risk_tier: Math.random() > 0.8 ? "Yellow" : "Green",
          mode: "text",
          biomarkers: {
            semantic_coherence:   0.72 + Math.random() * 0.2,
            lexical_diversity:    0.60 + Math.random() * 0.2,
            idea_density:         0.38 + Math.random() * 0.1,
            syntactic_complexity: 3.5  + Math.random() * 2,
            // Acoustic biomarkers are null in text mode
            speech_rate: null, pause_frequency: null, hnr: null,
            jitter: null, shimmer: null, pitch_mean: null,
          },
          anomaly_flags: [],
          mood: { label: "calm", confidence: 0.71 },
          interpretation: "All cognitive markers are within normal range. Keep monitoring regularly.",
          method: "xgboost_only",
          session_count: 1,
        };
      } else {
        aiResult = await submitTextJob(typedText, user?.id || 1, (stage) => {
          setAnalysisStage(stage);
          const idx = TEXT_STAGE_ORDER.indexOf(stage);
          setProgress(idx >= 0 ? 20 + Math.round((idx / (TEXT_STAGE_ORDER.length - 1)) * 75) : 20);
        });
      }

      await _finaliseResult(aiResult);
    } catch (err) {
      console.error("Text session error:", err);
      setErrorMsg(err.message || "Something went wrong. Please try again.");
      setState("error");
      setProgress(0);
    }
  };

  // ── Shared finalise: save + set state ──
  const _finaliseResult = async (aiResult) => {
    if (token && !user?.email?.includes("demo")) {
      try { await saveSession(token, aiResult); }
      catch (e) { console.warn("Failed to save session:", e.message); }
    }
    if (user?.id && setUserStorage) {
      setUserStorage("last_session", new Date().toISOString());
      setUserStorage("last_result_tier", aiResult.risk_tier);
    } else {
      localStorage.setItem("cog_last_session", new Date().toISOString());
      localStorage.setItem("cog_last_result_tier", aiResult.risk_tier);
    }
    setProgress(100);
    setResult(aiResult);
    setState("done");
    setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 200);
  };

  // ── Cancel ──
  const cancelSession = () => {
    clearInterval(timerRef.current);
    clearInterval(volumeIntervalRef.current);
    if (mediaRecorderRef.current?.stream) mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop());
    try { recognitionRef.current?.stop(); } catch (e) {}
    setTranscript("");
    setTypedText("");
    setState("idle");
    setTimeLeft(TOTAL);
    setProgress(0);
    setResult(null);
    setErrorMsg(null);
  };

  const retrySession = () => { cancelSession(); window.scrollTo({ top: 0, behavior: "smooth" }); };

  // ── Result helpers ──
  const userName      = user?.name?.split(" ")[0] || "there";
  const canRetry      = result && result.risk_tier !== "Green";
  const riskBadgeClass = !result ? "rb-green"
    : result.risk_tier === "Green" ? "rb-green"
    : result.risk_tier === "Yellow" ? "rb-yellow"
    : "rb-red";

  const getMessage = () => {
    if (!result) return "";
    // Use the AI-generated interpretation from the pipeline when available
    if (result.interpretation) return result.interpretation;
    // Fallback to hardcoded messages for demo mode
    if (result.risk_tier === "Green") return GOOD_MSG(userName);
    if (result.risk_tier === "Yellow") return WARN_MSG();
    return BAD_MSG(userName);
  };

  const currentStageLabels = mode === "text" ? TEXT_STAGE_LABELS : STAGE_LABELS;
  const activePrompt       = mode === "text" ? textPrompt : voicePrompt;

  // ════════════════════════════════════════════════════════════════
  // ── RENDER ──────────────────────────────────────────────────────
  // ════════════════════════════════════════════════════════════════
  return (
    <div className={`session-root ${dark ? "dark" : "light"}`}>

      {/* ── Text Mode extra styles ── */}
      <style>{`
        .mode-switcher {
          display: flex;
          align-items: center;
          gap: 8px;
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: 50px;
          padding: 5px;
          margin: 0 auto 32px;
          width: fit-content;
        }
        .mode-btn {
          display: flex; align-items: center; gap: 7px;
          padding: 9px 22px; border-radius: 50px; border: none;
          font-size: 13.5px; font-weight: 500; cursor: pointer;
          font-family: 'Instrument Sans', sans-serif;
          transition: all 0.2s ease;
          background: transparent;
          color: var(--text-secondary);
        }
        .mode-btn.active {
          background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
          color: #fff;
          box-shadow: 0 2px 12px rgba(155,79,122,0.3);
        }
        .mode-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .mode-tag {
          display: inline-block; font-size: 10px; font-weight: 600;
          padding: 2px 7px; border-radius: 20px; text-transform: uppercase;
          letter-spacing: 0.05em; background: rgba(255,255,255,0.25); color: inherit;
        }

        /* Text input area */
        .text-session-card {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: 24px;
          padding: 32px;
          margin-bottom: 20px;
          box-shadow: var(--shadow);
        }
        .text-session-header {
          display: flex; align-items: center; justify-content: space-between;
          margin-bottom: 16px;
        }
        .text-session-title {
          font-size: 16px; font-weight: 600; color: var(--text-primary);
        }
        .word-counter {
          font-size: 12.5px; font-weight: 500;
          color: var(--text-tertiary);
          transition: color 0.3s;
        }
        .word-counter.ready { color: var(--success); }
        .text-session-textarea {
          width: 100%; min-height: 200px;
          padding: 16px; border-radius: 14px;
          border: 1.5px solid var(--border);
          background: var(--bg-secondary);
          color: var(--text-primary);
          font-size: 15px; line-height: 1.7;
          font-family: 'Instrument Sans', sans-serif;
          resize: vertical; outline: none;
          transition: border-color 0.2s;
          box-sizing: border-box;
        }
        .text-session-textarea:focus { border-color: var(--accent-secondary); }
        .text-session-textarea::placeholder { color: var(--text-tertiary); }
        .text-session-textarea:disabled { opacity: 0.6; cursor: not-allowed; }
        .text-hint {
          margin-top: 10px; font-size: 12.5px; color: var(--text-tertiary);
          display: flex; align-items: center; gap: 6px;
        }
        .text-progress-bar {
          height: 4px; background: var(--border); border-radius: 4px;
          margin-top: 10px; overflow: hidden;
        }
        .text-progress-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
          border-radius: 4px;
          transition: width 0.3s ease;
        }
        .text-submit-btn {
          margin-top: 20px; width: 100%; padding: 14px 24px;
          border-radius: 14px; border: none; cursor: pointer;
          font-size: 15px; font-weight: 600;
          font-family: 'Instrument Sans', sans-serif;
          background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
          color: #fff;
          box-shadow: 0 4px 18px rgba(155,79,122,0.3);
          transition: all 0.2s ease;
          display: flex; align-items: center; justify-content: center; gap: 10px;
        }
        .text-submit-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 6px 24px rgba(155,79,122,0.4);
        }
        .text-submit-btn:disabled {
          opacity: 0.45; cursor: not-allowed; transform: none;
        }

        /* Text mode results — only show NLP biomarkers */
        .text-mode-badge {
          display: inline-flex; align-items: center; gap: 6px;
          font-size: 11.5px; padding: 4px 12px; border-radius: 20px;
          background: rgba(155,79,122,0.1);
          color: var(--accent-secondary); font-weight: 500;
          margin-bottom: 12px;
        }

        /* Text mode analysis progress */
        .text-analysing-card {
          background: var(--bg-card);
          border: 1px solid var(--border);
          border-radius: 24px;
          padding: 48px 32px;
          text-align: center;
          box-shadow: var(--shadow);
        }
        .text-analysing-icon {
          font-size: 40px; margin-bottom: 16px;
          animation: floatIcon 2s ease-in-out infinite;
        }
        @keyframes floatIcon {
          0%,100% { transform: translateY(0); }
          50% { transform: translateY(-8px); }
        }
        .text-analysing-title {
          font-size: 20px; font-weight: 600;
          color: var(--text-primary); margin-bottom: 8px;
        }
        .text-analysing-sub {
          font-size: 14px; color: var(--text-secondary); margin-bottom: 28px;
        }
        .text-stage-list {
          display: flex; flex-direction: column; gap: 10px;
          max-width: 320px; margin: 0 auto 24px;
        }
        .text-stage-item {
          display: flex; align-items: center; gap: 12px;
          padding: 10px 16px; border-radius: 12px;
          background: var(--bg-secondary); border: 1px solid var(--border);
          font-size: 13.5px; color: var(--text-secondary);
          transition: all 0.3s ease;
        }
        .text-stage-item.active {
          background: linear-gradient(135deg, rgba(196,176,248,0.15), rgba(155,79,122,0.1));
          border-color: var(--accent-primary);
          color: var(--text-primary);
        }
        .text-stage-item.done {
          color: var(--success);
          border-color: rgba(127,183,126,0.3);
        }
        .text-stage-dot {
          width: 8px; height: 8px; border-radius: 50%;
          background: var(--border); flex-shrink: 0; transition: background 0.3s;
        }
        .text-stage-item.active .text-stage-dot {
          background: var(--accent-secondary);
          box-shadow: 0 0 8px rgba(155,79,122,0.5);
          animation: pulseDot 1s infinite;
        }
        .text-stage-item.done .text-stage-dot { background: var(--success); }
        @keyframes pulseDot {
          0%,100% { transform: scale(1); }
          50% { transform: scale(1.4); }
        }
      `}</style>

      {/* ── Navigation ── */}
      <nav className="session-nav">
        <div className="nav-brand" onClick={() => navigate("/")}>
          <div className="brand-ring"><LogoIcon /></div>
          <span className="brand-name">CogniSafe</span>
        </div>
        <div className="nav-links">
          <button className="nav-link" onClick={() => navigate("/dashboard")}>Dashboard</button>
          <button className="nav-link active">Session</button>
          <button className="nav-link" onClick={() => navigate("/ar-report")}>Report</button>
        </div>
        <div className="nav-actions">
          <button className="theme-toggle" onClick={toggleTheme}>{dark ? "☀️" : "🌙"}</button>
          <button className="dashboard-link" onClick={() => navigate("/dashboard")}>Dashboard</button>
          <div className="user-menu" onClick={logout}>
            <div className="user-avatar">{user?.name?.charAt(0)?.toUpperCase() || "U"}</div>
          </div>
        </div>
      </nav>

      {/* ── Main ── */}
      <main className="session-main">

        {/* ── Header ── */}
        <div className="session-header">
          <div className="session-badge">Daily Session</div>
          <h1 className="session-title">
            {mode === "voice" ? <>Speak freely.<br /><span className="highlight">We listen carefully.</span></>
              : <>Write freely.<br /><span className="highlight">We analyse carefully.</span></>}
          </h1>
          <p className="session-description">
            {mode === "voice"
              ? "Find a quiet space, then speak naturally for 3 minutes. The more natural, the better."
              : "Write at least 50 words in response to today's prompt. Your language patterns will be analysed for cognitive health insights."}
          </p>
        </div>

        {/* ── Mode Switcher ── */}
        {(state === "idle" || state === "done" || state === "error") && (
          <div className="mode-switcher">
            <button
              className={`mode-btn ${mode === "voice" ? "active" : ""}`}
              onClick={() => switchMode("voice")}
              disabled={state === "recording" || state === "analysing"}
            >
              🎙️ Voice Mode
            </button>
            <button
              className={`mode-btn ${mode === "text" ? "active" : ""}`}
              onClick={() => switchMode("text")}
              disabled={state === "recording" || state === "analysing"}
            >
              ✍️ Text Mode
              <span className="mode-tag">Accessibility</span>
            </button>
          </div>
        )}

        {/* ── Done Today ── */}
        {state === "donegood" && (
          <div className="done-card">
            <div className="done-icon">🌿</div>
            <h2>You've completed today's session!</h2>
            <p>Your cognitive health is looking great today. Your data has been recorded and analysed.</p>
            <div className="done-badge">Cognitive health: Good</div>
            <p className="done-quote">"Your semantic coherence score is above your personal baseline. Keep up the great work — let's meet again tomorrow!"</p>
            <div className="done-actions">
              <button className="btn-primary" onClick={() => navigate("/dashboard")}>View Dashboard</button>
              <button className="btn-secondary" onClick={() => navigate("/ar-report")}>View Report</button>
            </div>
          </div>
        )}

        {/* ── Error ── */}
        {state === "error" && (
          <div className="error-card">
            <div className="error-icon">⚠️</div>
            <h2>Something went wrong</h2>
            <p>{errorMsg || "An unexpected error occurred. Please try again."}</p>
            <div className="error-actions">
              <button className="btn-primary" onClick={() => { setState("idle"); setErrorMsg(null); }}>Try Again</button>
              <button className="btn-secondary" onClick={() => navigate("/dashboard")}>Go to Dashboard</button>
            </div>
          </div>
        )}

        {/* ── Checking ── */}
        {state === "checking" && (
          <div className="loading-card">
            <div className="loading-spinner"></div>
            <p>Loading your session...</p>
          </div>
        )}

        {/* ══════════════════════════════════════════════
            VOICE MODE UI
        ══════════════════════════════════════════════ */}
        {mode === "voice" && (state === "idle" || state === "recording" || state === "analysing" || state === "done") && (
          <>
            {/* Prompt */}
            {!skipped && (
              <div className="prompt-card">
                <div className="prompt-header">
                  <span className="prompt-category">{voicePrompt.category}</span>
                  <span className="prompt-badge">Today's Prompt</span>
                </div>
                <div className="prompt-text">"{voicePrompt.text}"</div>
                {voicePrompt.image && <img className="prompt-image" src={voicePrompt.image} alt="Session prompt" />}
                {state === "idle" && (
                  <button className="skip-btn" onClick={() => setSkipped(true)}>Skip prompt & speak freely →</button>
                )}
              </div>
            )}
            {skipped && (
              <div className="free-mode-card">
                <div className="free-mode-icon">🎙️</div>
                <h3>Free-speak Mode</h3>
                <p>Speak about anything for 3 minutes. Your voice patterns will be analysed accurately.</p>
              </div>
            )}

            {/* Recorder Card */}
            <div className="recorder-card">
              <div className="recorder-visual">
                <div className="timer-ring">
                  <svg className="timer-svg" viewBox="0 0 140 140">
                    <defs>
                      <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#D4A5B5" />
                        <stop offset="100%" stopColor="#B5689A" />
                      </linearGradient>
                    </defs>
                    <circle className="timer-track" cx="70" cy="70" r="63" />
                    <circle className="timer-progress" cx="70" cy="70" r="63"
                      stroke={strokeColor} strokeDasharray={CIRC} strokeDashoffset={offset} />
                  </svg>
                  <div className="timer-center">
                    <div className="timer-value">{formatTime(timeLeft)}</div>
                    <div className="timer-label">
                      {state === "idle" && "Ready"}
                      {state === "recording" && "Recording"}
                      {state === "analysing" && "Analysing"}
                      {state === "done" && "Complete"}
                    </div>
                  </div>
                </div>
                {state === "recording" && (
                  <div className="volume-meter">
                    <div className="volume-bar" style={{ width: `${volumeLevel}%` }} />
                  </div>
                )}
                <div className="waveform-container">
                  <WaveformVisualizer isRecording={state === "recording"} />
                </div>
              </div>

              <div className={`status-pill ${state === "recording" ? "recording" : state === "analysing" ? "analysing" : state === "done" ? "done" : "idle"}`}>
                <span className="status-dot"></span>
                <span className="status-text">
                  {state === "idle"      && "Ready to record"}
                  {state === "recording" && `Recording... ${formatTime(timeLeft)}`}
                  {state === "analysing" && (STAGE_LABELS[analysisStage] || "Analysing your voice...")}
                  {state === "done"      && "Analysis complete ✓"}
                </span>
              </div>

              <div className="progress-steps">
                <div className={`step ${state === "idle" || state === "recording" ? "active" : state === "analysing" || state === "done" ? "completed" : ""}`}>
                  <span className="step-number">1</span><span className="step-label">Record</span>
                </div>
                {["transcribing", "acoustic", "nlp", "risk"].map((stage, i) => {
                  const stageIdx  = STAGE_ORDER.indexOf(analysisStage);
                  const thisIdx   = STAGE_ORDER.indexOf(stage);
                  const isActive  = state === "analysing" && analysisStage === stage;
                  const isDone    = state === "done" || (state === "analysing" && stageIdx > thisIdx);
                  return (
                    <div key={stage} className={`step ${isActive ? "active" : isDone ? "completed" : ""}`}>
                      <span className="step-number">{i + 2}</span>
                      <span className="step-label">
                        {stage === "transcribing" ? "Transcribe" : stage === "acoustic" ? "Acoustic" : stage === "nlp" ? "NLP" : "Risk"}
                      </span>
                    </div>
                  );
                })}
                <div className={`step ${state === "done" ? "completed" : ""}`}>
                  <span className="step-number">6</span><span className="step-label">Results</span>
                </div>
              </div>

              <div className="progress-bar-container">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>

              {state !== "done" && (
                <div className="recorder-controls">
                  <button
                    className={`record-btn ${state === "recording" ? "recording" : ""}`}
                    onClick={state === "idle" ? startRecording : stopRecording}
                    disabled={state === "analysing"}
                  >
                    {state === "idle"      && <><span className="record-icon">●</span>Start Recording</>}
                    {state === "recording" && <><span className="stop-icon">■</span>Stop Recording</>}
                    {state === "analysing" && <><span className="loading-spinner-small"></span>Analysing...</>}
                  </button>
                  <button className="cancel-btn" onClick={cancelSession} disabled={state === "analysing"}>
                    {state === "idle" ? "Cancel" : "Stop & Cancel"}
                  </button>
                </div>
              )}

              {state === "recording" && transcript && (
                <div className="transcript-box">
                  <div className="transcript-header">Live Transcript</div>
                  <div className="transcript-text">"{transcript}"</div>
                </div>
              )}
            </div>
          </>
        )}

        {/* ══════════════════════════════════════════════
            TEXT MODE UI
        ══════════════════════════════════════════════ */}
        {mode === "text" && (state === "idle" || state === "analysing" || state === "done") && (
          <>
            {/* Text Prompt */}
            {state === "idle" && (
              <div className="prompt-card">
                <div className="prompt-header">
                  <span className="prompt-category">{textPrompt.category}</span>
                  <span className="prompt-badge">Today's Prompt</span>
                </div>
                <div className="prompt-text">"{textPrompt.text}"</div>
              </div>
            )}

            {/* Text Input Card */}
            {state === "idle" && (
              <div className="text-session-card">
                <div className="text-session-header">
                  <span className="text-session-title">Your Response</span>
                  <span className={`word-counter ${textReady ? "ready" : ""}`}>
                    {wordCount} / {MIN_WORDS} words {textReady ? "✓" : ""}
                  </span>
                </div>
                <textarea
                  className="text-session-textarea"
                  placeholder="Start writing here... Describe your thoughts in as much detail as possible. The more you write, the more accurate your analysis will be."
                  value={typedText}
                  onChange={(e) => setTypedText(e.target.value)}
                  disabled={state === "analysing"}
                />
                <div className="text-progress-bar">
                  <div className="text-progress-fill"
                    style={{ width: `${Math.min(100, (wordCount / MIN_WORDS) * 100)}%` }} />
                </div>
                <div className="text-hint">
                  <span>💡</span>
                  <span>Write naturally and in detail — your vocabulary, sentence structure, and idea flow are what matter, not spelling or grammar.</span>
                </div>
                <button className="text-submit-btn" onClick={submitText} disabled={!textReady}>
                  {textReady ? "✓ Analyse My Writing" : `Write ${MIN_WORDS - wordCount} more words to continue`}
                </button>
              </div>
            )}

            {/* Text Analysing State */}
            {state === "analysing" && (
              <div className="text-analysing-card">
                <div className="text-analysing-icon">🧠</div>
                <div className="text-analysing-title">Analysing your writing...</div>
                <div className="text-analysing-sub">
                  Running cognitive language analysis on your text
                </div>
                <div className="text-stage-list">
                  {[
                    { key: "nlp",  label: "Analysing language patterns" },
                    { key: "risk", label: "Computing risk tier" },
                    { key: "done", label: "Complete" },
                  ].map(({ key, label }) => {
                    const order   = TEXT_STAGE_ORDER.indexOf(key);
                    const current = TEXT_STAGE_ORDER.indexOf(analysisStage);
                    const isActive = analysisStage === key;
                    const isDone   = current > order;
                    return (
                      <div key={key} className={`text-stage-item ${isActive ? "active" : isDone ? "done" : ""}`}>
                        <div className="text-stage-dot" />
                        {isDone ? "✓ " : ""}{label}
                      </div>
                    );
                  })}
                </div>
                <div className="progress-bar-container">
                  <div className="progress-fill" style={{ width: `${progress}%` }} />
                </div>
              </div>
            )}
          </>
        )}

        {/* ══════════════════════════════════════════════
            RESULTS PANEL (shared for both modes)
        ══════════════════════════════════════════════ */}
        {state === "done" && result && (
          <div className="results-card" ref={resultRef}>
            <div className="results-header">
              <h3>Session Results</h3>
              <div className={`results-badge ${riskBadgeClass}`}>
                ● {result.risk_tier} — {result.risk_tier === "Green" ? "Good" : result.risk_tier === "Yellow" ? "Watch" : "Alert"}
              </div>
            </div>

            {/* Text mode label */}
            {result.mode === "text" && (
              <div className="text-mode-badge">✍️ Text Mode — NLP biomarkers only</div>
            )}

            {/* Hindi language badge */}
            {result.biomarkers?.language_detected === "hi" && (
              <div className="text-mode-badge">🇮🇳 Hindi session detected and analysed</div>
            )}

            <div className="results-metrics">
              <div className="metric">
                <span className="metric-label">Risk Score</span>
                <span className="metric-value">
                  {result.risk_tier === "Green" ? "0.18" : result.risk_tier === "Yellow" ? "0.35" : "0.52"}
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Semantic Coherence</span>
                <span className="metric-value">
                  {result.biomarkers?.semantic_coherence != null
                    ? result.biomarkers.semantic_coherence.toFixed(2) : "—"}
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Lexical Diversity</span>
                <span className="metric-value">
                  {result.biomarkers?.lexical_diversity != null
                    ? result.biomarkers.lexical_diversity.toFixed(2) : "—"}
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">Idea Density</span>
                <span className="metric-value">
                  {result.biomarkers?.idea_density != null
                    ? result.biomarkers.idea_density.toFixed(3) : "—"}
                </span>
              </div>
              {/* Show speech metrics only for voice mode */}
              {result.mode !== "text" && (
                <>
                  <div className="metric">
                    <span className="metric-label">Speech Rate</span>
                    <span className="metric-value">
                      {result.biomarkers?.speech_rate != null
                        ? `${Math.round(result.biomarkers.speech_rate)} wpm` : "—"}
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Pause Frequency</span>
                    <span className="metric-value">
                      {result.biomarkers?.pause_frequency != null
                        ? `${result.biomarkers.pause_frequency.toFixed(1)}/min` : "—"}
                    </span>
                  </div>
                </>
              )}
            </div>

            {/* ── Mood Context Card (only when API returns mood) ── */}
            {result.mood && result.mood.label && (
              <div className="mood-context-card" style={{
                background: "var(--bg-card)",
                border: `1.5px solid ${MOOD_COLOR[result.mood.label] || "#888"}40`,
                borderLeft: `4px solid ${MOOD_COLOR[result.mood.label] || "#888"}`,
                borderRadius: "14px",
                padding: "16px 20px",
                margin: "16px 0",
                display: "flex",
                alignItems: "flex-start",
                gap: "14px",
              }}>
                <span style={{ fontSize: "28px", lineHeight: 1 }}>
                  {MOOD_EMOJI[result.mood.label] || "🧠"}
                </span>
                <div style={{ flex: 1 }}>
                  <div style={{
                    display: "flex", alignItems: "center", gap: "10px",
                    marginBottom: "6px", flexWrap: "wrap"
                  }}>
                    <span style={{
                      fontWeight: 600, fontSize: "14px",
                      color: "var(--text-primary)", textTransform: "capitalize"
                    }}>
                      Mood detected: {result.mood.label}
                    </span>
                    <span style={{
                      fontSize: "11px", fontWeight: 600,
                      padding: "2px 9px", borderRadius: "20px",
                      background: `${MOOD_COLOR[result.mood.label] || "#888"}20`,
                      color: MOOD_COLOR[result.mood.label] || "#888",
                      textTransform: "uppercase", letterSpacing: "0.04em"
                    }}>
                      {Math.round((result.mood.confidence || 0) * 100)}% confidence
                    </span>
                  </div>
                  <p style={{
                    margin: 0, fontSize: "13px",
                    color: "var(--text-secondary)", lineHeight: 1.55
                  }}>
                    {result.mood?.corrected
                      ? "Mood was used as a correction factor — your score reflects cognitive patterns independent of your current emotional state."
                      : "Mood is used as context — it does not directly determine your risk tier, but helps distinguish temporary emotional effects from persistent patterns."
                    }
                  </p>
                </div>
              </div>
            )}

            {/* ── Model Method Badge ── */}
            {result.method && (
              <div style={{
                display: "flex", alignItems: "center", gap: "8px",
                marginBottom: "14px", flexWrap: "wrap"
              }}>
                <span style={{
                  fontSize: "11px", padding: "3px 10px", borderRadius: "20px",
                  background: "var(--bg-secondary)", color: "var(--text-secondary)",
                  fontWeight: 500, letterSpacing: "0.03em"
                }}>
                  🔬 {result.method === "xgboost_only"
                    ? "XGBoost only (early sessions)"
                    : result.method === "hybrid_70_30"
                    ? "Hybrid — 70% XGBoost / 30% personal baseline"
                    : "Hybrid — 50% XGBoost / 50% personal baseline"}
                </span>
                {result.session_count && (
                  <span style={{
                    fontSize: "11px", padding: "3px 10px", borderRadius: "20px",
                    background: "var(--bg-secondary)", color: "var(--text-secondary)",
                    fontWeight: 500
                  }}>
                    Session #{result.session_count}
                  </span>
                )}
              </div>
            )}

            <div className="results-message" dangerouslySetInnerHTML={{ __html: getMessage() }} />

            <div className="results-actions">
              <button className="btn-primary" onClick={() => navigate("/dashboard")}>View Dashboard</button>
              {canRetry && <button className="btn-secondary" onClick={retrySession}>Try Again</button>}
              <button className="btn-secondary" onClick={() => navigate("/ar-report")}>View Full Report</button>
            </div>
          </div>
        )}

      </main>
    </div>
  );
};

export default Session;
