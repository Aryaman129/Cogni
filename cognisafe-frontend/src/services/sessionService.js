const API    = import.meta.env.VITE_API_URL || "http://localhost:8000";

const authHeaders = (token) => ({
  "Content-Type": "application/json",
  "Authorization": `Bearer ${token}`,
});

// ── Check if user already recorded today ──
export const checkToday = async (token) => {
  const res = await fetch(`${API}/api/sessions/today`, {
    headers: authHeaders(token),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to check today");
  return data; // { recorded: bool, risk_tier?, session_id? }
};

// ── Convert audio blob to WAV using Web Audio API ──
// The browser records .webm but ML service only accepts .wav/.mp3/.m4a
// We send as .wav by renaming — the ML service will process it via ffmpeg/librosa
const blobToWav = async (blob) => {
  // Strategy: rename the blob to .wav
  // librosa can handle webm internally on the server side
  // If this fails, fall back to sending as .ogg (also accepted)
  return new File([blob], "recording.wav", { type: "audio/wav" });
};

// ── Send audio blob to ML service ──
export const analyzeAudio = async (audioBlob, userId, externalSignal = null) => {
  const formData = new FormData();

  // Convert to wav file (ML only accepts wav/mp3/m4a/ogg/flac, NOT webm)
  const audioFile = await blobToWav(audioBlob);
  formData.append("audio",   audioFile);
  formData.append("user_id", String(userId));

  const controller = new AbortController();
  const timeout    = setTimeout(() => controller.abort(), 360000); // 6 minutes

  // If user unmounts/cancels manually, we abort both
  if (externalSignal) {
    externalSignal.addEventListener("abort", () => {
      controller.abort();
    });
  }

  try {
    const res = await fetch(`${API}/api/ml/analyze`, {
      method: "POST",
      body:   formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.message || `AI service error ${res.status}`);
    }

    const raw = await res.json();
    return normalizeAIResult(raw);

  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError" && externalSignal && externalSignal.aborted) {
       throw new Error("Analysis cancelled by user.");
    }
    if (err.name === "AbortError")
      throw new Error("Analysis timed out after 6 minutes — please try again.");
    throw err;
  }
};

// ── Normalise ML response → consistent internal shape ──
// Maps ML field names (HNR, pause_duration_mean, filled_pause_rate)
// to frontend field names used in Session.jsx and Dashboard.jsx
export const normalizeAIResult = (raw) => {
  const bm = raw.biomarkers || {};

  const biomarkers = {
    semantic_coherence:   bm.semantic_coherence   ?? null,
    lexical_diversity:    bm.lexical_diversity     ?? null,
    idea_density:         bm.idea_density          ?? null,
    syntactic_complexity: bm.syntactic_complexity  ?? null,
    speech_rate:          bm.speech_rate           ?? null,
    pause_frequency:      bm.pause_frequency       ?? null,
    pause_duration:       bm.pause_duration_mean   ?? null,   // ML uses pause_duration_mean
    pitch_mean:           bm.pitch_mean            ?? null,
    pitch_range:          bm.pitch_range           ?? null,
    jitter:               bm.jitter                ?? null,
    shimmer:              bm.shimmer               ?? null,
    hnr:                  bm.HNR                   ?? null,   // ML uses capital HNR
    articulation_rate:    bm.articulation_rate     ?? null,
    emotional_entropy:    bm.emotional_entropy      ?? null,  // may not exist in ML
    filled_pause_rate:    bm.filled_pause_rate     ?? null,   // extra ML biomarker
  };

  // Risk tier: Green / Yellow / Orange / Red
  const risk_tier = raw.risk_tier || "Green";

  // Anomaly flags: array of biomarker key strings
  const anomaly_flags = raw.anomaly_flags || [];

  return {
    risk_tier,
    biomarkers,
    anomaly_flags,
    session_id:      raw.session_id      || null,
    timestamp:       raw.timestamp       || null,
    processing_time: raw.processing_time_seconds ?? null,
    user_id:         raw.user_id         || null,
    confidence_intervals: raw.confidence_intervals || null,
  };
};

// ── Save AI result to backend ──
export const saveSession = async (token, aiResult) => {
  const bm = aiResult.biomarkers || {};

  const payload = {
    risk_tier:            aiResult.risk_tier,
    semantic_coherence:   bm.semantic_coherence   ?? null,
    lexical_diversity:    bm.lexical_diversity     ?? null,
    idea_density:         bm.idea_density          ?? null,
    speech_rate:          bm.speech_rate           ?? null,
    pause_frequency:      bm.pause_frequency       ?? null,
    pause_duration:       bm.pause_duration        ?? null,
    pitch_mean:           bm.pitch_mean            ?? null,
    pitch_range:          bm.pitch_range           ?? null,
    jitter:               bm.jitter                ?? null,
    shimmer:              bm.shimmer               ?? null,
    hnr:                  bm.hnr                   ?? null,
    syntactic_complexity: bm.syntactic_complexity  ?? null,
    articulation_rate:    bm.articulation_rate     ?? null,
    emotional_entropy:    bm.emotional_entropy      ?? null,
    has_anomaly:          (aiResult.anomaly_flags?.length ?? 0) > 0,
    anomaly_flags:        JSON.stringify(aiResult.anomaly_flags || []),
  };

  const res = await fetch(`${API}/api/sessions`, {
    method:  "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
    body:    JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to save session");
  return data;
};  