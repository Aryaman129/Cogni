const API    = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ✅ Call HF directly — bypasses Render 30s timeout
const HF_URL = "https://alamfarzann-cognisafe-ml.hf.space";

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
  return data;
};

// ── Send audio blob to ML service ──
export const analyzeAudio = async (audioBlob, userId) => {
  const formData = new FormData();

  // ✅ send as .webm — HF Space now accepts it
  formData.append("audio",   new File([audioBlob], "recording.webm", { type: "audio/webm" }));
  formData.append("user_id", String(userId));

  const controller = new AbortController();
  // ✅ increased from 95s to 290s — HF free tier needs up to 2 mins
  const timeout    = setTimeout(() => controller.abort(), 290000);

  try {
    // ✅ call HF directly instead of Render proxy
    const res = await fetch(`${HF_URL}/analyze`, {
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
    if (err.name === "AbortError")
      throw new Error("Analysis timed out after 290 seconds — please try again.");
    throw err;
  }
};

// ── Normalise ML response → consistent internal shape ──
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
    emotional_entropy:    bm.emotional_entropy      ?? null,
    filled_pause_rate:    bm.filled_pause_rate     ?? null,
  };

  const risk_tier     = raw.risk_tier     || "Green";
  const anomaly_flags = raw.anomaly_flags || [];

  return {
    risk_tier,
    biomarkers,
    anomaly_flags,
    session_id:           raw.session_id               || null,
    timestamp:            raw.timestamp                || null,
    processing_time:      raw.processing_time_seconds  ?? null,
    user_id:              raw.user_id                  || null,
    confidence_intervals: raw.confidence_intervals     || null,
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