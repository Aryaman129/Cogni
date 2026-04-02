const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const authHeaders = (token) => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${token}`,
});

// ── Check if user already recorded today ──────────────────────────────────────
export const checkToday = async (token) => {
  const res = await fetch(`${API}/api/sessions/today`, {
    headers: authHeaders(token),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to check today");
  return data;
};

// ── Convert audio blob → File named .wav ─────────────────────────────────────
const blobToWav = async (blob) =>
  new File([blob], "recording.wav", { type: "audio/wav" });

// ── Stage label map (for the progress bar UI) ─────────────────────────────────
export const STAGE_LABELS = {
  uploading:    "Uploading audio...",
  transcribing: "Transcribing speech (Whisper)...",
  acoustic:     "Extracting acoustic features...",
  nlp:          "Analysing language patterns...",
  risk:         "Computing risk tier...",
  done:         "Analysis complete ✓",
};

export const STAGE_ORDER = ["uploading", "transcribing", "acoustic", "nlp", "risk", "done"];

// ── Submit audio → get job_id immediately ─────────────────────────────────────
export const submitAudioJob = async (audioBlob, userId) => {
  const formData = new FormData();
  const audioFile = await blobToWav(audioBlob);
  formData.append("audio", audioFile);
  formData.append("user_id", String(userId));

  const res = await fetch(`${API}/api/ml/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Submit failed: ${res.status}`);
  }

  const data = await res.json();
  return data.job_id; // string UUID
};

// ── Poll job status until done or failed ──────────────────────────────────────
// onStageChange(stage) is called every time the stage label changes.
// Returns the final normalised AI result on success, throws on failure.
export const pollJobUntilDone = async (
  jobId,
  onStageChange,
  { intervalMs = 3000, maxWaitMs = 420000 } = {} // 7 min max
) => {
  const deadline = Date.now() + maxWaitMs;
  let lastStage = null;

  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, intervalMs));

    const res = await fetch(`${API}/api/ml/status/${jobId}`);
    if (!res.ok) {
      // Transient network error — keep trying
      console.warn("[pollJob] status fetch failed, retrying...");
      continue;
    }

    const data = await res.json();

    // Fire stage callback when stage changes
    if (data.stage !== lastStage) {
      lastStage = data.stage;
      onStageChange?.(data.stage);
    }

    if (data.status === "done") {
      return normalizeAIResult(data.result);
    }

    if (data.status === "failed") {
      throw new Error(data.error || "Analysis failed on the server.");
    }

    // status === "queued" | "processing" → keep polling
  }

  throw new Error("Analysis timed out waiting for result.");
};

// ── Legacy single-call analyzeAudio (kept for demo/fallback path) ─────────────
export const analyzeAudio = async (audioBlob, userId, externalSignal = null) => {
  const formData = new FormData();
  const audioFile = await blobToWav(audioBlob);
  formData.append("audio", audioFile);
  formData.append("user_id", String(userId));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 360000);

  if (externalSignal) {
    externalSignal.addEventListener("abort", () => controller.abort());
  }

  try {
    const res = await fetch(`${API}/api/ml/analyze`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.message || `AI service error ${res.status}`);
    }
    return normalizeAIResult(await res.json());
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === "AbortError" && externalSignal?.aborted)
      throw new Error("Analysis cancelled by user.");
    if (err.name === "AbortError")
      throw new Error("Analysis timed out after 6 minutes — please try again.");
    throw err;
  }
};

// ── Normalise ML response → consistent internal shape ────────────────────────
export const normalizeAIResult = (raw) => {
  const bm = raw.biomarkers || {};
  return {
    risk_tier: raw.risk_tier || "Green",
    biomarkers: {
      semantic_coherence:   bm.semantic_coherence   ?? null,
      lexical_diversity:    bm.lexical_diversity     ?? null,
      idea_density:         bm.idea_density          ?? null,
      syntactic_complexity: bm.syntactic_complexity  ?? null,
      speech_rate:          bm.speech_rate           ?? null,
      pause_frequency:      bm.pause_frequency       ?? null,
      pause_duration:       bm.pause_duration_mean   ?? null,
      pitch_mean:           bm.pitch_mean            ?? null,
      pitch_range:          bm.pitch_range           ?? null,
      jitter:               bm.jitter                ?? null,
      shimmer:              bm.shimmer               ?? null,
      hnr:                  bm.HNR                   ?? null,
      articulation_rate:    bm.articulation_rate     ?? null,
      emotional_entropy:    bm.emotional_entropy      ?? null,
      filled_pause_rate:    bm.filled_pause_rate     ?? null,
    },
    anomaly_flags:        raw.anomaly_flags         || [],
    session_id:           raw.session_id            || null,
    timestamp:            raw.timestamp             || null,
    processing_time:      raw.processing_time_seconds ?? null,
    user_id:              raw.user_id               || null,
    confidence_intervals: raw.confidence_intervals  || null,
  };
};

// ── Save AI result to backend ─────────────────────────────────────────────────
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
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to save session");
  return data;
};