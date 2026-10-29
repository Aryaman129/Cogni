const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const authHeaders = (token) => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${token}`,
});

// Helper to obtain the active JWT token
const getToken = (explicitToken) => {
  if (typeof explicitToken === "string" && explicitToken.length > 20) {
    return explicitToken;
  }
  return localStorage.getItem("cog_jwt") || "";
};

// ── Check if user already recorded today ──────────────────────────────────────
export const checkToday = async (token) => {
  const jwt = getToken(token);
  const res = await fetch(`${API}/api/sessions/today`, {
    headers: authHeaders(jwt),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to check today");
  return data;
};

// ── Convert audio blob → File named recording.wav ─────────────────────────────
const blobToWav = async (blob) =>
  new File([blob], "recording.wav", { type: "audio/wav" });

// ── Stage label map (for progress bar UI) ─────────────────────────────────────
export const STAGE_LABELS = {
  uploading:    "Uploading audio...",
  queued:       "Queued for processing...",
  processing:   "Processing audio stream...",
  transcribing: "Transcribing speech (Whisper)...",
  acoustic:     "Extracting acoustic features...",
  nlp:          "Analysing language patterns...",
  risk:         "Computing risk tier & baseline...",
  done:         "Analysis complete ✓",
  failed:       "Analysis failed",
};

// Text mode stage labels
export const TEXT_STAGE_LABELS = {
  nlp:  "Analysing language patterns...",
  risk: "Computing risk tier & baseline...",
  done: "Analysis complete ✓",
};

export const STAGE_ORDER      = ["uploading", "processing", "transcribing", "acoustic", "nlp", "risk", "done"];
export const TEXT_STAGE_ORDER = ["nlp", "risk", "done"];

// ── Submit audio to authenticated backend /api/ml/analyze & poll job status ───
export const submitAudioJob = async (audioBlob, tokenOrUserId, onStageChange) => {
  const jwt = getToken(tokenOrUserId);
  const formData = new FormData();
  const audioFile = await blobToWav(audioBlob);
  formData.append("audio", audioFile);

  onStageChange?.("uploading");

  // Step 1: Submit audio file to FastAPI backend
  const uploadController = new AbortController();
  const uploadTimeout = setTimeout(() => uploadController.abort(), 60000); // 1 min for upload

  let initialRes;
  try {
    const headers = {};
    if (jwt) {
      headers["Authorization"] = `Bearer ${jwt}`;
    }

    const res = await fetch(`${API}/api/ml/analyze`, {
      method: "POST",
      body: formData,
      headers,
      signal: uploadController.signal,
    });
    clearTimeout(uploadTimeout);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload error ${res.status}`);
    }

    initialRes = await res.json();
  } catch (err) {
    clearTimeout(uploadTimeout);
    if (err.name === "AbortError") {
      throw new Error("Audio upload timed out. Please check your connection.");
    }
    throw err;
  }

  const jobId = initialRes.job_id;
  if (!jobId) {
    throw new Error("Backend did not return a valid job ID.");
  }

  onStageChange?.("processing");

  // Step 2: Poll real job status (/api/ml/status/{job_id}) without artificial delays
  const pollIntervalMs = 1500;
  const maxPollTimeMs = 480000; // 8 minutes
  const startTime = Date.now();

  return new Promise((resolve, reject) => {
    const poll = async () => {
      if (Date.now() - startTime > maxPollTimeMs) {
        return reject(new Error("Analysis timed out after 8 minutes. Please try again."));
      }

      try {
        const headers = {};
        if (jwt) {
          headers["Authorization"] = `Bearer ${jwt}`;
        }

        const res = await fetch(`${API}/api/ml/status/${jobId}`, {
          method: "GET",
          headers,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          return reject(new Error(err.detail || `Job status error ${res.status}`));
        }

        const data = await res.json();

        // Update progress stage if reported
        if (data.stage) {
          onStageChange?.(data.stage);
        } else if (data.status) {
          onStageChange?.(data.status);
        }

        if (data.status === "done") {
          onStageChange?.("done");
          resolve(normalizeAIResult(data.result));
        } else if (data.status === "failed") {
          reject(new Error(data.error || "Speech biomarker analysis failed."));
        } else {
          // Job still queued or processing, schedule next poll
          setTimeout(poll, pollIntervalMs);
        }
      } catch (err) {
        reject(err);
      }
    };

    setTimeout(poll, pollIntervalMs);
  });
};

// ── Submit TEXT to authenticated backend /api/ml/analyze-text ─────────────────
export const submitTextJob = async (text, tokenOrUserId, onStageChange) => {
  const jwt = getToken(tokenOrUserId);
  onStageChange?.("nlp");

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000); // 1 min

    const headers = { "Content-Type": "application/json" };
    if (jwt) {
      headers["Authorization"] = `Bearer ${jwt}`;
    }

    const res = await fetch(`${API}/api/ml/analyze-text`, {
      method: "POST",
      headers,
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });

    clearTimeout(timeout);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Text analysis error ${res.status}`);
    }

    onStageChange?.("risk");
    const rawResult = await res.json();
    onStageChange?.("done");
    return normalizeAIResult(rawResult);

  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Text analysis timed out — please try again.");
    }
    throw err;
  }
};

// ── Normalise ML response → consistent internal shape ─────────────────────────
// Eliminates silent "Green" fallback; preserves explicit analysis_status and typed results
export const normalizeAIResult = (raw) => {
  if (!raw) return null;
  const bm = raw.biomarkers || {};

  return {
    risk_tier:            raw.risk_tier !== undefined ? raw.risk_tier : null,
    analysis_status:      raw.analysis_status || (raw.risk_tier ? "complete" : "insufficient_baseline"),
    mode:                 raw.mode || "voice",
    biomarkers: {
      // 10 Acoustic Biomarkers
      speech_rate:          bm.speech_rate           ?? null,
      articulation_rate:    bm.articulation_rate     ?? null,
      pause_frequency:      bm.pause_frequency       ?? null,
      pause_duration_mean:  bm.pause_duration_mean   ?? bm.pause_duration ?? null,
      pause_duration:       bm.pause_duration_mean   ?? bm.pause_duration ?? null,
      filled_pause_rate:    bm.filled_pause_rate     ?? null,
      pitch_mean:           bm.pitch_mean            ?? null,
      pitch_range:          bm.pitch_range           ?? null,
      jitter:               bm.jitter                ?? null,
      shimmer:              bm.shimmer               ?? null,
      hnr:                  bm.HNR                   ?? bm.hnr ?? null,
      HNR:                  bm.HNR                   ?? bm.hnr ?? null,
      // 4 Linguistic / NLP Biomarkers
      lexical_diversity:    bm.lexical_diversity     ?? null,
      semantic_coherence:   bm.semantic_coherence   ?? null,
      idea_density:         bm.idea_density          ?? null,
      syntactic_complexity: bm.syntactic_complexity  ?? null,
    },
    biomarker_results:    raw.biomarker_results      || raw.typed_features || null,
    anomaly_flags:        raw.anomaly_flags          || [],
    session_id:           raw.session_id             || null,
    timestamp:            raw.timestamp              || null,
    processing_time:      raw.processing_time_seconds ?? null,
    user_id:              raw.user_id                || null,
    confidence_intervals: raw.confidence_intervals   || null,
    mood:                 raw.mood                   || null,
    interpretation:       raw.interpretation         || null,
    method:               raw.method                 || null,
    session_count:        raw.session_count          || null,
    ai_insights:          raw.ai_insights            || null,
  };
};

// ── Save AI result to backend /api/sessions ──────────────────────────────────
// Persists all 14 canonical biomarkers including filled_pause_rate and pause_duration_mean
export const saveSession = async (token, aiResult) => {
  const jwt = getToken(token);
  const bm = aiResult.biomarkers || {};

  const payload = {
    risk_tier:            aiResult.risk_tier ?? null,
    analysis_status:      aiResult.analysis_status || "complete",
    mode:                 aiResult.mode || "voice",
    // 10 Acoustic Biomarkers
    speech_rate:          bm.speech_rate           ?? null,
    articulation_rate:    bm.articulation_rate     ?? null,
    pause_frequency:      bm.pause_frequency       ?? null,
    pause_duration_mean:  bm.pause_duration_mean   ?? bm.pause_duration ?? null,
    pause_duration:       bm.pause_duration_mean   ?? bm.pause_duration ?? null,
    filled_pause_rate:    bm.filled_pause_rate     ?? null,
    pitch_mean:           bm.pitch_mean            ?? null,
    pitch_range:          bm.pitch_range           ?? null,
    jitter:               bm.jitter                ?? null,
    shimmer:              bm.shimmer               ?? null,
    hnr:                  bm.hnr                   ?? bm.HNR ?? null,
    HNR:                  bm.HNR                   ?? bm.hnr ?? null,
    // 4 Linguistic Biomarkers
    lexical_diversity:    bm.lexical_diversity     ?? null,
    semantic_coherence:   bm.semantic_coherence   ?? null,
    idea_density:         bm.idea_density          ?? null,
    syntactic_complexity: bm.syntactic_complexity  ?? null,
    // Anomaly flags & insights
    has_anomaly:          (aiResult.anomaly_flags?.length ?? 0) > 0,
    anomaly_flags:        JSON.stringify(aiResult.anomaly_flags || []),
    biomarker_results:    aiResult.biomarker_results || null,
    ai_insights:          aiResult.ai_insights || (aiResult.interpretation ? { narrative: aiResult.interpretation } : null),
  };

  const res = await fetch(`${API}/api/sessions`, {
    method:  "POST",
    headers: authHeaders(jwt),
    body:    JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to save session");
  return data;
};
