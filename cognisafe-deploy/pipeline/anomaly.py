"""
anomaly.py — Hybrid anomaly detection pipeline.

Strategy:
  - Sessions 1-4  : XGBoost only (no personal history yet)
  - Sessions 5+   : XGBoost + Z-score combined, weighted average
  - Final tier    : stricter of the two when they disagree

XGBoost gives population-level risk from day 1.
Z-score personalises to the individual over time.
Together they are more accurate than either alone.
"""

import sqlite3
import numpy as np
import os
import json
import xgboost as xgb
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(__file__)
DB_PATH     = os.path.join(_DIR, "..", "data", "sessions.db")
MODEL_PATH  = os.path.join(_DIR, "xgb_model.json")
LABELS_PATH = os.path.join(_DIR, "xgb_model_labels.json")

# ── Biomarker feature order — must match train_model.py exactly ───────────────
BIOMARKERS = [
    "speech_rate", "articulation_rate", "pause_frequency",
    "pause_duration_mean", "filled_pause_rate", "pitch_mean",
    "pitch_range", "jitter", "shimmer", "HNR",
    "lexical_diversity", "semantic_coherence",
    "idea_density", "syntactic_complexity",
]

# ── Risk tier ordering (for comparison logic) ─────────────────────────────────
TIER_ORDER = {"Green": 0, "Yellow": 1, "Orange": 2, "Red": 3}
TIER_NAMES = ["Green", "Yellow", "Orange", "Red"]

# ── Load XGBoost model once at startup ────────────────────────────────────────
_xgb_model  = None
_xgb_labels = None  # { "0": "Green", "1": "Orange", ... }

def _load_xgb():
    global _xgb_model, _xgb_labels
    if _xgb_model is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        print("[XGB] ⚠️  Model file not found — run pipeline/train_model.py first")
        return False
    try:
        _xgb_model = xgb.XGBClassifier()
        _xgb_model.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            _xgb_labels = json.load(f)
        print("[XGB] ✅ XGBoost model loaded")
        return True
    except Exception as e:
        print(f"[XGB] ❌ Failed to load model: {e}")
        return False


# ── Database Setup ────────────────────────────────────────────────────────────
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT    NOT NULL,
            timestamp     TEXT    NOT NULL,
            biomarkers    TEXT    NOT NULL,
            risk_tier     TEXT    NOT NULL,
            anomaly_flags TEXT    NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_session(user_id, biomarkers, risk_tier, anomaly_flags):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (user_id, timestamp, biomarkers, risk_tier, anomaly_flags) VALUES (?,?,?,?,?)",
        (user_id, datetime.utcnow().isoformat(),
         json.dumps(biomarkers), risk_tier, json.dumps(anomaly_flags))
    )
    conn.commit()
    conn.close()
    print(f"[DB] Session saved for user {user_id} ✅")


def load_sessions(user_id):
    init_db()
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT biomarkers, timestamp FROM sessions WHERE user_id=? ORDER BY timestamp ASC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [{"biomarkers": json.loads(r[0]), "timestamp": r[1]} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# ── 1. XGBoost Prediction ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def predict_xgb(current_biomarkers: dict) -> dict:
    """
    Run the XGBoost model on the current session biomarkers.

    Returns:
        {
          "tier":        "Green" | "Yellow" | "Orange" | "Red",
          "confidence":  0.0 – 1.0,
          "probabilities": { "Green": x, "Yellow": x, ... }
        }
    """
    if not _load_xgb():
        return {"tier": None, "confidence": 0.0, "probabilities": {}}

    # Build feature vector in the correct order
    # Use 0.0 for any missing feature (text mode has no acoustic features)
    x = np.array([[
        float(current_biomarkers.get(feat, 0.0) or 0.0)
        for feat in BIOMARKERS
    ]])

    proba  = _xgb_model.predict_proba(x)[0]          # shape: (n_classes,)
    pred   = int(np.argmax(proba))
    tier   = _xgb_labels[str(pred)]
    confidence = float(proba[pred])

    prob_map = {
        _xgb_labels[str(i)]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    print(f"[XGB] Prediction: {tier} (confidence {confidence:.2%})")
    print(f"[XGB] Probabilities: {prob_map}")

    return {"tier": tier, "confidence": confidence, "probabilities": prob_map}


# ══════════════════════════════════════════════════════════════════════════════
# ── 2. Z-Score Anomaly Detection (personalised baseline) ─────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def detect_anomalies_zscore(user_id: str, current_biomarkers: dict) -> list:
    """
    Compare current session against user's personal historical baseline.
    Only meaningful once 5+ sessions exist.
    """
    past_sessions = load_sessions(user_id)
    if len(past_sessions) < 5:
        print(f"[Z-Score] Only {len(past_sessions)} sessions — skipping (need 5+)")
        return []

    anomaly_flags = []
    for biomarker in BIOMARKERS:
        history = [
            float(s["biomarkers"][biomarker])
            for s in past_sessions
            if biomarker in s["biomarkers"] and s["biomarkers"][biomarker] not in (None, 0.0)
        ]
        if len(history) < 3:
            continue

        mean      = np.mean(history)
        std       = np.std(history)
        current   = float(current_biomarkers.get(biomarker, 0.0) or 0.0)
        if std == 0:
            continue

        deviation = abs(current - mean) / std
        if deviation >= 2.0:
            severity = "severe" if deviation >= 3.0 else "moderate" if deviation >= 2.5 else "mild"
            anomaly_flags.append({
                "biomarker": biomarker,
                "severity":  severity,
                "current":   round(current, 4),
                "baseline":  round(mean, 4),
                "deviation": round(deviation, 2),
            })
            print(f"[Z-Score] ⚠️  {biomarker}: {severity} ({deviation:.2f}σ)")

    return anomaly_flags


def zscore_to_tier(anomaly_flags: list) -> str:
    """Convert z-score anomaly flags → risk tier."""
    if not anomaly_flags:
        return "Green"
    mild     = sum(1 for f in anomaly_flags if f["severity"] == "mild")
    moderate = sum(1 for f in anomaly_flags if f["severity"] == "moderate")
    severe   = sum(1 for f in anomaly_flags if f["severity"] == "severe")

    if severe >= 2 or moderate >= 3:
        return "Red"
    elif moderate >= 2 or severe >= 1:
        return "Orange"
    elif mild >= 2 or moderate >= 1:
        return "Yellow"
    return "Green"


# ══════════════════════════════════════════════════════════════════════════════
# ── 3. Hybrid Combination Logic ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def combine_tiers(xgb_tier: str, zscore_tier: str, session_count: int, xgb_confidence: float) -> str:
    """
    Combine XGBoost and Z-score predictions into one final tier.

    Rules:
    - Sessions 1–4  : XGBoost only (no personal history)
    - Sessions 5–9  : XGBoost 70% weight, Z-score 30% weight
    - Sessions 10+  : XGBoost 50% weight, Z-score 50% weight
    - If XGBoost confidence < 0.55: lean more on Z-score
    - Always take the STRICTER (higher) tier when they disagree by 2+ levels
      (safety-first: better to flag too early than miss something)
    """
    xgb_idx    = TIER_ORDER.get(xgb_tier, 0)
    zscore_idx = TIER_ORDER.get(zscore_tier, 0)

    # Cold start — XGBoost only
    if session_count < 5:
        print(f"[Hybrid] Cold start ({session_count} sessions) — using XGBoost only: {xgb_tier}")
        return xgb_tier

    # Both available — weighted combination
    if session_count < 10:
        xgb_w, z_w = 0.70, 0.30
    else:
        xgb_w, z_w = 0.50, 0.50

    # Low confidence XGBoost → shift weight to z-score
    if xgb_confidence < 0.55:
        xgb_w, z_w = max(0.3, xgb_w - 0.2), min(0.7, z_w + 0.2)

    weighted = xgb_w * xgb_idx + z_w * zscore_idx
    blended_idx = int(round(weighted))

    # Safety rule: if they disagree by 2+ tiers, take the stricter one
    if abs(xgb_idx - zscore_idx) >= 2:
        blended_idx = max(xgb_idx, zscore_idx)
        print(f"[Hybrid] Large disagreement ({xgb_tier} vs {zscore_tier}) — taking stricter tier")

    final = TIER_NAMES[min(blended_idx, 3)]
    print(f"[Hybrid] XGB={xgb_tier} Z-Score={zscore_tier} → Final={final} (weights: {xgb_w:.0%}/{z_w:.0%})")
    return final


# ══════════════════════════════════════════════════════════════════════════════
# ── 4. Confidence Intervals (for dashboard display) ──────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def compute_confidence_intervals(user_id: str) -> dict:
    past_sessions = load_sessions(user_id)
    if len(past_sessions) < 3:
        return {}

    intervals = {}
    for biomarker in BIOMARKERS:
        values = [
            float(s["biomarkers"][biomarker])
            for s in past_sessions
            if biomarker in s["biomarkers"] and s["biomarkers"][biomarker] not in (None, 0.0)
        ]
        if len(values) < 3:
            continue
        mean = np.mean(values)
        std  = np.std(values)
        intervals[biomarker] = {
            "mean":     round(mean, 4),
            "std":      round(std, 4),
            "lower_95": round(mean - 1.96 * std, 4),
            "upper_95": round(mean + 1.96 * std, 4),
        }
    return intervals


# ══════════════════════════════════════════════════════════════════════════════
# ── 5. Main Pipeline Entry Point ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def run_anomaly_pipeline(user_id: str, current_biomarkers: dict) -> dict:
    """
    Full hybrid pipeline:
    1. XGBoost prediction (population-level, works from session 1)
    2. Z-score anomaly detection (personal baseline, activates at session 5)
    3. Combine both into final risk tier
    4. Save session to DB
    5. Return flags + tier + confidence intervals + XGBoost metadata
    """
    # Load past sessions to know how many the user has
    past_sessions = load_sessions(user_id)
    session_count = len(past_sessions)
    print(f"\n[Pipeline] User {user_id} — session #{session_count + 1}")

    # Step 1: XGBoost
    xgb_result = predict_xgb(current_biomarkers)
    xgb_tier   = xgb_result["tier"] or "Green"

    # Step 2: Z-score (skips internally if < 5 sessions)
    anomaly_flags = detect_anomalies_zscore(user_id, current_biomarkers)
    zscore_tier   = zscore_to_tier(anomaly_flags)

    # Step 3: Combine
    final_tier = combine_tiers(
        xgb_tier, zscore_tier, session_count, xgb_result["confidence"]
    )

    # Step 4: Confidence intervals
    confidence_intervals = compute_confidence_intervals(user_id)

    # Step 5: Save
    save_session(user_id, current_biomarkers, final_tier, anomaly_flags)

    print(f"[Pipeline] ✅ Final risk tier: {final_tier}\n")

    return {
        "anomaly_flags":        anomaly_flags,
        "risk_tier":            final_tier,
        "confidence_intervals": confidence_intervals,
        # Extra metadata for debugging / dashboard display
        "xgb": {
            "tier":          xgb_tier,
            "confidence":    xgb_result["confidence"],
            "probabilities": xgb_result["probabilities"],
        },
        "zscore_tier":          zscore_tier,
        "session_count":        session_count + 1,
        "method": (
            "xgboost_only"  if session_count < 5
            else "hybrid_70_30" if session_count < 10
            else "hybrid_50_50"
        ),
    }