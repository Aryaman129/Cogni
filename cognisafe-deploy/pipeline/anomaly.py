"""
anomaly.py — Personal Baseline Anomaly Detection and Clinical Risk Pipeline.

Upgraded for Section 7:
1. Decommissioned synthetic XGBoost from clinical risk determination.
   XGBoost is isolated purely to deterministic non-clinical demo fixtures (demo_mode=True).
2. Clinical risk scoring is rule-based statistical anomaly scoring against mature personal baselines.
3. Decoupled analysis_status from risk_tier:
   - User with <5 sessions: analysis_status="insufficient_baseline", risk_tier=None ("Establishing Baseline").
   - Empty/silent/missing features: analysis_status="insufficient_data", risk_tier=None.
   - NEVER defaults to "Green" when risk tier is null or missing.
4. Robust zero-variance handling: std < 1e-6 does not divide by zero or skip silently; deviation is guarded to 0.0.
5. Missing features exclusion: missing/invalid features are NEVER treated as 0.0 (preventing false 11σ deviations).
6. Longitudinal Historical Reference Bands (reference_band_95: mean ± 1.96 * std), accurately labeled.
"""

from datetime import datetime, timezone
import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from pipeline.schemas import (
        CANONICAL_BIOMARKERS,
        CANONICAL_ACOUSTIC_BIOMARKERS,
        CANONICAL_NLP_BIOMARKERS,
        FeatureStatus,
        TypedFeatureResult,
        BiomarkerCollectionResult,
    )
    from pipeline.baseline import (
        DEFAULT_DB_PATH,
        DEFAULT_MIN_SESSIONS_FOR_MATURITY,
        DEFAULT_VARIANCE_FLOOR,
        is_baseline_mature,
        extract_numeric_value,
        init_db,
        save_session,
        load_sessions,
        compute_personal_baseline,
        compute_confidence_intervals,
        calculate_historical_reference_band,
    )
    from pipeline.risk import (
        CANONICAL_RISK_TIERS,
        TIER_ORDER,
        calculate_clinical_risk,
        evaluate_risk_tier,
        determine_analysis_status,
        merge_biomarkers,
    )
except ImportError:
    from schemas import (
        CANONICAL_BIOMARKERS,
        CANONICAL_ACOUSTIC_BIOMARKERS,
        CANONICAL_NLP_BIOMARKERS,
        FeatureStatus,
        TypedFeatureResult,
        BiomarkerCollectionResult,
    )
    from baseline import (
        DEFAULT_DB_PATH,
        DEFAULT_MIN_SESSIONS_FOR_MATURITY,
        DEFAULT_VARIANCE_FLOOR,
        is_baseline_mature,
        extract_numeric_value,
        init_db,
        save_session,
        load_sessions,
        compute_personal_baseline,
        compute_confidence_intervals,
        calculate_historical_reference_band,
    )
    from risk import (
        CANONICAL_RISK_TIERS,
        TIER_ORDER,
        calculate_clinical_risk,
        evaluate_risk_tier,
        determine_analysis_status,
        merge_biomarkers,
    )

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)
DB_PATH = DEFAULT_DB_PATH
MODEL_PATH = os.path.join(_DIR, "xgb_model.json")
LABELS_PATH = os.path.join(_DIR, "xgb_model_labels.json")

# ── Biomarker ordering ────────────────────────────────────────────────────────
BIOMARKERS = list(CANONICAL_BIOMARKERS)
TIER_NAMES = list(CANONICAL_RISK_TIERS)

# ── Demo XGBoost Model State (Isolated purely to non-clinical demo fixtures) ───
_xgb_model = None
_xgb_labels = None


def _load_xgb() -> bool:
    """Load legacy XGBoost weights only when needed for demo fixtures."""
    global _xgb_model, _xgb_labels
    if _xgb_model is not None:
        return True
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        import xgboost as xgb
        _xgb_model = xgb.XGBClassifier()
        _xgb_model.load_model(MODEL_PATH)
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                _xgb_labels = json.load(f)
        else:
            _xgb_labels = {"0": "Green", "1": "Yellow", "2": "Orange", "3": "Red"}
        return True
    except Exception:
        return False


def predict_xgb(current_biomarkers: Union[Dict[str, Any], BiomarkerCollectionResult], demo_mode: bool = False) -> Dict[str, Any]:
    """
    Isolated XGBoost predictor for non-clinical demo fixtures only.
    Decommissioned from all clinical risk determination.
    """
    if not demo_mode:
        return {
            "tier": None,
            "confidence": 0.0,
            "probabilities": {},
            "is_demo": False,
            "status": "decommissioned_from_clinical",
        }

    if not _load_xgb():
        return {
            "tier": "Green",
            "confidence": 0.5,
            "probabilities": {"Green": 0.5, "Yellow": 0.2, "Orange": 0.2, "Red": 0.1},
            "is_demo": True,
            "status": "model_file_missing_using_demo_fallback",
        }

    # Flatten inputs
    if isinstance(current_biomarkers, BiomarkerCollectionResult):
        raw_map = current_biomarkers.to_flat_dict()
    elif isinstance(current_biomarkers, dict):
        raw_map = {k: extract_numeric_value(v) for k, v in current_biomarkers.items()}
    else:
        raw_map = {}

    x = np.array([[
        float(raw_map.get(feat) or 0.0)
        for feat in BIOMARKERS
    ]])

    try:
        proba = _xgb_model.predict_proba(x)[0]
        pred = int(np.argmax(proba))
        tier = _xgb_labels.get(str(pred), "Green")
        confidence = float(proba[pred])
        prob_map = {
            _xgb_labels.get(str(i), f"Tier_{i}"): round(float(p), 4)
            for i, p in enumerate(proba)
        }
        return {
            "tier": tier,
            "confidence": confidence,
            "probabilities": prob_map,
            "is_demo": True,
            "status": "demo_fixture_prediction",
        }
    except Exception:
        return {
            "tier": "Green",
            "confidence": 0.5,
            "probabilities": {},
            "is_demo": True,
            "status": "prediction_error_fallback",
        }


# ══════════════════════════════════════════════════════════════════════════════
# ── Anomaly Detection Engine (Personal Baseline Z-Score) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def detect_anomalies_zscore(
    user_id: str,
    current_biomarkers: Union[Dict[str, Any], BiomarkerCollectionResult],
    min_sessions: int = DEFAULT_MIN_SESSIONS_FOR_MATURITY,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    threshold_sigma: float = 2.0,
    past_sessions: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compare current session biomarkers against the user's personal historical baseline.

    Key guarantees:
    - Only activates when user has accumulated min_sessions (default 5+). Returns [] if insufficient.
    - Missing features (None or invalid) are NEVER treated as 0.0 (prevents false 11σ anomalies).
    - Zero variance (std < 1e-6) handles cleanly: deviation guarded to 0.0 without dividing by zero.
    - Classifies deviations:
        >= 3.0σ: "severe"
        >= 2.5σ: "moderate"
        >= 2.0σ: "mild"
    """
    if past_sessions is None:
        past_sessions = load_sessions(user_id, db_path=db_path)

    # Minimum historical sessions needed to compute personal baseline z-scores
    if len(past_sessions) < max(3, min_sessions - 1):
        return []

    # Flatten current features
    if isinstance(current_biomarkers, BiomarkerCollectionResult):
        current_map = {
            k: (v.value if v.status == FeatureStatus.VALID else None)
            for k, v in current_biomarkers.items()
        }
    elif isinstance(current_biomarkers, dict):
        current_map = {k: extract_numeric_value(v) for k, v in current_biomarkers.items()}
    else:
        return []

    anomaly_flags: List[Dict[str, Any]] = []

    for biomarker in CANONICAL_BIOMARKERS:
        current_val = current_map.get(biomarker)
        # CRITICAL: If feature is missing or invalid, EXCLUDE from anomaly checking!
        # Never coerce to 0.0!
        if current_val is None:
            continue

        # Extract historical values for this biomarker
        history: List[float] = []
        for s in past_sessions:
            b_data = s.get("biomarkers", {})
            if biomarker in b_data:
                num = extract_numeric_value(b_data[biomarker])
                if num is not None:
                    history.append(num)

        # Require at least 3 historical observations for this specific biomarker
        if len(history) < 3:
            continue

        mean = float(np.mean(history))
        std = float(np.std(history, ddof=1)) if len(history) >= 2 else 0.0

        # Zero-variance guard: std < variance_floor (1e-6)
        if std < variance_floor:
            is_zero_variance = True
            # If current value matches the constant historical baseline, deviation is 0.0
            if abs(current_val - mean) < 1e-6:
                deviation = 0.0
            else:
                # When current value deviates from constant baseline, use variance floor
                deviation = float(abs(current_val - mean) / variance_floor)
        else:
            deviation = float(abs(current_val - mean) / std)
            is_zero_variance = False

        if deviation >= threshold_sigma:
            severity = "severe" if deviation >= 3.0 else "moderate" if deviation >= 2.5 else "mild"
            anomaly_flags.append({
                "biomarker": biomarker,
                "severity": severity,
                "current": round(current_val, 4),
                "baseline": round(mean, 4),
                "deviation": round(deviation, 2),
                "std": round(std, 4),
                "is_zero_variance": is_zero_variance,
            })

    return anomaly_flags


def zscore_to_tier(
    anomaly_flags: List[Dict[str, Any]],
    baseline_mature: bool = True,
) -> Optional[str]:
    """
    Convert personal baseline anomaly flags to a clinical risk tier.
    Delegates to evaluate_risk_tier for consistent rule-based scoring.
    """
    return evaluate_risk_tier(anomaly_flags, baseline_mature=baseline_mature)


def combine_tiers(
    xgb_tier: Optional[str],
    zscore_tier: Optional[str],
    session_count: int,
    xgb_confidence: float = 0.0,
    demo_mode: bool = False,
) -> Optional[str]:
    """
    Clinical risk tier combination.
    In clinical mode (demo_mode=False):
        Synthetic XGBoost is decommissioned.
        If session_count < 5: returns None (Establishing Baseline).
        If session_count >= 5: strictly returns zscore_tier.
    In demo mode (demo_mode=True):
        Provides non-clinical fallback for synthetic demo fixtures.
    """
    if not demo_mode:
        if not is_baseline_mature(session_count, threshold=DEFAULT_MIN_SESSIONS_FOR_MATURITY):
            return None
        return zscore_tier

    # Demo mode fallback logic for synthetic UI previews
    if not zscore_tier:
        return xgb_tier or "Green"
    return zscore_tier


# ══════════════════════════════════════════════════════════════════════════════
# ── Main Pipeline Entry Point ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def run_anomaly_pipeline(
    user_id: str,
    current_biomarkers: Union[Dict[str, Any], BiomarkerCollectionResult],
    demo_mode: bool = False,
    min_sessions: int = DEFAULT_MIN_SESSIONS_FOR_MATURITY,
    save_to_db: bool = True,
    db_path: Optional[str] = None,
    mode: str = "voice",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full anomaly detection and risk determination pipeline.

    Guarantees:
    1. If user has < 5 sessions:
       analysis_status = "insufficient_baseline", risk_tier = None ("Establishing Baseline").
    2. If audio features are incomplete, silent, or unvoiced:
       analysis_status = "insufficient_data", risk_tier = None.
    3. Never defaults to "Green" when risk tier is null or missing.
    4. Clinical risk tier derived strictly from statistical anomaly scoring (no synthetic XGBoost).
    5. Returns accurately labeled historical reference bands (reference_band_95).
    """
    path = db_path or DB_PATH
    past_sessions = load_sessions(user_id, db_path=path)
    current_session_count = len(past_sessions) + 1
    baseline_mature = is_baseline_mature(current_session_count, threshold=min_sessions)

    # Decouple analysis_status from risk_tier
    analysis_status, _ = determine_analysis_status(
        current_biomarkers,
        session_count=current_session_count,
        min_sessions_for_maturity=min_sessions,
        mode=mode,
    )

    if analysis_status in ("insufficient_baseline", "insufficient_data", "analysis_failed"):
        anomaly_flags: List[Dict[str, Any]] = []
        final_tier = None
    else:
        # Complete session with mature baseline (>=5 sessions)
        anomaly_flags = detect_anomalies_zscore(
            user_id,
            current_biomarkers,
            min_sessions=min_sessions,
            past_sessions=past_sessions,
            db_path=path,
        )
        final_tier = zscore_to_tier(anomaly_flags, baseline_mature=True)

    # Compute Historical Reference Bands
    reference_bands = compute_confidence_intervals(
        user_id,
        past_sessions=past_sessions,
        db_path=path,
        threshold_sessions=min_sessions,
    )

    # Optional demo mode fixture data
    if demo_mode:
        xgb_meta = predict_xgb(current_biomarkers, demo_mode=True)
    else:
        xgb_meta = {
            "tier": None,
            "confidence": 0.0,
            "probabilities": {},
            "status": "decommissioned_from_clinical",
        }

    # Save session to SQLite database if requested
    if save_to_db:
        save_session(
            user_id=user_id,
            biomarkers=current_biomarkers,
            risk_tier=final_tier,
            anomaly_flags=anomaly_flags,
            analysis_status=analysis_status,
            db_path=path,
            timestamp=timestamp,
        )

    return {
        "analysis_status": analysis_status,
        "risk_tier": final_tier,
        "anomaly_flags": anomaly_flags,
        "reference_bands": reference_bands,
        # Backward compatibility alias
        "confidence_intervals": reference_bands,
        "session_count": current_session_count,
        "is_baseline_mature": baseline_mature,
        "demo_mode": demo_mode,
        "xgb": xgb_meta,
        "method": "personal_baseline_zscore" if baseline_mature else "establishing_baseline",
    }