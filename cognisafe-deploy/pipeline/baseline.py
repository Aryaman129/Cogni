"""
baseline.py — Personal historical baseline computation and longitudinal reference bands.

Implements Section 7 requirements:
1. Configurable session threshold distinguishing preliminary (<5) vs mature (≥5) baselines.
2. Accurately labels mean ± 1.96 * std as Historical Reference Bands (reference_band_95), NOT confidence intervals.
3. Zero-variance handling without division-by-zero or silent skipping.
4. Missing feature exclusion: missing/invalid values are NEVER treated as 0.0.
5. Integration with TypedFeatureResult and BiomarkerCollectionResult.
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
        FeatureStatus,
        TypedFeatureResult,
        BiomarkerCollectionResult,
    )
except ImportError:
    # Direct import fallback when run within pipeline directory
    from schemas import (
        CANONICAL_BIOMARKERS,
        FeatureStatus,
        TypedFeatureResult,
        BiomarkerCollectionResult,
    )

# ── Defaults & Paths ──────────────────────────────────────────────────────────
_DIR = os.path.dirname(__file__)
DEFAULT_DB_PATH = os.path.join(_DIR, "..", "data", "sessions.db")
DEFAULT_MIN_SESSIONS_FOR_MATURITY: int = 5
DEFAULT_VARIANCE_FLOOR: float = 1e-6
REFERENCE_BAND_Z_MULTIPLIER: float = 1.96  # Covering ~95% of individual observations under normality


def is_baseline_mature(session_count: int, threshold: int = DEFAULT_MIN_SESSIONS_FOR_MATURITY) -> bool:
    """
    Check if a patient has accumulated enough sessions for a clinically mature personal baseline.
    Preliminary baselines (<5 sessions) must not be used to compute clinical risk tiers.
    """
    return session_count >= threshold


def extract_numeric_value(val: Any) -> Optional[float]:
    """
    Safely extract a finite numeric float from a biomarker entry.
    Handles None, TypedFeatureResult, raw floats, and legacy dict structures.
    NEVER silently coerces None, NaN, inf, or failed extractions to 0.0.
    """
    if val is None:
        return None

    # Handle TypedFeatureResult
    if isinstance(val, TypedFeatureResult):
        if val.status == FeatureStatus.VALID and val.value is not None:
            v = float(val.value)
            return None if (math.isnan(v) or math.isinf(v)) else v
        return None

    # Handle dictionary representation of TypedFeatureResult
    if isinstance(val, dict):
        status = val.get("status")
        if status is not None and status not in (FeatureStatus.VALID, FeatureStatus.VALID.value, "valid"):
            return None
        v = val.get("value")
        if v is None:
            return None
        try:
            fv = float(v)
            return None if (math.isnan(fv) or math.isinf(fv)) else fv
        except (ValueError, TypeError):
            return None

    # Handle raw numeric types
    if isinstance(val, (int, float)):
        fv = float(val)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv

    try:
        fv = float(val)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    except (ValueError, TypeError):
        return None


def init_db(db_path: Optional[str] = None) -> str:
    """Ensure sessions SQLite database and schema exist."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT    NOT NULL,
            timestamp     TEXT    NOT NULL,
            biomarkers    TEXT    NOT NULL,
            risk_tier     TEXT,
            anomaly_flags TEXT    NOT NULL,
            analysis_status TEXT  DEFAULT 'complete'
        )
    ''')
    conn.commit()

    # Ensure analysis_status column exists in legacy databases
    cur.execute("PRAGMA table_info(sessions)")
    columns = [row[1] for row in cur.fetchall()]
    if "analysis_status" not in columns:
        cur.execute("ALTER TABLE sessions ADD COLUMN analysis_status TEXT DEFAULT 'complete'")
        conn.commit()

    conn.close()
    return path


def save_session(
    user_id: str,
    biomarkers: Union[Dict[str, Any], BiomarkerCollectionResult],
    risk_tier: Optional[str] = None,
    anomaly_flags: Optional[List[Dict[str, Any]]] = None,
    analysis_status: str = "complete",
    db_path: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> int:
    """
    Save a completed session to SQLite storage.
    Supports nullable risk_tier and explicit analysis_status.
    """
    path = init_db(db_path)
    if anomaly_flags is None:
        anomaly_flags = []

    # Flatten BiomarkerCollectionResult to dict if needed
    if isinstance(biomarkers, BiomarkerCollectionResult):
        serializable_biomarkers = biomarkers.to_flat_dict()
    elif isinstance(biomarkers, dict):
        serializable_biomarkers = {
            k: extract_numeric_value(v) for k, v in biomarkers.items()
        }
    else:
        serializable_biomarkers = {}

    ts = timestamp or datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions (user_id, timestamp, biomarkers, risk_tier, anomaly_flags, analysis_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(user_id),
            ts,
            json.dumps(serializable_biomarkers),
            risk_tier,
            json.dumps(anomaly_flags),
            analysis_status,
        ),
    )
    session_id = cur.lastrowid or 0
    conn.commit()
    conn.close()
    return session_id


def load_sessions(user_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load all historical sessions for a user ordered chronologically.
    Returns list of dicts with parsed biomarkers and metadata.
    """
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, biomarkers, timestamp, risk_tier, anomaly_flags, analysis_status
        FROM sessions
        WHERE user_id = ?
        ORDER BY timestamp ASC
        """,
        (str(user_id),),
    )
    rows = cur.fetchall()
    conn.close()

    sessions: List[Dict[str, Any]] = []
    for r in rows:
        try:
            b_data = json.loads(r[1]) if r[1] else {}
        except Exception:
            b_data = {}
        try:
            a_flags = json.loads(r[4]) if r[4] else []
        except Exception:
            a_flags = []

        sessions.append({
            "id": r[0],
            "biomarkers": b_data,
            "timestamp": r[2],
            "risk_tier": r[3],
            "anomaly_flags": a_flags,
            "analysis_status": r[5] or "complete",
        })
    return sessions


def calculate_historical_reference_band(
    mean: float,
    std: float,
    multiplier: float = REFERENCE_BAND_Z_MULTIPLIER,
) -> Dict[str, float]:
    """
    Calculate the 95% Historical Reference Band: mean ± 1.96 * std.
    Note: This is an empirical prediction interval covering ~95% of individual observations
    under a Gaussian assumption, NOT a confidence interval for the mean.
    """
    lower = round(mean - multiplier * std, 4)
    upper = round(mean + multiplier * std, 4)
    return {"lower": lower, "upper": upper}


def compute_personal_baseline(
    user_id: str,
    sessions: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[str] = None,
    threshold_sessions: int = DEFAULT_MIN_SESSIONS_FOR_MATURITY,
    variance_floor: float = DEFAULT_VARIANCE_FLOOR,
) -> Dict[str, Any]:
    """
    Compute personal historical baseline across all 14 canonical biomarkers.

    Key guarantees:
    - Distinguishes preliminary (<threshold) vs mature (≥threshold) baselines.
    - Zero-variance handling: detects std < variance_floor without division-by-zero.
    - Missing feature exclusion: missing or null values are omitted, never treated as 0.0.
    - Accurately labels output as Historical Reference Bands (reference_band_95).
    """
    if sessions is None:
        sessions = load_sessions(user_id, db_path=db_path)

    total_sessions = len(sessions)
    mature = is_baseline_mature(total_sessions, threshold=threshold_sessions)

    biomarker_baselines: Dict[str, Any] = {}

    for biomarker in CANONICAL_BIOMARKERS:
        # Collect only valid numerical observations for this biomarker
        valid_values: List[float] = []
        for s in sessions:
            b_map = s.get("biomarkers", {})
            if biomarker in b_map:
                extracted = extract_numeric_value(b_map[biomarker])
                if extracted is not None:
                    valid_values.append(extracted)

        count = len(valid_values)
        if count == 0:
            biomarker_baselines[biomarker] = {
                "count": 0,
                "mean": None,
                "std": None,
                "variance": None,
                "is_zero_variance": False,
                "reference_band_95": None,
                "lower_95": None,
                "upper_95": None,
                "label": "historical_reference_band",
                "status": "no_data",
            }
            continue

        mean_val = float(np.mean(valid_values))
        # Compute sample standard deviation if n >= 2, else 0.0
        if count >= 2:
            std_val = float(np.std(valid_values, ddof=1))
        else:
            std_val = 0.0

        is_zero_var = std_val < variance_floor
        if is_zero_var:
            std_val = 0.0
            variance_val = 0.0
        else:
            variance_val = float(std_val ** 2)

        ref_band = calculate_historical_reference_band(mean_val, std_val)

        biomarker_baselines[biomarker] = {
            "count": count,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "variance": round(variance_val, 4),
            "is_zero_variance": is_zero_var,
            "reference_band_95": ref_band,
            # Backward-compatible convenience aliases
            "lower_95": ref_band["lower"],
            "upper_95": ref_band["upper"],
            "label": "historical_reference_band",
            "status": "mature" if count >= threshold_sessions else "preliminary",
        }

    return {
        "user_id": str(user_id),
        "session_count": total_sessions,
        "is_mature": mature,
        "threshold_sessions": threshold_sessions,
        "biomarkers": biomarker_baselines,
    }


def compute_confidence_intervals(
    user_id: str,
    past_sessions: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[str] = None,
    min_sessions: int = 3,
    threshold_sessions: int = DEFAULT_MIN_SESSIONS_FOR_MATURITY,
) -> Dict[str, Any]:
    """
    Backward-compatible adapter for legacy callers requesting 'confidence_intervals'.
    Returns historical reference bands labeled accurately with reference_band_95.
    Requires at least min_sessions (default 3) before generating intervals.
    """
    if past_sessions is None:
        past_sessions = load_sessions(user_id, db_path=db_path)

    if len(past_sessions) < min_sessions:
        return {}

    baseline_data = compute_personal_baseline(
        user_id,
        sessions=past_sessions,
        db_path=db_path,
        threshold_sessions=threshold_sessions,
    )

    result: Dict[str, Any] = {}
    for bm, info in baseline_data["biomarkers"].items():
        if info["count"] >= min_sessions and info["mean"] is not None:
            result[bm] = {
                "mean": info["mean"],
                "std": info["std"],
                "reference_band_95": info["reference_band_95"],
                "lower_95": info["lower_95"],
                "upper_95": info["upper_95"],
                "sample_count": info["count"],
                "is_zero_variance": info["is_zero_variance"],
                "label": "historical_reference_band",
            }

    return result
