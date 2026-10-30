"""
Unit Tests for CogniSafe Personal Baselines, Anomaly Detection, and Risk Tiering.

Verifies:
1. Preliminary (<5) vs mature (≥5) baselines.
2. Decoupled analysis_status and nullable risk_tier.
3. Never defaulting to "Green" on empty/corrupt/insufficient inputs.
4. Clean zero-variance handling (std == 0.0) without division-by-zero or crashes.
5. Missing feature exclusion (NEVER treated as 0.0, avoiding false 11σ anomalies).
6. Longitudinal Historical Reference Bands (mean ± 1.96 * std) vs confidence intervals.
7. Decommissioning of synthetic XGBoost from clinical risk determination.
8. Rule-based clinical risk scoring (Green: 0 anomalies, Yellow: 1 anomaly, Orange: 2, Red: ≥3 severe).
9. Typed feature integration with TypedFeatureResult and BiomarkerCollectionResult.
"""

import math
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest

# Ensure cognisafe-deploy is in Python path for test discovery
deploy_path = str(Path(__file__).resolve().parent.parent.parent / "cognisafe-deploy")
if deploy_path not in sys.path:
    sys.path.insert(0, deploy_path)

from pipeline.schemas import (
    FeatureStatus,
    TypedFeatureResult,
    BiomarkerCollectionResult,
    CANONICAL_ACOUSTIC_BIOMARKERS,
    CANONICAL_NLP_BIOMARKERS,
    CANONICAL_BIOMARKERS,
)
from pipeline.baseline import (
    is_baseline_mature,
    extract_numeric_value,
    init_db,
    save_session,
    load_sessions,
    compute_personal_baseline,
    compute_confidence_intervals,
    calculate_historical_reference_band,
    DEFAULT_MIN_SESSIONS_FOR_MATURITY,
    DEFAULT_VARIANCE_FLOOR,
)
from pipeline.risk import (
    merge_biomarkers,
    calculate_clinical_risk,
    evaluate_risk_tier,
    determine_analysis_status,
    CANONICAL_RISK_TIERS,
)
from pipeline.anomaly import (
    detect_anomalies_zscore,
    zscore_to_tier,
    combine_tiers,
    predict_xgb,
    run_anomaly_pipeline,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database path for isolated session testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


@pytest.fixture
def mature_sessions_data():
    """Generate 5 baseline sessions with realistic, stable biomarker values."""
    sessions = []
    base_values = {
        "speech_rate": 130.0,
        "articulation_rate": 150.0,
        "pause_frequency": 12.0,
        "pause_duration_mean": 0.45,
        "filled_pause_rate": 1.5,
        "pitch_mean": 175.0,
        "pitch_range": 65.0,
        "jitter": 0.025,
        "shimmer": 0.045,
        "HNR": 18.5,
        "lexical_diversity": 75.0,
        "semantic_coherence": 0.85,
        "idea_density": 0.52,
        "syntactic_complexity": 3.8,
    }
    for i in range(5):
        # Add slight natural jitter
        b_map = {}
        for k, v in base_values.items():
            b_map[k] = round(v + (i - 2) * (v * 0.02), 4)
        sessions.append({
            "id": i + 1,
            "user_id": "test_user_mature",
            "biomarkers": b_map,
            "timestamp": f"2026-10-0{i+1}T10:00:00Z",
            "risk_tier": None if i < 4 else "Green",
            "anomaly_flags": [],
            "analysis_status": "insufficient_baseline" if i < 4 else "complete",
        })
    return sessions


# ══════════════════════════════════════════════════════════════════════════════
# 1. Baseline Maturity Tests (Preliminary <5 vs Mature ≥5)
# ══════════════════════════════════════════════════════════════════════════════

class TestBaselineMaturity:
    """Verify threshold evaluation and maturity classification."""

    @pytest.mark.parametrize("session_count, expected_mature", [
        (0, False),
        (1, False),
        (3, False),
        (4, False),
        (5, True),
        (8, True),
        (30, True),
    ])
    def test_default_maturity_threshold(self, session_count, expected_mature):
        assert is_baseline_mature(session_count) == expected_mature
        assert is_baseline_mature(session_count, threshold=5) == expected_mature

    def test_configurable_maturity_threshold(self):
        assert is_baseline_mature(3, threshold=3) is True
        assert is_baseline_mature(3, threshold=7) is False
        assert is_baseline_mature(10, threshold=10) is True

    def test_personal_baseline_maturity_flag(self, mature_sessions_data):
        # 4 sessions -> preliminary
        preliminary_result = compute_personal_baseline("u1", sessions=mature_sessions_data[:4])
        assert preliminary_result["is_mature"] is False
        assert preliminary_result["session_count"] == 4

        # 5 sessions -> mature
        mature_result = compute_personal_baseline("u1", sessions=mature_sessions_data)
        assert mature_result["is_mature"] is True
        assert mature_result["session_count"] == 5


# ══════════════════════════════════════════════════════════════════════════════
# 2. Historical Reference Bands vs Confidence Intervals
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoricalReferenceBands:
    """Verify mean ± 1.96 * std is accurately labeled as reference bands."""

    @pytest.mark.parametrize("mean, std, expected_lower, expected_upper", [
        (100.0, 10.0, 80.4, 119.6),
        (150.0, 15.0, 120.6, 179.4),
        (175.0, 5.0, 165.2, 184.8),
        (0.5, 0.05, 0.402, 0.598),
    ])
    def test_reference_band_math(self, mean, std, expected_lower, expected_upper):
        band = calculate_historical_reference_band(mean, std)
        assert math.isclose(band["lower"], expected_lower, rel_tol=1e-3)
        assert math.isclose(band["upper"], expected_upper, rel_tol=1e-3)

    def test_reference_band_labeling_in_baseline_output(self, mature_sessions_data):
        baseline = compute_personal_baseline("u1", sessions=mature_sessions_data)
        for bm in CANONICAL_BIOMARKERS:
            bm_data = baseline["biomarkers"][bm]
            assert "reference_band_95" in bm_data
            assert bm_data["label"] == "historical_reference_band"
            ref_band = bm_data["reference_band_95"]
            assert ref_band is not None
            assert "lower" in ref_band
            assert "upper" in ref_band
            assert ref_band["lower"] < bm_data["mean"] < ref_band["upper"]

    def test_confidence_intervals_compatibility_adapter(self, mature_sessions_data):
        # compute_confidence_intervals returns reference bands accurately labeled
        intervals = compute_confidence_intervals("u1", past_sessions=mature_sessions_data)
        assert len(intervals) == 14
        for bm, data in intervals.items():
            assert data["label"] == "historical_reference_band"
            assert "reference_band_95" in data
            assert "lower_95" in data
            assert "upper_95" in data
            assert data["lower_95"] == data["reference_band_95"]["lower"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Decoupling Analysis Status from Risk Tier & No Default Green
# ══════════════════════════════════════════════════════════════════════════════

class TestDecoupledAnalysisStatusAndRiskTier:
    """Verify analysis_status and risk_tier contract: never default to Green."""

    @pytest.mark.parametrize("status, expected_tier", [
        ("insufficient_baseline", None),
        ("insufficient_data", None),
        ("analysis_failed", None),
    ])
    def test_null_risk_tier_on_incomplete_or_insufficient_sessions(self, status, expected_tier):
        if status == "insufficient_baseline":
            stat, tier = determine_analysis_status({"speech_rate": 120.0}, session_count=3)
        elif status == "insufficient_data":
            stat, tier = determine_analysis_status({}, session_count=10)
        else:
            stat, tier = determine_analysis_status({"speech_rate": 120.0}, session_count=10, has_extraction_errors=True)

        assert stat == status
        assert tier is expected_tier
        assert tier != "Green", "Risk tier must NEVER default to Green on incomplete or unestablished baseline!"

    def test_voice_mode_silent_audio_insufficient_data(self):
        # In voice mode, if all acoustic features are missing/None, analysis_status is insufficient_data
        nlp_only = {"lexical_diversity": 80.0, "semantic_coherence": 0.8}
        status, tier = determine_analysis_status(nlp_only, session_count=10, mode="voice")
        assert status == "insufficient_data"
        assert tier is None

    def test_pipeline_under_five_sessions_returns_null_tier(self, temp_db):
        # Run pipeline with 2 existing sessions
        save_session("user_p1", {"speech_rate": 120.0}, db_path=temp_db)
        save_session("user_p1", {"speech_rate": 122.0}, db_path=temp_db)

        current = {"speech_rate": 121.0, "pitch_mean": 170.0}
        result = run_anomaly_pipeline("user_p1", current, db_path=temp_db)

        assert result["analysis_status"] == "insufficient_baseline"
        assert result["risk_tier"] is None
        assert result["risk_tier"] != "Green"
        assert result["is_baseline_mature"] is False
        assert result["anomaly_flags"] == []

    def test_pipeline_empty_input_returns_insufficient_data(self, temp_db, mature_sessions_data):
        # Even with mature baseline, empty input produces insufficient_data and None tier
        for s in mature_sessions_data:
            save_session("user_empty", s["biomarkers"], db_path=temp_db)

        result = run_anomaly_pipeline("user_empty", {}, db_path=temp_db)
        assert result["analysis_status"] == "insufficient_data"
        assert result["risk_tier"] is None
        assert result["risk_tier"] != "Green"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Zero-Variance Handling
# ══════════════════════════════════════════════════════════════════════════════

class TestZeroVarianceHandling:
    """Verify std == 0.0 does not divide by zero or cause unhandled exceptions."""

    def test_baseline_std_zero_clean_handling(self):
        identical_sessions = [
            {"id": i, "biomarkers": {"pitch_mean": 160.0, "speech_rate": 120.0}}
            for i in range(5)
        ]
        baseline = compute_personal_baseline("u_zero", sessions=identical_sessions)
        pitch_info = baseline["biomarkers"]["pitch_mean"]

        assert pitch_info["mean"] == 160.0
        assert pitch_info["std"] == 0.0
        assert pitch_info["is_zero_variance"] is True
        assert pitch_info["reference_band_95"]["lower"] == 160.0
        assert pitch_info["reference_band_95"]["upper"] == 160.0

    def test_zscore_zero_variance_deviation_guarded_to_zero(self):
        # When historical std is 0.0, deviation is guarded to 0.0
        identical_sessions = [
            {"id": i, "biomarkers": {"pitch_mean": 160.0, "speech_rate": 120.0}}
            for i in range(6)
        ]
        current = {"pitch_mean": 160.0, "speech_rate": 120.0}
        anomalies = detect_anomalies_zscore("u_zero", current, past_sessions=identical_sessions)
        # No false anomalies triggered!
        assert len(anomalies) == 0

    def test_zero_variance_boundary_formula(self):
        # Explicit test of boundary formula: std < 1e-6 -> z = 0.0
        mean = 120.0
        std = 0.0
        val = 120.0
        z = (val - mean) / std if std > 1e-6 else 0.0
        assert z == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Missing Feature Handling & No False Anomalies
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingFeatureHandling:
    """Verify missing features are NEVER treated as 0.0 and trigger no false 11σ anomalies."""

    def test_missing_features_excluded_from_baseline_averages(self):
        # 5 sessions where pitch_mean is only measured in 3 sessions
        sessions = [
            {"id": 1, "biomarkers": {"pitch_mean": 150.0}},
            {"id": 2, "biomarkers": {"pitch_mean": 160.0}},
            {"id": 3, "biomarkers": {"pitch_mean": None}},     # Missing
            {"id": 4, "biomarkers": {"pitch_mean": 170.0}},
            {"id": 5, "biomarkers": {}},                       # Omitted
        ]
        baseline = compute_personal_baseline("u_missing", sessions=sessions)
        pitch_info = baseline["biomarkers"]["pitch_mean"]

        assert pitch_info["count"] == 3
        assert pitch_info["mean"] == 160.0  # (150 + 160 + 170) / 3, NOT dragged down by 0.0!

    def test_missing_current_feature_does_not_trigger_false_anomaly(self, mature_sessions_data):
        # User has pitch_mean historical baseline of ~175 Hz.
        # Current session has pitch_mean as None (e.g. unvoiced speech).
        # In the old flawed code, pitch_mean defaulted to 0.0 -> (0 - 175) / 5 = -35σ!
        current_with_missing = {
            "speech_rate": 130.0,
            "pitch_mean": None,  # Missing!
            "HNR": None,         # Missing!
        }
        anomalies = detect_anomalies_zscore(
            "u_mature",
            current_with_missing,
            past_sessions=mature_sessions_data,
        )

        anomaly_biomarkers = [a["biomarker"] for a in anomalies]
        assert "pitch_mean" not in anomaly_biomarkers
        assert "HNR" not in anomaly_biomarkers
        assert len(anomalies) == 0

    def test_merge_biomarkers_preserves_none(self):
        acoustic = {"speech_rate": 125.0, "pitch_mean": None}
        nlp = {"lexical_diversity": 65.0}

        merged = merge_biomarkers(acoustic, nlp)
        assert merged["speech_rate"] == 125.0
        assert merged["pitch_mean"] is None
        assert merged["HNR"] is None
        assert merged["semantic_coherence"] is None
        assert 0.0 not in [merged["pitch_mean"], merged["HNR"]], "Missing features must NOT default to 0.0!"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Decommissioning Synthetic XGBoost from Clinical Risk
# ══════════════════════════════════════════════════════════════════════════════

class TestSupervisedClassifierDecommissioning:
    """Verify synthetic XGBoost is removed from clinical risk and isolated to demo mode."""

    def test_predict_xgb_clinical_mode_returns_decommissioned(self):
        res = predict_xgb({"speech_rate": 120.0}, demo_mode=False)
        assert res["status"] == "decommissioned_from_clinical"
        assert res["tier"] is None
        assert res["is_demo"] is False

    def test_predict_xgb_demo_mode_isolated(self):
        res = predict_xgb({"speech_rate": 120.0}, demo_mode=True)
        assert res["is_demo"] is True
        assert res["tier"] in CANONICAL_RISK_TIERS

    def test_clinical_risk_is_strictly_rule_based(self):
        # 0 anomalies -> Green
        assert calculate_clinical_risk(0, baseline_mature=True) == "Green"
        # 1-2 anomalies -> Yellow
        assert calculate_clinical_risk(1, baseline_mature=True) == "Yellow"
        assert calculate_clinical_risk(2, baseline_mature=True) == "Yellow"
        # 3-4 anomalies -> Orange
        assert calculate_clinical_risk(3, baseline_mature=True) == "Orange"
        assert calculate_clinical_risk(4, baseline_mature=True) == "Orange"
        # 5+ anomalies -> Red
        assert calculate_clinical_risk(5, baseline_mature=True) == "Red"
        # Premature baseline -> None
        assert calculate_clinical_risk(5, baseline_mature=False) is None

    def test_severe_anomaly_count_clinical_risk(self):
        # Clinical risk counting severe deviations (>=2.5σ or 3.0σ)
        assert calculate_clinical_risk(0, baseline_mature=True, severe_count=0) == "Green"
        assert calculate_clinical_risk(1, baseline_mature=True, severe_count=1) == "Yellow"
        assert calculate_clinical_risk(2, baseline_mature=True, severe_count=2) == "Orange"
        assert calculate_clinical_risk(3, baseline_mature=True, severe_count=3) == "Red"
        assert calculate_clinical_risk(5, baseline_mature=True, severe_count=4) == "Red"
        assert calculate_clinical_risk(3, baseline_mature=False, severe_count=3) is None

    def test_evaluate_risk_tier_from_flags(self):
        flags_green = []
        assert evaluate_risk_tier(flags_green, baseline_mature=True) == "Green"
        assert evaluate_risk_tier(flags_green, baseline_mature=False) is None

        flags_yellow = [{"biomarker": "speech_rate", "severity": "moderate", "deviation": 2.6}]
        assert evaluate_risk_tier(flags_yellow, baseline_mature=True) == "Yellow"

        flags_orange = [
            {"biomarker": "speech_rate", "severity": "severe", "deviation": 3.2},
            {"biomarker": "pause_frequency", "severity": "moderate", "deviation": 2.7},
        ]
        assert evaluate_risk_tier(flags_orange, baseline_mature=True) == "Orange"

        flags_red = [
            {"biomarker": "speech_rate", "severity": "severe", "deviation": 3.5},
            {"biomarker": "pause_frequency", "severity": "severe", "deviation": 3.8},
            {"biomarker": "pitch_range", "severity": "severe", "deviation": 3.1},
        ]
        assert evaluate_risk_tier(flags_red, baseline_mature=True) == "Red"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Typed Feature Result Integration (M1 Schemas)
# ══════════════════════════════════════════════════════════════════════════════

class TestTypedFeatureIntegration:
    """Verify interoperability with TypedFeatureResult and BiomarkerCollectionResult."""

    def test_merge_biomarkers_typed_collection(self):
        acoustic_res = BiomarkerCollectionResult()
        acoustic_res["speech_rate"] = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=132.0,
            unit="words/min",
            method="test_vad",
        )
        acoustic_res["pitch_mean"] = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="Hz",
            method="pyin",
            reason="unvoiced",
        )

        nlp_res = BiomarkerCollectionResult()
        nlp_res["lexical_diversity"] = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=82.5,
            unit="mtld",
            method="mtld",
        )

        merged_typed = merge_biomarkers(acoustic_res, nlp_res, return_typed=True)
        assert isinstance(merged_typed, BiomarkerCollectionResult)
        assert merged_typed["speech_rate"].value == 132.0
        assert merged_typed["speech_rate"].status == FeatureStatus.VALID
        assert merged_typed["pitch_mean"].value is None
        assert merged_typed["pitch_mean"].status == FeatureStatus.INSUFFICIENT_INPUT
        assert merged_typed["lexical_diversity"].value == 82.5

    def test_anomaly_detection_with_biomarker_collection_result(self, mature_sessions_data):
        current_collection = BiomarkerCollectionResult()
        # Normal speech rate
        current_collection["speech_rate"] = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=130.0,
            unit="words/min",
            method="test",
        )
        # Severe deviation in pitch_mean (baseline ~175 Hz, current 260 Hz -> ~17σ)
        current_collection["pitch_mean"] = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=260.0,
            unit="Hz",
            method="test",
        )
        # Unvoiced/insufficient HNR -> must be ignored, not triggered as anomaly
        current_collection["HNR"] = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="dB",
            method="test",
        )

        anomalies = detect_anomalies_zscore(
            "u_mature",
            current_collection,
            past_sessions=mature_sessions_data,
        )

        flagged = {a["biomarker"]: a for a in anomalies}
        assert "pitch_mean" in flagged
        assert flagged["pitch_mean"]["severity"] == "severe"
        assert flagged["pitch_mean"]["deviation"] > 3.0
        assert "speech_rate" not in flagged
        assert "HNR" not in flagged


# ══════════════════════════════════════════════════════════════════════════════
# 8. End-to-End Anomaly Pipeline Integration
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineEndToEnd:
    """Verify run_anomaly_pipeline from cold-start to mature longitudinal monitoring."""

    def test_full_pipeline_progression(self, temp_db):
        user_id = "patient_progression"

        # Sessions 1-4: Baseline establishing
        for i in range(4):
            features = {
                "speech_rate": 120.0 + i,
                "pause_frequency": 10.0,
                "pitch_mean": 160.0,
            }
            res = run_anomaly_pipeline(user_id, features, db_path=temp_db)
            assert res["analysis_status"] == "insufficient_baseline"
            assert res["risk_tier"] is None
            assert res["is_baseline_mature"] is False
            assert res["session_count"] == i + 1

        # Session 5: Baseline reaches maturity!
        normal_s5 = {
            "speech_rate": 122.0,
            "pause_frequency": 10.0,
            "pitch_mean": 160.0,
        }
        res5 = run_anomaly_pipeline(user_id, normal_s5, db_path=temp_db)
        assert res5["analysis_status"] == "complete"
        assert res5["risk_tier"] == "Green"
        assert res5["is_baseline_mature"] is True
        assert res5["session_count"] == 5
        assert len(res5["anomaly_flags"]) == 0

        # Session 6: Severe cognitive drift (abnormally slow speech rate + excessive pauses)
        drift_s6 = {
            "speech_rate": 60.0,        # Dramatic slowdown
            "pause_frequency": 35.0,    # Dramatic pause increase
            "pitch_mean": 160.0,        # Normal pitch
        }
        res6 = run_anomaly_pipeline(user_id, drift_s6, db_path=temp_db)
        assert res6["analysis_status"] == "complete"
        assert res6["risk_tier"] in ("Orange", "Red")
        assert len(res6["anomaly_flags"]) >= 2
        flagged_names = [f["biomarker"] for f in res6["anomaly_flags"]]
        assert "speech_rate" in flagged_names
        assert "pause_frequency" in flagged_names
