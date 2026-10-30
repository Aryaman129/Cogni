"""
tests/e2e/test_challenger1_speech_math.py
=========================================
Empirical Adversarial Verification Suite by Challenger 1 (Milestone 6).

Adversarially stress-tests:
1. Speech Processing & Acoustic Pipeline:
   - 0-byte audio files
   - Corrupted/truncated RIFF/WAV headers
   - Pure digital silence (0.0 amplitude)
   - Saturated clipping (>99% of samples pegged at rails)
   - Sub-second audio (0.05s - 0.49s)
   - Extreme sample rates (4,000 Hz telephony to 192,000 Hz studio)
2. Linguistic & NLP Pipeline:
   - Empty, whitespace, and zero-width strings
   - Single-word and very short transcripts
   - Repetitive loops ("word "*500)
   - Extreme emojis, control characters, and Unicode formatting symbols
   - Windows cp1252 encoding traps (smart quotes, em-dashes, non-Latin scripts)
3. Mathematical Baselines & Anomaly Detection:
   - Zero-variance personal baselines (std == 0.0)
   - Intermittent and missing features across session history
   - Injection of NaN, Inf, and non-numeric values
   - Strict cold-start non-Green enforcement (0 to 4 sessions produce null risk tier)
   - Verification that synthetic XGBoost is strictly decommissioned from clinical risk scoring.
"""

import io
import math
import os
import struct
import sys
import tempfile
import wave
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pytest

# Ensure cognisafe-deploy is in Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEPLOY_DIR = os.path.join(BASE_DIR, "cognisafe-deploy")
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

from pipeline.schemas import (
    FeatureStatus,
    TypedFeatureResult,
    BiomarkerCollectionResult,
    CANONICAL_ACOUSTIC_BIOMARKERS,
    CANONICAL_NLP_BIOMARKERS,
    CANONICAL_BIOMARKERS,
)
from pipeline.audio_validator import (
    AudioValidationStatus,
    AudioValidationResult,
    validate_audio_file,
    detect_audio_format_from_bytes,
)
from pipeline.acoustic import extract_acoustic_features
from pipeline.nlp import (
    analyze_text,
    compute_mtld,
    compute_semantic_coherence,
    compute_idea_density,
    compute_syntactic_complexity,
)
from pipeline.transcription import transcribe_audio
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


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Audio Helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_test_db():
    """Create an isolated temporary SQLite database for baseline tests."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def generate_adversarial_wav(
    duration_s: float = 3.0,
    sample_rate: int = 16000,
    freq_hz: float = 220.0,
    silence: bool = False,
    clipped: bool = False,
    amplitude: float = 0.5,
) -> str:
    """Write temporary WAV file with exact signal characteristics."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    total_samples = max(1, int(duration_s * sample_rate))

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        if silence:
            raw_data = b"\x00\x00" * total_samples
        elif clipped:
            # Peg samples to extreme limits (>99% clipped)
            samples = [
                struct.pack("<h", 32767 if (i % 2 == 0) else -32768)
                for i in range(total_samples)
            ]
            raw_data = b"".join(samples)
        else:
            samples = []
            for i in range(total_samples):
                t = float(i) / sample_rate
                val = amplitude * math.sin(2.0 * math.pi * freq_hz * t)
                int_val = int(val * 32767)
                int_val = max(-32768, min(32767, int_val))
                samples.append(struct.pack("<h", int_val))
            raw_data = b"".join(samples)

        wf.writeframes(raw_data)

    return path


# ═════════════════════════════════════════════════════════════════════════════
# 1. ADVERSARIAL AUDIO STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAdversarialAudio:
    """Stress-test the audio validation, acoustic extraction, and transcription pipeline."""

    def test_zero_byte_audio_handling(self):
        """0-byte file must be rejected cleanly without unhandled exceptions or silent 0.0s."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            val_res = validate_audio_file(path)
            assert not val_res.is_valid
            assert val_res.status == AudioValidationStatus.EMPTY_FILE
            assert val_res.file_size_bytes == 0

            # Direct extraction must handle 0-byte file without crashing
            res = extract_acoustic_features(path, return_typed=True)
            assert res.overall_status in ("failed", "insufficient_data")
            assert len(res.features) == 10

            for bm_name, feat in res.features.items():
                assert feat.value is None, f"Biomarker {bm_name} silently output non-None value {feat.value} on 0-byte audio!"
                assert feat.status != FeatureStatus.VALID

            # Anomaly pipeline must produce null risk tier
            anomaly_res = run_anomaly_pipeline("test_user_0byte", res)
            assert anomaly_res["risk_tier"] is None
            assert anomaly_res["analysis_status"] in ("insufficient_data", "analysis_failed")
            assert anomaly_res["risk_tier"] != "Green", "Silently defaulted to Green on 0-byte file!"

        finally:
            if os.path.exists(path):
                os.remove(path)

    @pytest.mark.parametrize("corrupt_bytes, label", [
        (b"RIFF", "Truncated 4-byte RIFF"),
        (b"RIFF\x08\x00\x00\x00WAV", "Truncated 11-byte WAVE header"),
        (b"NOT_A_WAV_HEADER_DATA_1234567890", "Random invalid magic bytes"),
        (b"\x00" * 64, "All zeros 64-byte header"),
        (b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00" + b"\xff" * 20, "Corrupted PCM payload"),
        (b"RIFF\xff\xff\xff\x7fWAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00", "2GB claimed chunk with 0 data"),
    ])
    def test_corrupted_riff_headers(self, corrupt_bytes, label):
        """Corrupted/truncated RIFF headers must fail gracefully without crash or silent 0.0."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with open(path, "wb") as f:
                f.write(corrupt_bytes)

            val_res = validate_audio_file(path)
            assert not val_res.is_valid, f"Corrupted header treated as valid: {label}"

            # Direct extraction must catch and return null values
            res = extract_acoustic_features(path, return_typed=True)
            assert res.overall_status in ("failed", "insufficient_data")
            for bm_name, feat in res.features.items():
                assert feat.value is None, f"{bm_name} produced {feat.value} on corrupt header {label}"
                assert feat.status in (FeatureStatus.EXTRACTION_FAILED, FeatureStatus.INSUFFICIENT_INPUT)

        finally:
            os.close(fd)
            if os.path.exists(path):
                os.remove(path)

    def test_pure_digital_silence(self):
        """Pure digital silence (all zeros) must report silence, never 0.0 or Green tier."""
        path = generate_adversarial_wav(duration_s=3.0, silence=True)
        try:
            val_res = validate_audio_file(path)
            assert not val_res.is_valid
            assert val_res.status == AudioValidationStatus.SILENT
            assert val_res.max_amplitude == 0.0

            res = extract_acoustic_features(path, return_typed=True)
            assert res.overall_status == "insufficient_data"

            # Check every acoustic biomarker
            for bm_name, feat in res.features.items():
                assert feat.value is None, f"Biomarker {bm_name} output {feat.value} on pure silence!"
                assert feat.status == FeatureStatus.INSUFFICIENT_INPUT

            # Feed to anomaly pipeline
            anomaly_res = run_anomaly_pipeline("user_silence", res)
            assert anomaly_res["risk_tier"] is None
            assert anomaly_res["analysis_status"] == "insufficient_data"
            assert anomaly_res["risk_tier"] != "Green", "Silent audio defaulted to Green risk tier!"

        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_saturated_clipping_detection_and_handling(self):
        """Severely clipped audio (>99% samples at rail) must be flagged and handled safely."""
        path = generate_adversarial_wav(duration_s=3.0, clipped=True)
        try:
            val_res = validate_audio_file(path)
            assert not val_res.is_valid
            assert val_res.status == AudioValidationStatus.CLIPPED
            assert val_res.clipping_ratio > 0.95

            # If forced extraction on clipped audio, pipeline must not crash or output silent 0.0
            res = extract_acoustic_features(path, return_typed=True)
            assert res.overall_status in ("insufficient_data", "failed")
            for bm_name, feat in res.features.items():
                if feat.value is not None:
                    assert not math.isnan(feat.value)
                    assert not math.isinf(feat.value)

        finally:
            if os.path.exists(path):
                os.remove(path)

    @pytest.mark.parametrize("duration_s", [0.05, 0.1, 0.2, 0.49])
    def test_sub_second_audio_handling(self, duration_s):
        """Sub-second audio (<0.5s min duration) must be rejected as TOO_SHORT without crashing librosa."""
        path = generate_adversarial_wav(duration_s=duration_s, freq_hz=220.0)
        try:
            val_res = validate_audio_file(path, min_duration=0.5)
            assert not val_res.is_valid
            assert val_res.status == AudioValidationStatus.TOO_SHORT
            assert val_res.duration_seconds < 0.5

            res = extract_acoustic_features(path, return_typed=True)
            assert res.overall_status in ("insufficient_data", "failed")
            for bm_name, feat in res.features.items():
                assert feat.value is None
                assert feat.status == FeatureStatus.INSUFFICIENT_INPUT

        finally:
            if os.path.exists(path):
                os.remove(path)

    @pytest.mark.parametrize("sr", [4000, 8000, 11025, 22050, 44100, 48000, 96000, 192000])
    def test_extreme_sample_rates(self, sr):
        """Pipeline must handle extreme sample rates (telephony 4kHz-8kHz to studio 192kHz) without uncaught crashes."""
        path = generate_adversarial_wav(duration_s=2.0, sample_rate=sr, freq_hz=220.0, amplitude=0.6)
        try:
            val_res = validate_audio_file(path, min_duration=0.5)
            assert val_res.is_valid
            assert val_res.sample_rate == sr

            res = extract_acoustic_features(
                path,
                transcript_data={"text": "the picture shows a family in the kitchen", "words": ["the", "picture"]},
                return_typed=True
            )
            assert res is not None
            assert res.overall_status in ("complete", "partial", "failed")

            for bm_name, feat in res.features.items():
                if feat.status == FeatureStatus.VALID:
                    assert feat.value is not None
                    assert not math.isnan(feat.value)
                    assert not math.isinf(feat.value)
                else:
                    assert feat.value is None

        finally:
            if os.path.exists(path):
                os.remove(path)


# ═════════════════════════════════════════════════════════════════════════════
# 2. ADVERSARIAL LINGUISTIC STRESS TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAdversarialLinguistics:
    """Stress-test NLP biomarkers against pathological, empty, repetitive, and unicode text."""

    @pytest.mark.parametrize("empty_input", [
        "",
        "   ",
        "\t\t\n\r\n",
        {"text": ""},
        {"text": "   \n\t  "},
        None,
    ])
    def test_empty_and_whitespace_transcripts(self, empty_input):
        """Empty or whitespace-only text must return INSUFFICIENT_INPUT with value=None for all 4 NLP features."""
        res = analyze_text(empty_input, return_typed=True)
        assert res.overall_status in ("insufficient_data", "failed")
        assert len(res.features) == 4

        for name in CANONICAL_NLP_BIOMARKERS:
            assert name in res.features
            feat = res.features[name]
            assert feat.value is None, f"{name} produced {feat.value} on empty transcript!"
            assert feat.status == FeatureStatus.INSUFFICIENT_INPUT

    @pytest.mark.parametrize("short_text", [
        "Hello",
        "Yes.",
        "CogniSafe test",
        "A quick brown fox",
    ])
    def test_single_word_and_short_transcripts(self, short_text):
        """Short transcripts (<20 tokens, <2 sentences) must gracefully report insufficient input."""
        res = analyze_text(short_text, return_typed=True)
        assert res is not None

        # MTLD requires >=20 tokens
        mtld = res.features["lexical_diversity"]
        assert mtld.value is None
        assert mtld.status == FeatureStatus.INSUFFICIENT_INPUT

        # Semantic coherence requires >=2 sentences
        coherence = res.features["semantic_coherence"]
        assert coherence.value is None
        assert coherence.status == FeatureStatus.INSUFFICIENT_INPUT

        # Idea density requires >=10 words
        density = res.features["idea_density"]
        if len(short_text.split()) < 10:
            assert density.value is None
            assert density.status == FeatureStatus.INSUFFICIENT_INPUT

        # Syntactic complexity must not crash and never output 0.0 without validity
        syn = res.features["syntactic_complexity"]
        if syn.status != FeatureStatus.VALID:
            assert syn.value is None

    def test_repetitive_loops_mtld_zero_division_guard(self):
        """Repetitive word loops ('word '*500) must not cause division-by-zero or infinite loop in MTLD."""
        repetitive_text = "word " * 500
        res = analyze_text(repetitive_text, return_typed=True)

        mtld = res.features["lexical_diversity"]
        assert mtld.status == FeatureStatus.VALID
        assert mtld.value is not None
        assert not math.isnan(mtld.value)
        assert not math.isinf(mtld.value)
        assert mtld.value > 0.0

    def test_repetitive_sentences_semantic_coherence(self):
        """Identical repeated sentences must not cause zero-norm or NaN in cosine similarity."""
        repetitive_sentences = "The mother is washing dishes at the sink. " * 30
        res = analyze_text(repetitive_sentences, return_typed=True)

        coherence = res.features["semantic_coherence"]
        assert coherence.status == FeatureStatus.VALID
        assert coherence.value is not None
        assert not math.isnan(coherence.value)
        assert 0.95 <= coherence.value <= 1.0

    def test_deep_nested_sentences_syntactic_complexity_depth_guard(self):
        """Extremely deep or syntactically chained text must be clamped by max_depth_guard (<=20)."""
        clauses = ["the cat"] + [f"that chased the dog_{i}" for i in range(50)]
        deep_sentence = " ".join(clauses) + " slept peacefully."

        res = analyze_text(deep_sentence, return_typed=True)
        syn = res.features["syntactic_complexity"]
        assert syn.status == FeatureStatus.VALID
        assert syn.value is not None
        assert syn.value <= 20.0, f"Syntactic complexity {syn.value} exceeded max_depth_guard 20!"

    def test_extreme_emojis_and_control_characters(self):
        """Text containing control characters, high-codepoint emojis, and BOMs must not crash."""
        pathological_text = (
            "\x01\x02\x08\x0b\x0c\x0e\x1f"
            "🧠 The patient observed the kitchen scene with deep concentration. 🩺 "
            "🔥 A boy is reaching for cookies while the stool is tipping over. 💥 "
            "\u200b\u200c\u200d\ufeff Water is overflowing from the sink onto the kitchen floor. 🌊"
        )
        res = analyze_text(pathological_text, return_typed=True)
        assert res.overall_status in ("complete", "partial")

        for bm_name, feat in res.features.items():
            if feat.status == FeatureStatus.VALID:
                assert feat.value is not None
                assert not math.isnan(feat.value)
            else:
                assert feat.value is None

    def test_windows_cp1252_encoding_trap_resilience(self):
        """
        Verify that characters outside Windows cp1252 (smart quotes, em-dashes, Cyrillic, Asian scripts,
        mathematical symbols) process cleanly without raising UnicodeEncodeError / UnicodeDecodeError.
        """
        cp1252_traps = (
            "“The patient remarked: ‘Everything is overflowing!’—” "
            "She noticed Café Résumé naïve façade. "
            "Doctor’s note: €100 copay; ±0.5 deviation; 漢字 tested; Москва checked. "
            "The water continues to overflow onto the linoleum tiles."
        )
        res = analyze_text(cp1252_traps, return_typed=True)
        assert res is not None

        dump = res.to_dict()
        assert isinstance(dump, dict)

        anomaly_res = run_anomaly_pipeline(
            user_id="user_cp1252_test",
            current_biomarkers=res,
        )
        assert anomaly_res is not None


# ═════════════════════════════════════════════════════════════════════════════
# 3. ADVERSARIAL BASELINE & MATHEMATICAL ANOMALY TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestAdversarialBaselinesAndAnomaly:
    """Stress-test anomaly detection against degenerate baselines, zero-variance, and missing data."""

    def test_zero_variance_identical_sessions(self, temp_test_db):
        """
        Adversarial baseline: 5 identical historical sessions where std == 0.0 for all biomarkers.
        Verifies:
        - std == 0.0 does NOT raise ZeroDivisionError.
        - Identical 6th session produces 0 anomalies and risk_tier == 'Green'.
        - Deviant 6th session detects anomaly via variance floor without crash.
        """
        identical_biomarkers = {
            "speech_rate": 130.0,
            "articulation_rate": 150.0,
            "pause_frequency": 10.0,
            "pause_duration_mean": 0.5,
            "filled_pause_rate": 1.0,
            "pitch_mean": 180.0,
            "pitch_range": 60.0,
            "jitter": 0.02,
            "shimmer": 0.04,
            "HNR": 18.0,
            "lexical_diversity": 70.0,
            "semantic_coherence": 0.85,
            "idea_density": 0.50,
            "syntactic_complexity": 3.5,
        }

        # Seed 5 identical sessions
        for i in range(5):
            save_session(
                user_id="user_zero_var",
                biomarkers=identical_biomarkers,
                risk_tier="Green",
                db_path=temp_test_db,
                timestamp=(datetime.now(timezone.utc) - timedelta(days=5 - i)).isoformat()
            )

        # 1. Evaluate 6th session: EXACT SAME VALUES
        res_identical = run_anomaly_pipeline(
            user_id="user_zero_var",
            current_biomarkers=identical_biomarkers,
            db_path=temp_test_db,
            save_to_db=False,
        )
        assert res_identical["risk_tier"] == "Green"
        assert len(res_identical["anomaly_flags"]) == 0
        assert res_identical["analysis_status"] == "complete"

        # Check reference bands: std is 0.0, band lower == mean == upper
        bands = res_identical["confidence_intervals"]
        for bm in CANONICAL_BIOMARKERS:
            b_info = bands[bm]
            assert b_info["std"] == 0.0
            assert b_info["lower_95"] == b_info["mean"] == b_info["upper_95"]

        # 2. Evaluate 6th session: DEVIANT VALUE (speech_rate dropped from 130 to 80)
        deviant_biomarkers = dict(identical_biomarkers)
        deviant_biomarkers["speech_rate"] = 80.0
        deviant_biomarkers["pitch_mean"] = 240.0

        res_deviant = run_anomaly_pipeline(
            user_id="user_zero_var",
            current_biomarkers=deviant_biomarkers,
            db_path=temp_test_db,
            save_to_db=False,
        )
        flags = res_deviant["anomaly_flags"]
        flagged_biomarkers = [f["biomarker"] for f in flags]
        assert "speech_rate" in flagged_biomarkers
        assert "pitch_mean" in flagged_biomarkers
        assert res_deviant["risk_tier"] in ("Yellow", "Orange", "Red")

    def test_sparse_and_missing_features_across_history(self, temp_test_db):
        """
        Adversarial baseline: historical sessions with intermittent None values.
        Missing features must NEVER be treated as 0.0 (preventing false 11σ deviations).
        """
        for i in range(6):
            b_map = {}
            if i % 2 == 0:
                b_map["speech_rate"] = 120.0 + i
                b_map["pitch_mean"] = 180.0 + i
                b_map["lexical_diversity"] = None
                b_map["semantic_coherence"] = None
            else:
                b_map["speech_rate"] = None
                b_map["pitch_mean"] = None
                b_map["lexical_diversity"] = 65.0 + i
                b_map["semantic_coherence"] = 0.80

            save_session(
                user_id="user_sparse",
                biomarkers=b_map,
                risk_tier="Green",
                db_path=temp_test_db,
                timestamp=(datetime.now(timezone.utc) - timedelta(days=6 - i)).isoformat()
            )

        current = {
            "speech_rate": None,
            "pitch_mean": 182.0,
            "lexical_diversity": 68.0,
            "semantic_coherence": 0.82,
        }

        past_sessions = load_sessions("user_sparse", db_path=temp_test_db)
        flags = detect_anomalies_zscore(
            user_id="user_sparse",
            current_biomarkers=current,
            past_sessions=past_sessions,
        )
        flagged_names = [f["biomarker"] for f in flags]
        assert "speech_rate" not in flagged_names, "Missing feature coerced to 0.0 and flagged as anomaly!"

    def test_nan_and_inf_injection_sanitization(self):
        """NaN and Inf injected into biomarker inputs must be sanitized to None."""
        raw_inputs = {
            "speech_rate": float("nan"),
            "pitch_mean": float("inf"),
            "jitter": -float("inf"),
            "shimmer": 0.045,
            "lexical_diversity": "NaN",
            "semantic_coherence": "infinity",
        }

        assert extract_numeric_value(raw_inputs["speech_rate"]) is None
        assert extract_numeric_value(raw_inputs["pitch_mean"]) is None
        assert extract_numeric_value(raw_inputs["jitter"]) is None
        assert extract_numeric_value(raw_inputs["shimmer"]) == 0.045
        assert extract_numeric_value(raw_inputs["lexical_diversity"]) is None
        assert extract_numeric_value(raw_inputs["semantic_coherence"]) is None

        res = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=float("nan"),
            unit="Hz",
            method="test"
        )
        assert res.value is None
        assert res.status == FeatureStatus.INVALID_MEASUREMENT

    def test_strict_no_default_to_green_on_cold_start_or_empty(self, temp_test_db):
        """
        Verify the non-negotiable rule: Pipeline NEVER defaults to 'Green' on cold starts,
        preliminary baselines, or missing/insufficient data.
        """
        valid_biomarkers = {bm: 100.0 for bm in CANONICAL_BIOMARKERS}

        # Case 1: Cold start (0 sessions in history)
        res_cold = run_anomaly_pipeline(
            user_id="user_cold_start",
            current_biomarkers=valid_biomarkers,
            db_path=temp_test_db,
            save_to_db=False,
        )
        assert res_cold["risk_tier"] is None, f"Cold start defaulted to {res_cold['risk_tier']} instead of None!"
        assert res_cold["analysis_status"] == "insufficient_baseline"

        # Case 2: Preliminary baseline (1 to 4 sessions in history)
        for count in range(1, 5):
            save_session(
                user_id=f"user_prelim_{count}",
                biomarkers=valid_biomarkers,
                risk_tier=None,
                db_path=temp_test_db,
            )
            res_prelim = run_anomaly_pipeline(
                user_id=f"user_prelim_{count}",
                current_biomarkers=valid_biomarkers,
                db_path=temp_test_db,
                save_to_db=False,
            )
            assert res_prelim["risk_tier"] is None, f"Session count {count} defaulted to {res_prelim['risk_tier']}!"
            assert res_prelim["analysis_status"] == "insufficient_baseline"

        # Case 3: Mature baseline (5 sessions), but current session is completely unvoiced / silent
        for i in range(5):
            save_session(
                user_id="user_mature_silent_current",
                biomarkers=valid_biomarkers,
                risk_tier="Green",
                db_path=temp_test_db,
            )

        silent_current = {bm: None for bm in CANONICAL_BIOMARKERS}
        res_silent = run_anomaly_pipeline(
            user_id="user_mature_silent_current",
            current_biomarkers=silent_current,
            db_path=temp_test_db,
            save_to_db=False,
        )
        assert res_silent["risk_tier"] is None, f"Silent session on mature baseline defaulted to {res_silent['risk_tier']}!"
        assert res_silent["analysis_status"] == "insufficient_data"

    def test_synthetic_xgboost_strictly_decommissioned_from_clinical_risk(self):
        """
        Verify that synthetic XGBoost is strictly decommissioned from clinical risk scoring:
        - predict_xgb(demo_mode=False) returns status 'decommissioned_from_clinical' and tier None.
        - combine_tiers(demo_mode=False) ignores xgb_tier completely.
        """
        mock_biomarkers = {bm: 50.0 for bm in CANONICAL_BIOMARKERS}
        res_clinical = predict_xgb(mock_biomarkers, demo_mode=False)
        assert res_clinical["tier"] is None
        assert res_clinical["status"] == "decommissioned_from_clinical"

        # Even if someone attempts to pass xgb_tier="Red" with 1.0 confidence:
        tier_mature = combine_tiers(
            xgb_tier="Red",
            zscore_tier="Green",
            session_count=7,
            xgb_confidence=1.0,
            demo_mode=False,
        )
        assert tier_mature == "Green", "XGBoost 'Red' overrode personal baseline 'Green'!"

        # When baseline is preliminary (<5 sessions), returns None even if XGBoost says Green
        tier_prelim = combine_tiers(
            xgb_tier="Green",
            zscore_tier=None,
            session_count=3,
            xgb_confidence=0.99,
            demo_mode=False,
        )
        assert tier_prelim is None, "XGBoost forced Green on preliminary baseline!"
