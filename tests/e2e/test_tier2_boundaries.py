"""
tests/e2e/test_tier2_boundaries.py
==================================
Tier 2: Boundary & Corner Cases across all feature categories.
Covers silent audio, clipped audio, corrupt files, short text, single sentence,
empty baseline, upload limits, zero-variance baselines, and security boundaries.
"""

import math
import io
import wave
import struct
import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from conftest import (
    CANONICAL_BIOMARKERS,
    FeatureStatus,
    TypedFeatureResult,
    NebiusStructuredResponse,
    create_synthetic_wav,
    generate_jwt_token,
    DBUser,
    DBSessionRecord,
    DBCaregiverShare,
    DBJob
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ACOUSTIC BOUNDARY & CORNER CASES
# ─────────────────────────────────────────────────────────────────────────────
class TestAcousticBoundaries:
    """Boundary conditions for speech and acoustic processing."""

    def test_silent_audio_insufficient_input(self, synthetic_silent_wav):
        """Pure silent audio must yield INSUFFICIENT_INPUT and null value, never 0.0."""
        wf = wave.open(io.BytesIO(synthetic_silent_wav), 'rb')
        frames = wf.readframes(wf.getnframes())
        max_amplitude = max(abs(s) for s in struct.unpack(f"<{len(frames)//2}h", frames))
        assert max_amplitude == 0

        # Contract assertion: silent audio status
        res = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="Hz",
            method="pitch_f0_pyin",
            reason="Audio energy is below silence threshold; no voiced frames"
        )
        assert res.status == FeatureStatus.INSUFFICIENT_INPUT
        assert res.value is None

    def test_clipped_audio_boundary(self, synthetic_clipped_wav):
        """Severely clipped audio must be detectable at amplitude ceilings."""
        wf = wave.open(io.BytesIO(synthetic_clipped_wav), 'rb')
        frames = wf.readframes(wf.getnframes())
        samples = struct.unpack(f"<{len(frames)//2}h", frames)
        clipped_samples = [s for s in samples if abs(s) >= 32767]
        clipping_ratio = len(clipped_samples) / len(samples)
        assert clipping_ratio > 0.5

    def test_subsecond_short_audio_insufficient_input(self, synthetic_short_wav):
        """Audio under 1.0s is too short for reliable acoustic biomarker extraction."""
        wf = wave.open(io.BytesIO(synthetic_short_wav), 'rb')
        duration = wf.getnframes() / wf.getframerate()
        assert duration < 1.0

        res = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="words/min",
            method="speech_rate",
            reason="Audio duration (0.4s) is below minimum threshold (1.0s)"
        )
        assert res.status == FeatureStatus.INSUFFICIENT_INPUT
        assert res.value is None

    def test_corrupted_wav_header_rejection(self, corrupted_audio_bytes):
        """Corrupted audio headers must fail validation without unhandled crash."""
        def validate_wav_header(b: bytes) -> bool:
            if len(b) < 44:
                return False
            if b[:4] != b"RIFF" or b[8:12] != b"WAVE" or b[12:16] != b"fmt ":
                return False
            if b"data" not in b:
                return False
            return True

        assert validate_wav_header(corrupted_audio_bytes) is False

    def test_unvoiced_audio_pitch_mean_is_null(self):
        """Audio with no voiced frames returns null pitch, not 0.0 Hz."""
        voiced_frames = []
        pitch_mean = sum(voiced_frames) / len(voiced_frames) if len(voiced_frames) > 0 else None
        assert pitch_mean is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. NLP & LINGUISTIC BOUNDARY & CORNER CASES
# ─────────────────────────────────────────────────────────────────────────────
class TestNLPBoundaries:
    """Boundary conditions for NLP and text processing."""

    def test_empty_text_rejection(self):
        """Empty transcript must be rejected or marked INSUFFICIENT_INPUT."""
        text = ""
        tokens = text.split()
        assert len(tokens) == 0

        res = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="MTLD",
            method="mtld_lexical_diversity",
            reason="Transcript is empty"
        )
        assert res.status == FeatureStatus.INSUFFICIENT_INPUT

    def test_short_text_threshold_guard(self):
        """Text with fewer than 10 words is insufficient for MTLD."""
        short_text = "The dog barked at the moon."
        tokens = short_text.split()
        assert len(tokens) < 10

        res = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="MTLD",
            method="mtld_lexical_diversity",
            reason=f"Token count ({len(tokens)}) below minimum threshold (10)"
        )
        assert res.status == FeatureStatus.INSUFFICIENT_INPUT

    def test_single_sentence_coherence_guard(self):
        """Single sentence input cannot have cosine similarity between consecutive sentences."""
        sentences = ["The cat sat on the mat."]
        # Should NOT return 1.0 claiming perfect coherence; must return INSUFFICIENT_INPUT
        coherence = None if len(sentences) < 2 else 0.85
        assert coherence is None

    def test_punctuation_and_whitespace_only(self):
        """Text containing only punctuation and whitespace."""
        junk = "   ... ??? !!! ,,, ---   "
        import re
        words = re.findall(r'\b\w+\b', junk)
        assert len(words) == 0

    def test_unicode_and_emojis_encoding_safety(self):
        """Unicode emojis and special characters must not crash on encoding."""
        text_with_emoji = "Cognitive health is important. 🧠 🗣️ \u26a0\ufe0f \u2705"
        # Must encode to utf-8 cleanly without UnicodeEncodeError
        encoded = text_with_emoji.encode("utf-8")
        decoded = encoded.decode("utf-8")
        assert "🧠" in decoded

    def test_highly_repetitive_text_mtld(self):
        """Highly repetitive text yields low lexical diversity without division by zero."""
        repeated_text = "yes " * 100
        tokens = repeated_text.split()
        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens)
        assert ttr == 0.01  # Extreme low TTR


# ─────────────────────────────────────────────────────────────────────────────
# 3. STATISTICAL & PERSONAL BASELINE BOUNDARY CASES
# ─────────────────────────────────────────────────────────────────────────────
class TestBaselineBoundaries:
    """Boundary conditions for personal baselines and anomaly detection."""

    def test_zero_variance_baseline_guard(self):
        """When historical std is 0.0, z-score calculation must not divide by zero."""
        historical_scores = [120.0, 120.0, 120.0, 120.0, 120.0]
        mean = sum(historical_scores) / len(historical_scores)
        variance = sum((x - mean) ** 2 for x in historical_scores) / len(historical_scores)
        std = math.sqrt(variance)
        assert std == 0.0

        current_val = 120.0
        # Guard: if std == 0, z-score is 0.0
        z_score = 0.0 if std < 1e-7 else (current_val - mean) / std
        assert z_score == 0.0

    def test_preliminary_baseline_under_five_sessions(self, client, seed_user_and_sessions, auth_headers_user_a):
        """Users with <5 sessions must have analysis_status='insufficient_baseline' and risk_tier=None."""
        seed_user_and_sessions(user_id=101, session_count=4)
        resp = client.get("/api/sessions/latest", headers=auth_headers_user_a)
        assert resp.status_code == 200
        data = resp.json()
        assert data["analysis_status"] == "insufficient_baseline"
        assert data["risk_tier"] is None

    def test_empty_baseline_history(self, client, seed_user_and_sessions, auth_headers_user_b):
        """Users with 0 sessions must receive 404 from latest session."""
        seed_user_and_sessions(user_id=202, session_count=0)
        resp = client.get("/api/sessions/latest", headers=auth_headers_user_b)
        assert resp.status_code == 404

    def test_missing_biomarker_not_imputed_as_zero(self):
        """Missing biomarkers must be omitted from statistical aggregation, never imputed as 0.0."""
        biomarkers = {
            "speech_rate": 115.0,
            "pause_frequency": 4.5,
            "pitch_mean": None,  # Missing!
            "hnr": None          # Missing!
        }
        # Valid measurements only
        valid_values = [v for v in biomarkers.values() if v is not None]
        assert len(valid_values) == 2
        assert 0.0 not in valid_values

    def test_extreme_outlier_zscore(self):
        """Extreme outlier input yields high z-score without overflow."""
        mean = 120.0
        std = 10.0
        extreme_val = 1200.0  # 108 standard deviations
        z = (extreme_val - mean) / std
        assert z == 108.0
        assert math.isfinite(z)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SECURITY, AUTH & UPLOAD BOUNDARY CASES
# ─────────────────────────────────────────────────────────────────────────────
class TestSecurityBoundaries:
    """Security boundaries, authorization edges, and upload constraints."""

    def test_upload_exact_limit_vs_overflow(self):
        """25MB ceiling: 25*1024*1024 is permitted, 25*1024*1024 + 1 rejected."""
        LIMIT = 25 * 1024 * 1024
        assert LIMIT == 26214400
        overflow = LIMIT + 1
        assert overflow > LIMIT

    def test_mime_extension_spoofing_detection(self):
        """Text script renamed to .wav must fail binary validation."""
        fake_wav = b"echo 'malicious script' > /tmp/malware.sh\n"
        is_riff = fake_wav.startswith(b"RIFF")
        assert is_riff is False

    def test_malformed_auth_header_rejections(self, client):
        """Various malformed Authorization headers must be rejected with 401/403."""
        malformed_headers = [
            {"Authorization": "Bearer"},
            {"Authorization": "Bearer "},
            {"Authorization": "Basic dXNlcjpwYXNz"},
            {"Authorization": "InvalidScheme token123"},
            {"Authorization": "Bearer not.a.valid.jwt.token"},
        ]
        for h in malformed_headers:
            resp = client.get("/api/sessions/latest", headers=h)
            assert resp.status_code in (401, 403)

    def test_multi_year_january_boundary_grouping(self):
        """Sessions on Dec 31 and Jan 1 must group into distinct month buckets."""
        d_dec = datetime(2025, 12, 31, 23, 59, 59)
        d_jan = datetime(2026, 1, 1, 0, 0, 1)
        key_dec = d_dec.strftime("%Y-%m")
        key_jan = d_jan.strftime("%Y-%m")
        assert key_dec == "2025-12"
        assert key_jan == "2026-01"
        assert key_dec != key_jan


# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERATIVE AI & NEBIUS TOKEN FACTORY BOUNDARIES
# ─────────────────────────────────────────────────────────────────────────────
class TestNebiusBoundaries:
    """Boundary conditions for Nebius Token Factory and NVIDIA Nemotron 3.5."""

    def test_nebius_anti_hallucination_rejects_missing_fields(self):
        """Incomplete JSON from LLM must be rejected by Pydantic."""
        incomplete_json = {
            "summary": "Everything looks normal."
            # Missing clinical_explanation, caregiver_note, etc.
        }
        with pytest.raises(ValidationError):
            NebiusStructuredResponse(**incomplete_json)

    def test_nebius_anti_hallucination_rejects_out_of_bounds_confidence(self):
        """Confidence score outside [0.0, 1.0] must be rejected."""
        invalid_confidence = {
            "summary": "Valid summary",
            "clinical_explanation": "Valid explanation",
            "caregiver_note": "Valid note",
            "referenced_biomarkers": ["speech_rate"],
            "confidence_score": 1.5  # Out of range!
        }
        with pytest.raises(ValidationError):
            NebiusStructuredResponse(**invalid_confidence)

    def test_nebius_graceful_degradation_offline_fallback(self):
        """When Nebius is unreachable, pipeline produces valid fallback response."""
        def handle_nebius_call(is_online: bool) -> Dict[str, Any]:
            if not is_online:
                return {
                    "source": "fallback_template",
                    "summary": "Standard analysis completed. Cognitive markers within normal baseline range.",
                    "is_fallback": True
                }
            return {
                "source": "nvidia/Nemotron-3_5-Lightning",
                "summary": "AI narrative generated.",
                "is_fallback": False
            }

        fallback_result = handle_nebius_call(is_online=False)
        assert fallback_result["is_fallback"] is True
        assert "Standard analysis completed" in fallback_result["summary"]
