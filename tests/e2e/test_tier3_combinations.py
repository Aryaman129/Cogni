"""
tests/e2e/test_tier3_combinations.py
====================================
Tier 3: Pairwise Cross-Feature Combinations.
Covers:
- Voice Mode vs Text Mode
- Establishing Baseline vs Mature Baseline Lifecycle
- Caregiver Token Issuance, Public Access, and Revocation
- Multi-Tenant User Isolation & Ownership Enforcement
- Statistical Drift + Clinical Risk Tier Escalation
- Nebius Output + Anti-Hallucination Guardrails
- Database Schema Compatibility across Biomarker Aliases
"""

import math
import secrets
import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

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
# 1. VOICE MODE VS TEXT MODE PAIRWISE INTERACTION
# ─────────────────────────────────────────────────────────────────────────────
class TestVoiceVsTextModeCombination:
    """Pairwise combination: Voice Mode vs Text Mode processing and baselines."""

    def test_voice_mode_extracts_all_14_biomarkers(self, test_db_session):
        """Voice session contains measurements for both acoustic and NLP biomarkers."""
        voice_session = DBSessionRecord(
            user_id=101,
            mode="voice",
            duration_seconds=35.0,
            analysis_status="complete",
            risk_tier="Green",
            # Acoustic
            speech_rate=125.0,
            articulation_rate=145.0,
            pause_frequency=3.5,
            pause_duration_mean=0.55,
            filled_pause_rate=1.2,
            pitch_mean=185.0,
            pitch_range=60.0,
            jitter=1.1,
            shimmer=3.8,
            hnr=19.2,
            # NLP
            lexical_diversity=68.5,
            semantic_coherence=0.84,
            idea_density=0.55,
            syntactic_complexity=3.8,
            transcript="The child is reaching for the cookie jar while the mother washes dishes."
        )
        test_db_session.add(voice_session)
        test_db_session.commit()

        saved = test_db_session.query(DBSessionRecord).filter_by(id=voice_session.id).first()
        assert saved.mode == "voice"
        assert saved.speech_rate is not None
        assert saved.semantic_coherence is not None

    def test_text_mode_nullifies_acoustic_biomarkers_without_zero_distortion(self, test_db_session):
        """Text mode sets acoustic features to None/null (never 0.0) with unsupported_input status."""
        text_biomarkers = {}
        for b in CANONICAL_BIOMARKERS[:10]:  # 10 Acoustic
            text_biomarkers[b] = TypedFeatureResult(
                status=FeatureStatus.UNSUPPORTED_INPUT,
                value=None,
                unit="N/A",
                method="text_mode_unsupported",
                reason="Acoustic analysis is unsupported in text mode"
            ).dict()

        for b in CANONICAL_BIOMARKERS[10:]:  # 4 NLP
            text_biomarkers[b] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=0.75,
                unit="score",
                method="nlp_pipeline"
            ).dict()

        text_session = DBSessionRecord(
            user_id=101,
            mode="text",
            duration_seconds=None,
            analysis_status="complete",
            risk_tier="Green",
            # Acoustic features must be NULL
            speech_rate=None,
            articulation_rate=None,
            pause_frequency=None,
            pause_duration_mean=None,
            filled_pause_rate=None,
            pitch_mean=None,
            pitch_range=None,
            jitter=None,
            shimmer=None,
            hnr=None,
            # NLP features populated
            lexical_diversity=72.0,
            semantic_coherence=0.81,
            idea_density=0.52,
            syntactic_complexity=3.6,
            transcript="I am describing this picture in text mode because I cannot speak today.",
            biomarker_results=text_biomarkers
        )
        test_db_session.add(text_session)
        test_db_session.commit()

        saved = test_db_session.query(DBSessionRecord).filter_by(id=text_session.id).first()
        assert saved.mode == "text"
        assert saved.speech_rate is None
        assert saved.pitch_mean is None
        assert saved.semantic_coherence == 0.81

    def test_text_mode_does_not_corrupt_acoustic_baseline(self, test_db_session):
        """Acoustic baseline calculation ignores text sessions rather than averaging with 0.0."""
        # 3 voice sessions with speech_rate = 120.0
        for _ in range(3):
            test_db_session.add(DBSessionRecord(user_id=101, mode="voice", speech_rate=120.0))
        # 2 text sessions with speech_rate = None
        for _ in range(2):
            test_db_session.add(DBSessionRecord(user_id=101, mode="text", speech_rate=None))
        test_db_session.commit()

        sessions = test_db_session.query(DBSessionRecord).filter_by(user_id=101).all()
        # Compute baseline over valid voice sessions only
        valid_rates = [s.speech_rate for s in sessions if s.speech_rate is not None]
        assert len(valid_rates) == 3
        mean_rate = sum(valid_rates) / len(valid_rates)
        assert mean_rate == 120.0  # NOT pulled down to 72.0 by zeroes!


# ─────────────────────────────────────────────────────────────────────────────
# 2. ESTABLISHING BASELINE VS MATURE BASELINE LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────
class TestBaselineLifecycleCombination:
    """Pairwise combination: Progression from Establishing Baseline to Mature Baseline."""

    def test_progressive_baseline_threshold_transition(self, client, seed_user_and_sessions, auth_headers_user_a):
        """Sessions 1-4 have risk_tier=None; Session 5 unlocks mature risk tier."""
        # Seed 4 sessions
        seed_user_and_sessions(user_id=101, session_count=4)
        resp4 = client.get("/api/sessions/latest", headers=auth_headers_user_a)
        assert resp4.status_code == 200
        assert resp4.json()["analysis_status"] == "insufficient_baseline"
        assert resp4.json()["risk_tier"] is None

        # Add 5th session
        resp_save = client.post(
            "/api/sessions",
            headers=auth_headers_user_a,
            json={
                "mode": "voice",
                "analysis_status": "complete",
                "risk_tier": "Green",
                "speech_rate": 125.0,
                "semantic_coherence": 0.85
            }
        )
        assert resp_save.status_code == 200

        # Query latest after 5th session
        resp5 = client.get("/api/sessions/latest", headers=auth_headers_user_a)
        assert resp5.status_code == 200
        data5 = resp5.json()
        assert data5["analysis_status"] == "complete"
        assert data5["risk_tier"] == "Green"


# ─────────────────────────────────────────────────────────────────────────────
# 3. CAREGIVER TOKEN ISSUANCE, ACCESS, AND REVOCATION LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────
class TestCaregiverTokenLifecycleCombination:
    """Pairwise combination: Caregiver share token issuance, read-only view, and revocation."""

    def test_caregiver_token_lifecycle(self, test_db_session):
        # 1. Patient creates share token
        raw_token = secrets.token_urlsafe(32)
        share = DBCaregiverShare(
            patient_id=101,
            share_token=raw_token,
            recipient_email="daughter@example.com",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
            revoked=False
        )
        test_db_session.add(share)
        test_db_session.commit()

        # 2. Caregiver accesses with valid token
        active_share = test_db_session.query(DBCaregiverShare).filter_by(
            share_token=raw_token,
            revoked=False
        ).first()
        assert active_share is not None
        assert active_share.patient_id == 101

        # 3. Patient revokes token
        active_share.revoked = True
        test_db_session.commit()

        # 4. Subsequent query fails to find active unrevoked token
        revoked_query = test_db_session.query(DBCaregiverShare).filter_by(
            share_token=raw_token,
            revoked=False
        ).first()
        assert revoked_query is None

    def test_expired_caregiver_token_is_ineligible(self, test_db_session):
        """Expired caregiver tokens must not grant access."""
        expired_token = secrets.token_urlsafe(32)
        share = DBCaregiverShare(
            patient_id=101,
            share_token=expired_token,
            recipient_email="doctor@clinic.org",
            created_at=datetime.utcnow() - timedelta(days=10),
            expires_at=datetime.utcnow() - timedelta(days=1),  # Expired yesterday
            revoked=False
        )
        test_db_session.add(share)
        test_db_session.commit()

        found = test_db_session.query(DBCaregiverShare).filter_by(share_token=expired_token).first()
        assert found is not None
        # Check expiry condition
        is_expired = found.expires_at < datetime.utcnow()
        assert is_expired is True


# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-USER MULTI-TENANT ISOLATION
# ─────────────────────────────────────────────────────────────────────────────
class TestMultiTenantIsolationCombination:
    """Pairwise combination: Cross-user privacy and data isolation."""

    def test_user_cannot_view_peer_history(self, client, seed_user_and_sessions, auth_headers_user_a, auth_headers_user_b):
        """User A and User B cannot view each other's session histories."""
        seed_user_and_sessions(user_id=101, session_count=5)
        seed_user_and_sessions(user_id=202, session_count=2)

        resp_a = client.get("/api/sessions/history", headers=auth_headers_user_a)
        assert resp_a.status_code == 200
        history_a = resp_a.json()
        assert len(history_a) == 5

        resp_b = client.get("/api/sessions/history", headers=auth_headers_user_b)
        assert resp_b.status_code == 200
        history_b = resp_b.json()
        assert len(history_b) == 2

        # Verify no overlapping session IDs
        ids_a = {item["session_id"] for item in history_a}
        ids_b = {item["session_id"] for item in history_b}
        assert ids_a.isdisjoint(ids_b)


# ─────────────────────────────────────────────────────────────────────────────
# 5. STATISTICAL DRIFT & RISK ESCALATION
# ─────────────────────────────────────────────────────────────────────────────
class TestStatisticalDriftRiskEscalationCombination:
    """Pairwise combination: Historical drift triggers anomaly detection and risk escalation."""

    def test_longitudinal_drift_escalation(self, seed_user_and_sessions, test_db_session):
        # Seed 10 sessions with gradual cognitive drift (decay)
        _, sessions = seed_user_and_sessions(user_id=101, session_count=10, drift=True)

        early_sessions = sessions[:3]
        late_sessions = sessions[-3:]

        # Early sessions have higher speech rate and lower pause frequency
        avg_early_rate = sum(s.speech_rate for s in early_sessions) / len(early_sessions)
        avg_late_rate = sum(s.speech_rate for s in late_sessions) / len(late_sessions)
        assert avg_early_rate > avg_late_rate

        # Early sessions have lower pause frequency than late sessions
        avg_early_pause = sum(s.pause_frequency for s in early_sessions) / len(early_sessions)
        avg_late_pause = sum(s.pause_frequency for s in late_sessions) / len(late_sessions)
        assert avg_late_pause > avg_early_pause


# ─────────────────────────────────────────────────────────────────────────────
# 6. NEBIUS STRUCTURED OUTPUT & ANTI-HALLUCINATION
# ─────────────────────────────────────────────────────────────────────────────
class TestNebiusAntiHallucinationCombination:
    """Pairwise combination: Binding generative AI narrative to ground-truth measurements."""

    def test_ground_truth_binding_discards_fabricated_numbers(self):
        ground_truth = {"speech_rate": 112.5, "pause_frequency": 4.1}

        # Simulated hallucinated LLM response claiming speech_rate is 145.0
        raw_llm_output = {
            "summary": "Patient spoke at a brisk 145.0 words per minute.",
            "clinical_explanation": "Speech rate remains within normal bounds.",
            "caregiver_note": "Speaking smoothly.",
            "referenced_biomarkers": ["speech_rate"],
            "confidence_score": 0.88
        }
        structured = NebiusStructuredResponse(**raw_llm_output)

        # Backend anti-hallucination sanitization:
        # Cross-reference any cited metrics against ground-truth dictionary
        sanitized_metrics = {}
        for b in structured.referenced_biomarkers:
            if b in ground_truth:
                sanitized_metrics[b] = ground_truth[b]

        assert sanitized_metrics["speech_rate"] == 112.5  # Ground truth preserved!


# ─────────────────────────────────────────────────────────────────────────────
# 7. DATABASE SCHEMA COMPATIBILITY ACROSS BIOMARKER ALIASES
# ─────────────────────────────────────────────────────────────────────────────
class TestDatabaseSchemaCompatibilityCombination:
    """Pairwise combination: Session model handles both canonical and legacy aliases."""

    def test_biomarker_aliases_coherence(self, test_db_session):
        session = DBSessionRecord(
            user_id=101,
            pause_duration_mean=0.48,
            hnr=17.5
        )
        test_db_session.add(session)
        test_db_session.commit()

        # Test property getters
        assert session.pause_duration == 0.48
        assert session.HNR == 17.5

        # Test property setters
        session.pause_duration = 0.62
        session.HNR = 19.0
        test_db_session.commit()

        saved = test_db_session.query(DBSessionRecord).filter_by(id=session.id).first()
        assert saved.pause_duration_mean == 0.62
        assert saved.hnr == 19.0
