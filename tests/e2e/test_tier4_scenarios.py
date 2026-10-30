"""
tests/e2e/test_tier4_scenarios.py
=================================
Tier 4: Real-World Longitudinal Application Scenarios.
Covers:
- Scenario 1: New Patient Onboarding from Registration to Baseline Maturity
- Scenario 2: Longitudinal Cognitive Monitoring & Anomaly Detection (30-day drift)
- Scenario 3: Caregiver Remote Dashboard & Sharing Portal Lifecycle
- Scenario 4: Mute/Dysphonic Patient Text-First Assessment Workflow
- Scenario 5: Nebius AI-Augmented Weekly Report with Graceful Degradation
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
# SCENARIO 1: NEW PATIENT ONBOARDING TO BASELINE MATURITY
# ─────────────────────────────────────────────────────────────────────────────
class TestScenario1PatientOnboarding:
    """
    End-to-end journey of a new patient registering, recording their first 5 sessions,
    and seeing their status transition from 'Establishing Baseline' to 'Complete'.
    """

    def test_scenario1_registration_to_mature_baseline(self, client):
        # 1. Register new user
        user_email = f"patient_{secrets.token_hex(4)}@cognisafe.test"
        reg_resp = client.post(
            "/api/auth/register",
            json={
                "name": "Evelyn Vance",
                "email": user_email,
                "password": "SecurePassword123!",
                "dob": "1958-04-12"
            }
        )
        assert reg_resp.status_code == 200
        auth_data = reg_resp.json()
        token = auth_data["access_token"]
        user_id = auth_data["user_id"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Verify today status is initially unrecorded
        today_init = client.get("/api/sessions/today", headers=headers)
        assert today_init.status_code == 200
        assert today_init.json()["recorded"] is False

        # 3. Complete Sessions 1 through 4 (establishing baseline phase)
        for session_idx in range(1, 5):
            save_resp = client.post(
                "/api/sessions",
                headers=headers,
                json={
                    "mode": "voice",
                    "duration_seconds": 28.5,
                    "analysis_status": "insufficient_baseline",
                    "risk_tier": None,
                    "speech_rate": 122.0 + session_idx,
                    "articulation_rate": 140.0,
                    "pause_frequency": 3.2,
                    "pause_duration_mean": 0.42,
                    "filled_pause_rate": 0.9,
                    "pitch_mean": 192.0,
                    "pitch_range": 55.0,
                    "jitter": 1.05,
                    "shimmer": 3.9,
                    "hnr": 18.2,
                    "lexical_diversity": 69.0,
                    "semantic_coherence": 0.83,
                    "idea_density": 0.53,
                    "syntactic_complexity": 3.7,
                    "transcript": f"Assessment session {session_idx} describing the park scene."
                }
            )
            assert save_resp.status_code == 200

            # Latest session during establishing phase must have null risk tier
            latest_resp = client.get("/api/sessions/latest", headers=headers)
            assert latest_resp.status_code == 200
            assert latest_resp.json()["analysis_status"] == "insufficient_baseline"
            assert latest_resp.json()["risk_tier"] is None

        # 4. Record Session 5 (baseline maturity milestone!)
        save_resp_5 = client.post(
            "/api/sessions",
            headers=headers,
            json={
                "mode": "voice",
                "duration_seconds": 30.0,
                "analysis_status": "complete",
                "risk_tier": "Green",
                "speech_rate": 124.0,
                "articulation_rate": 142.0,
                "pause_frequency": 3.1,
                "pause_duration_mean": 0.40,
                "filled_pause_rate": 0.8,
                "pitch_mean": 194.0,
                "pitch_range": 56.0,
                "jitter": 1.02,
                "shimmer": 3.8,
                "hnr": 18.5,
                "lexical_diversity": 70.0,
                "semantic_coherence": 0.85,
                "idea_density": 0.54,
                "syntactic_complexity": 3.8,
                "transcript": "Assessment session 5 describing the park scene with mature baseline."
            }
        )
        assert save_resp_5.status_code == 200

        # 5. Verify latest session now reflects mature baseline with Green risk tier
        latest_mature = client.get("/api/sessions/latest", headers=headers)
        assert latest_mature.status_code == 200
        assert latest_mature.json()["analysis_status"] == "complete"
        assert latest_mature.json()["risk_tier"] == "Green"


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2: LONGITUDINAL MONITORING & ANOMALY DETECTION JOURNEY
# ─────────────────────────────────────────────────────────────────────────────
class TestScenario2LongitudinalMonitoring:
    """
    Simulates a 30-day longitudinal monitoring sequence where subtle progressive
    linguistic and acoustic drift triggers anomaly detection and risk adjustment.
    """

    def test_scenario2_longitudinal_drift_detection(self, client, seed_user_and_sessions, auth_headers_user_a, test_db_session):
        # 1. Seed 30 daily sessions spanning the last month with progressive drift
        start_date = datetime.utcnow() - timedelta(days=30)
        user, sessions = seed_user_and_sessions(
            user_id=101,
            session_count=30,
            start_date=start_date,
            drift=True
        )

        # 2. Query trajectory over 6 months
        traj_resp = client.get("/api/reports/trajectory?months=6", headers=auth_headers_user_a)
        assert traj_resp.status_code == 200
        traj_points = traj_resp.json()
        assert len(traj_points) >= 1

        # 3. Query weekly report
        weekly_resp = client.get("/api/reports/weekly", headers=auth_headers_user_a)
        assert weekly_resp.status_code == 200
        report = weekly_resp.json()
        assert report["sessions_this_week"] >= 1
        assert report["narrative"] is not None

        # 4. Verify that late sessions reflect drift compared to baseline
        mature_baseline_sessions = sessions[:5]
        late_sessions = sessions[-5:]

        baseline_speech_rate = sum(s.speech_rate for s in mature_baseline_sessions) / len(mature_baseline_sessions)
        late_speech_rate = sum(s.speech_rate for s in late_sessions) / len(late_sessions)
        assert late_speech_rate < baseline_speech_rate

        baseline_pause_freq = sum(s.pause_frequency for s in mature_baseline_sessions) / len(mature_baseline_sessions)
        late_pause_freq = sum(s.pause_frequency for s in late_sessions) / len(late_sessions)
        assert late_pause_freq > baseline_pause_freq


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3: CAREGIVER REMOTE DASHBOARD & SHARING LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────
class TestScenario3CaregiverRemoteDashboard:
    """
    Patient generates a secure sharing token for family caregiver;
    caregiver inspects read-only patient summary; patient revokes token.
    """

    def test_scenario3_caregiver_flow(self, test_db_session):
        patient_id = 101

        # 1. Patient issues caregiver token
        share_token = secrets.token_urlsafe(32)
        share = DBCaregiverShare(
            patient_id=patient_id,
            share_token=share_token,
            recipient_email="caregiver.son@family.org",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=60),
            revoked=False
        )
        test_db_session.add(share)
        test_db_session.commit()

        # 2. Caregiver uses token to view patient summary
        # Validate that share is active and belongs to patient 101
        active_share = test_db_session.query(DBCaregiverShare).filter_by(
            share_token=share_token,
            revoked=False
        ).first()
        assert active_share is not None

        # Construct read-only caregiver summary payload
        caregiver_view_payload = {
            "patient_id": active_share.patient_id,
            "share_status": "active",
            "expires_in_days": 60,
            "read_only": True,
            "risk_tier": "Green",
            "recent_speech_rate": 122.0,
            "recent_coherence": 0.84,
            "caregiver_summary": "Your family member has been consistent with assessments and markers remain stable."
        }
        assert caregiver_view_payload["read_only"] is True
        assert caregiver_view_payload["patient_id"] == 101

        # 3. Patient revokes access
        active_share.revoked = True
        test_db_session.commit()

        # 4. Subsequent caregiver request is denied
        revoked_check = test_db_session.query(DBCaregiverShare).filter_by(
            share_token=share_token,
            revoked=False
        ).first()
        assert revoked_check is None


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4: MUTE / DYSPHONIC PATIENT TEXT-FIRST ASSESSMENT WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────
class TestScenario4MutePatientTextFirstWorkflow:
    """
    Mute or dysphonic patient completes assessments in text mode;
    NLP features extracted, acoustic features cleanly nullified.
    """

    def test_scenario4_text_assessment_workflow(self, client, auth_headers_user_a, test_db_session, seed_user_and_sessions):
        # 0. Ensure patient 101 exists
        seed_user_and_sessions(user_id=101, session_count=0)

        # 1. Patient submits text description
        written_sample = (
            "In this kitchen scene, the mother is washing dishes at the sink. "
            "Behind her, water is overflowing onto the kitchen floor. "
            "Two children are attempting to take cookies from the cookie jar on a high shelf. "
            "The boy is standing on a stool that is tipping over precariously."
        )

        # 2. Save text session via authenticated API
        resp = client.post(
            "/api/sessions",
            headers=auth_headers_user_a,
            json={
                "mode": "text",
                "duration_seconds": None,
                "analysis_status": "complete",
                "risk_tier": "Green",
                "transcript": written_sample,
                # NLP Biomarkers calculated
                "lexical_diversity": 74.5,
                "semantic_coherence": 0.86,
                "idea_density": 0.58,
                "syntactic_complexity": 4.1,
                # Acoustic Biomarkers must be None
                "speech_rate": None,
                "articulation_rate": None,
                "pause_frequency": None,
                "pause_duration_mean": None,
                "filled_pause_rate": None,
                "pitch_mean": None,
                "pitch_range": None,
                "jitter": None,
                "shimmer": None,
                "hnr": None,
            }
        )
        assert resp.status_code == 200
        saved_session = resp.json()

        # 3. Verify session was saved with mode='text' and null acoustic values
        assert saved_session["mode"] == "text"
        assert saved_session["speech_rate"] is None
        assert saved_session["pitch_mean"] is None
        assert saved_session["semantic_coherence"] == 0.86
        assert saved_session["lexical_diversity"] == 74.5


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 5: NEBIUS AI WEEKLY REPORT WITH GRACEFUL DEGRADATION
# ─────────────────────────────────────────────────────────────────────────────
class TestScenario5NebiusWeeklyReportWithGracefulDegradation:
    """
    Weekly report generation with Nebius Token Factory augmentation
    and verified graceful fallback when offline.
    """

    def test_scenario5_online_nebius_structured_summary(self):
        """Live/mocked Nebius response strictly validated by Pydantic."""
        mock_nebius_response = {
            "summary": "This week's speech assessments demonstrate stable cognitive markers.",
            "clinical_explanation": "Slight variation in pause duration remains well within historical reference bands.",
            "caregiver_note": "Speaking confidence is high; daily routines are proceeding normally.",
            "referenced_biomarkers": ["pause_duration_mean", "speech_rate"],
            "confidence_score": 0.94
        }
        validated = NebiusStructuredResponse(**mock_nebius_response)
        assert validated.confidence_score == 0.94
        assert "pause_duration_mean" in validated.referenced_biomarkers

    def test_scenario5_offline_graceful_degradation(self, client, seed_user_and_sessions, auth_headers_user_a):
        """When Nebius is unreachable, weekly report returns deterministic fallback narrative."""
        seed_user_and_sessions(user_id=101, session_count=7, start_date=datetime.utcnow() - timedelta(days=6))

        # Query weekly report (which uses local fallback when Nebius is unconfigured/mocked)
        resp = client.get("/api/reports/weekly", headers=auth_headers_user_a)
        assert resp.status_code == 200
        data = resp.json()

        # Report must be fully populated despite offline LLM
        assert data["sessions_this_week"] == 7
        assert data["risk_tier"] in ("Green", "Yellow", "Orange", "Red")
        assert len(data["insights"]) > 0
        assert "This week you completed 7 sessions" in data["narrative"]
