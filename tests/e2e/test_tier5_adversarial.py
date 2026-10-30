"""
tests/e2e/test_tier5_adversarial.py
===================================
Tier 5: Adversarial Hardening and Boundary Stress Suite for CogniSafe.

Covers the 5 Critical Adversarial Stress Vectors:
1. Cross-User Access & IDOR:
   - Querying User A's jobs, sessions, and caregiver shares with User B's JWT token (403 Forbidden).
   - Parameter tampering / user_id spoofing prevention in uploads and persistence.
   - Cross-user history, weekly reports, and trajectory data leakage prevention.
2. Upload Boundary Violations:
   - File size exceeding 25MB (413 Payload Too Large).
   - Non-audio files renamed .wav (PHP, Python, HTML, ELF, text, zero-byte) (400 Bad Request).
   - Audio files exceeding 10-minute duration ceiling (400 Bad Request).
   - Malformed/truncated WAV headers (400 Bad Request).
3. Caregiver Share Token Lifecycle & Security:
   - Expired share token returns 410 Gone (public view and AI summary).
   - Revoked share token returns 404 or 403 Forbidden.
   - Tampered / non-existent tokens return 404 Not Found.
   - Credential leakage audit: public view returns NO passwords, hashes, emails, or user IDs.
4. Nebius AI Anti-Hallucination & Ground-Truth Re-binding:
   - Fabricated biomarker names (e.g. cortisol_level, vocal_entropy) are rejected/discarded.
   - Mutated/deviated numerical values from LLM are strictly bound back to verified ground truth.
   - Canonical units enforced against hallucinated units.
   - Ground-truth risk tier protected against LLM override.
5. Offline Degradation & Fault Tolerance:
   - Clean fallback to deterministic clinical templates when NEBIUS_API_KEY is unset/invalid.
   - Graceful degradation on upstream timeouts, 502/503 errors, and corrupted/malformed JSON.
   - Verification that is_fallback=True is set and no 500 error is propagated.
"""

import io
import json
import wave
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import (
    CANONICAL_BIOMARKERS,
    create_synthetic_wav,
    generate_jwt_token,
    DBUser,
    DBSessionRecord,
    DBCaregiverShare,
    DBJob,
)
from schemas.ai import (
    BiomarkerExplanationItem,
    AnomalyExplanationResponse,
    CANONICAL_BIOMARKER_UNITS,
    normalize_biomarker_name,
)
from services.nebius_client import NebiusClient


@pytest.fixture(autouse=True)
def setup_users(test_db_session):
    """Ensure User 101 (Alice) and User 202 (Bob) exist in DB for authentication dependencies."""
    u1 = test_db_session.query(DBUser).filter(DBUser.id == 101).first()
    if not u1:
        u1 = DBUser(
            id=101,
            name="Alice Patient",
            email="alice@cognisafe.test",
            password_hash="$2b$12$e80y..samplehash",
            dob="1960-01-01"
        )
        test_db_session.add(u1)

    u2 = test_db_session.query(DBUser).filter(DBUser.id == 202).first()
    if not u2:
        u2 = DBUser(
            id=202,
            name="Bob Imposter",
            email="bob@cognisafe.test",
            password_hash="$2b$12$e80y..samplehash",
            dob="1970-01-01"
        )
        test_db_session.add(u2)

    test_db_session.commit()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CROSS-USER ACCESS & IDOR ATTACK VECTORS
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossUserAccessAndIDOR:
    """
    Adversarial IDOR attack suite: User B (attacker) attempts to access,
    modify, or leak User A's (victim) protected medical records, jobs, and shares.
    """

    def test_user_b_cannot_view_user_a_session_by_id(self, client, test_db_session, auth_headers_user_a, auth_headers_user_b):
        """Cross-user session access must return 403 Forbidden."""
        # Setup: User A has a session
        sess_a = DBSessionRecord(
            user_id=101,
            risk_tier="Green",
            speech_rate=125.0,
            pause_frequency=3.0,
            pause_duration_mean=0.45,
            semantic_coherence=0.85,
            recorded_at=datetime.now(timezone.utc),
        )
        test_db_session.add(sess_a)
        test_db_session.commit()
        test_db_session.refresh(sess_a)
        sess_id = sess_a.id

        # Attack: User B attempts to view User A's session
        res = client.get(f"/api/sessions/{sess_id}", headers=auth_headers_user_b)
        assert res.status_code == 403, f"Expected 403 Forbidden on IDOR attempt, got {res.status_code}: {res.text}"
        assert "forbidden" in res.json()["detail"].lower()

        # Legitimate access: User A accesses their own session -> 200 OK
        res_legit = client.get(f"/api/sessions/{sess_id}", headers=auth_headers_user_a)
        assert res_legit.status_code == 200
        assert res_legit.json()["id"] == sess_id

    def test_user_b_cannot_view_user_a_job_status(self, client, test_db_session, auth_headers_user_a, auth_headers_user_b):
        """Cross-user job status query must return 403 Forbidden."""
        # Setup: Job owned by User A
        job_id = f"job_user_a_{secrets.token_hex(4)}"
        job = DBJob(
            id=job_id,
            user_id=101,
            status="done",
            stage="done",
            result={"risk_tier": "Green", "speech_rate": 120.0},
        )
        test_db_session.add(job)
        test_db_session.commit()

        # Attack: User B attempts to poll User A's job
        res = client.get(f"/api/ml/status/{job_id}", headers=auth_headers_user_b)
        assert res.status_code == 403, f"Expected 403 Forbidden on cross-user job access, got {res.status_code}"
        assert "forbidden" in res.json()["detail"].lower()

        # Legitimate access: User A polls their own job -> 200 OK
        res_legit = client.get(f"/api/ml/status/{job_id}", headers=auth_headers_user_a)
        assert res_legit.status_code == 200
        assert res_legit.json()["job_id"] == job_id

    def test_user_b_cannot_revoke_user_a_caregiver_share(self, client, test_db_session, auth_headers_user_a, auth_headers_user_b):
        """Cross-user caregiver share revocation must return 403 Forbidden."""
        # Setup: Share owned by User A
        share = DBCaregiverShare(
            patient_id=101,
            share_token=secrets.token_urlsafe(32),
            recipient_email="nurse@care.org",
            label="Home Care Nurse",
            created_at=datetime.now(timezone.utc),
            revoked=False,
        )
        test_db_session.add(share)
        test_db_session.commit()
        test_db_session.refresh(share)
        share_id = share.id

        # Attack: User B attempts to revoke User A's share
        res = client.delete(f"/api/caregiver/shares/{share_id}", headers=auth_headers_user_b)
        assert res.status_code == 403, f"Expected 403 Forbidden on unauthorized share revocation, got {res.status_code}"

        # Share must still be active (unrevoked) in database
        refreshed = test_db_session.query(DBCaregiverShare).filter(DBCaregiverShare.id == share_id).first()
        assert refreshed.revoked is False, "Share was revoked by unauthorized attacker!"

        # Legitimate revocation by User A -> 200 OK
        res_legit = client.delete(f"/api/caregiver/shares/{share_id}", headers=auth_headers_user_a)
        assert res_legit.status_code == 200
        refreshed_legit = test_db_session.query(DBCaregiverShare).filter(DBCaregiverShare.id == share_id).first()
        assert refreshed_legit.revoked is True

    def test_user_b_cannot_trigger_ai_explain_on_user_a_session(self, client, test_db_session, auth_headers_user_b):
        """User B cannot trigger AI analysis/explanation on User A's session."""
        sess_a = DBSessionRecord(
            user_id=101,
            risk_tier="Orange",
            speech_rate=88.0,
            pause_frequency=7.5,
            semantic_coherence=0.62,
        )
        test_db_session.add(sess_a)
        test_db_session.commit()
        test_db_session.refresh(sess_a)

        # Attack: User B sends AI explain request referencing User A's session
        res = client.post("/api/ai/explain", json={"session_id": sess_a.id}, headers=auth_headers_user_b)
        assert res.status_code in (403, 404), f"Expected 403 or 404, got {res.status_code}: {res.text}"

    def test_user_b_cannot_spoof_user_id_in_session_creation(self, client, test_db_session, auth_headers_user_b):
        """When User B posts a new session, spoofing user_id=101 in JSON payload must be ignored."""
        res = client.post(
            "/api/sessions",
            headers=auth_headers_user_b,
            json={
                "user_id": 101,  # Attacker attempts to inject session into User A's history
                "mode": "voice",
                "speech_rate": 115.0,
                "risk_tier": "Green",
                "analysis_status": "complete",
            }
        )
        assert res.status_code == 200
        created = res.json()
        assert "id" in created

        # Verify in database that session is strictly owned by authenticated user (202), not spoofed 101
        db_record = test_db_session.query(DBSessionRecord).filter(DBSessionRecord.id == created["id"]).first()
        assert db_record is not None
        assert db_record.user_id == 202, f"Expected DB record user_id=202, got {db_record.user_id}"

    def test_user_history_and_reports_multi_tenant_isolation(self, client, test_db_session, auth_headers_user_a, auth_headers_user_b):
        """Ensure sessions, history, and weekly reports are strictly partition-isolated between users."""
        now = datetime.now(timezone.utc)
        # Seed 3 sessions for User A
        for i in range(3):
            test_db_session.add(DBSessionRecord(
                user_id=101,
                speech_rate=110.0 + i,
                semantic_coherence=0.80,
                risk_tier="Green",
                recorded_at=now - timedelta(days=i),
            ))
        # Seed 2 sessions for User B
        for i in range(2):
            test_db_session.add(DBSessionRecord(
                user_id=202,
                speech_rate=80.0 + i,
                semantic_coherence=0.55,
                risk_tier="Red",
                recorded_at=now - timedelta(days=i),
            ))
        test_db_session.commit()

        # Query history for User B: must only see 2 sessions
        res_b = client.get("/api/sessions/history?months=1", headers=auth_headers_user_b)
        assert res_b.status_code == 200
        history_b = res_b.json()
        assert len(history_b) == 2
        for h in history_b:
            assert h["risk_tier"] == "Red"

        # Query weekly report for User B: must reflect Red tier and 2 sessions
        rep_b = client.get("/api/reports/weekly", headers=auth_headers_user_b)
        assert rep_b.status_code == 200
        rep_data_b = rep_b.json()
        assert rep_data_b["sessions_this_week"] == 2
        assert rep_data_b["risk_tier"] == "Red"


# ─────────────────────────────────────────────────────────────────────────────
# 2. UPLOAD BOUNDARY VIOLATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadBoundaryViolations:
    """
    Adversarial testing of file upload limits, MIME/header checks, and duration limits.
    """

    def test_upload_exceeding_25mb_returns_413(self, client, auth_headers_user_a):
        """Uploads strictly larger than 25MB must be rejected with 413 Payload Too Large."""
        oversized = b"RIFF" + b"WAVE" + (b"\x00" * (25 * 1024 * 1024 + 1024))
        res = client.post(
            "/api/ml/analyze",
            headers=auth_headers_user_a,
            files={"audio": ("huge.wav", oversized, "audio/wav")},
        )
        assert res.status_code == 413, f"Expected 413 Payload Too Large, got {res.status_code}: {res.text}"
        assert "25mb" in res.json()["detail"].lower()

    def test_upload_non_audio_php_renamed_wav_returns_400(self, client, auth_headers_user_a):
        """Non-audio script files renamed to .wav must return 400 Bad Request."""
        malicious_content = b"<?php system($_GET['cmd']); ?>"
        res = client.post(
            "/api/ml/analyze",
            headers=auth_headers_user_a,
            files={"audio": ("exploit.wav", malicious_content, "audio/wav")},
        )
        assert res.status_code == 400, f"Expected 400 Bad Request, got {res.status_code}"
        assert "invalid audio file format" in res.json()["detail"].lower()

    def test_upload_text_file_renamed_wav_returns_400(self, client, auth_headers_user_a):
        """Plain text payload renamed to .wav must return 400 Bad Request."""
        text_content = b"This is just a plain English text file pretending to be audio."
        res = client.post(
            "/api/ml/analyze",
            headers=auth_headers_user_a,
            files={"audio": ("speech.wav", text_content, "audio/wav")},
        )
        assert res.status_code == 400
        assert "invalid audio file format" in res.json()["detail"].lower()

    def test_upload_elf_binary_renamed_wav_returns_400(self, client, auth_headers_user_a):
        """ELF executable binary payload renamed to .wav must return 400 Bad Request."""
        elf_bytes = b"\x7fELF\x02\x01\x01\x00" + (b"\x00" * 128)
        res = client.post(
            "/api/ml/analyze",
            headers=auth_headers_user_a,
            files={"audio": ("malware.wav", elf_bytes, "audio/wav")},
        )
        assert res.status_code == 400
        assert "invalid audio file format" in res.json()["detail"].lower()

    def test_upload_empty_zero_byte_file_returns_400(self, client, auth_headers_user_a):
        """Zero-byte audio upload must return 400 Bad Request."""
        res = client.post(
            "/api/ml/analyze",
            headers=auth_headers_user_a,
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert res.status_code == 400
        assert "invalid audio file format" in res.json()["detail"].lower()

    def test_upload_truncated_riff_header_returns_400(self, client, auth_headers_user_a):
        """RIFF header truncated before WAVE marker must return 400 Bad Request."""
        truncated_header = b"RIFF\x10\x00\x00\x00XXXX"  # Missing WAVE tag
        res = client.post(
            "/api/ml/analyze",
            headers=auth_headers_user_a,
            files={"audio": ("corrupt.wav", truncated_header, "audio/wav")},
        )
        assert res.status_code == 400
        assert "invalid audio file format" in res.json()["detail"].lower()

    def test_upload_wav_duration_over_600s_returns_400(self, client, auth_headers_user_a):
        """WAV audio with duration exceeding 600 seconds (10 minutes) must return 400 Bad Request."""
        # Synthesize a compact 605-second WAV by using 50 Hz framerate
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(50)
            # 605 seconds * 50 frames/sec = 30250 frames (~60KB)
            wf.writeframes(b"\x00\x00" * int(605 * 50))
        oversized_duration_wav = buf.getvalue()

        res = client.post(
            "/api/ml/analyze",
            headers=auth_headers_user_a,
            files={"audio": ("long_assessment.wav", oversized_duration_wav, "audio/wav")},
        )
        assert res.status_code == 400
        assert "exceeds maximum allowed 10 minutes" in res.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. CAREGIVER SHARE TOKEN LIFECYCLE & CREDENTIAL LEAKAGE
# ─────────────────────────────────────────────────────────────────────────────

class TestCaregiverShareLifecycleAndSecurity:
    """
    Stress-test the caregiver share token lifecycle:
    - Expired tokens -> 410 Gone
    - Revoked tokens -> 404 or 403
    - Tampered/invalid tokens -> 404
    - Strict absence of credentials/hashes/emails in public view
    """

    def test_expired_caregiver_share_returns_410(self, client, test_db_session):
        """Expired caregiver share token must return 410 Gone on public view and AI summary."""
        expired_token = f"expired_token_{secrets.token_urlsafe(16)}"
        share = DBCaregiverShare(
            patient_id=101,
            share_token=expired_token,
            recipient_email="guardian@family.org",
            label="Guardian",
            created_at=datetime.now(timezone.utc) - timedelta(days=15),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),  # Expired 1 hour ago
            revoked=False,
        )
        test_db_session.add(share)
        test_db_session.commit()

        # 1. Public view must return 410 Gone
        res_view = client.get(f"/api/caregiver/view/{expired_token}")
        assert res_view.status_code == 410, f"Expected 410 Gone for expired token, got {res_view.status_code}"
        assert "expired" in res_view.json()["detail"].lower()

        # 2. AI caregiver-summary endpoint must also return 410 Gone
        res_ai = client.post("/api/ai/caregiver-summary", json={"share_token": expired_token})
        assert res_ai.status_code == 410
        assert "expired" in res_ai.json()["detail"].lower()

    def test_revoked_caregiver_share_returns_404_or_403(self, client, test_db_session):
        """Revoked caregiver share must return 404 on public view and 403 on AI summary."""
        revoked_token = f"revoked_token_{secrets.token_urlsafe(16)}"
        share = DBCaregiverShare(
            patient_id=101,
            share_token=revoked_token,
            recipient_email="caregiver@home.org",
            label="Home Caregiver",
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
            revoked=True,  # Explicitly revoked by patient
        )
        test_db_session.add(share)
        test_db_session.commit()

        # 1. Public view returns 404
        res_view = client.get(f"/api/caregiver/view/{revoked_token}")
        assert res_view.status_code == 404
        assert "revoked" in res_view.json()["detail"].lower() or "invalid" in res_view.json()["detail"].lower()

        # 2. AI caregiver summary returns 403
        res_ai = client.post("/api/ai/caregiver-summary", json={"share_token": revoked_token})
        assert res_ai.status_code == 403
        assert "revoked" in res_ai.json()["detail"].lower()

    def test_tampered_or_nonexistent_share_token_returns_404(self, client):
        """Tampered or random caregiver share tokens must return 404 Not Found."""
        bogus_token = "tampered_token_that_does_not_exist_in_database_12345"
        res_view = client.get(f"/api/caregiver/view/{bogus_token}")
        assert res_view.status_code == 404

        res_ai = client.post("/api/ai/caregiver-summary", json={"share_token": bogus_token})
        assert res_ai.status_code == 404

    def test_caregiver_public_view_sanitization_no_credentials_leaked(self, client, test_db_session):
        """Public caregiver endpoint must sanitize output and leak zero credentials/hashes/PII."""
        token = secrets.token_urlsafe(32)
        patient = DBUser(
            id=101,
            name="Eleanor Rigby",
            email="eleanor@sensitive-domain.com",
            password_hash="$argon2id$v=19$m=65536,t=3,p=4$super_sensitive_salt_and_hash_xyz",
            dob="1955-06-18",
        )
        test_db_session.merge(patient)
        share = DBCaregiverShare(
            patient_id=101,
            share_token=token,
            recipient_email="family@domain.org",
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            revoked=False,
        )
        test_db_session.add(share)
        test_db_session.add(DBSessionRecord(
            user_id=101,
            risk_tier="Green",
            speech_rate=118.0,
            semantic_coherence=0.82,
            pause_frequency=3.1,
            recorded_at=datetime.now(timezone.utc),
        ))
        test_db_session.commit()

        res = client.get(f"/api/caregiver/view/{token}")
        assert res.status_code == 200
        view_data = res.json()

        raw_json_str = json.dumps(view_data).lower()
        # Ensure sensitive keywords and fields are completely absent
        assert "password" not in raw_json_str
        assert "password_hash" not in view_data
        assert "eleanor@sensitive-domain.com" not in raw_json_str
        assert "$argon2id$" not in raw_json_str
        assert "token" not in view_data
        assert "patient_id" not in view_data
        assert "user_id" not in view_data


# ─────────────────────────────────────────────────────────────────────────────
# 4. NEBIUS AI ANTI-HALLUCINATION & GROUND-TRUTH RE-BINDING
# ─────────────────────────────────────────────────────────────────────────────

class TestNebiusAntiHallucinationDefense:
    """
    Stress-test schemas/ai.py and services/nebius_client.py against:
    - Fabricated biomarker names (cortisol, vocal_entropy, etc.)
    - Mutated numerical biomarker values
    - Mutated units
    - Risk tier overwrite attempts
    """

    def test_fabricated_biomarker_keys_rejected_by_pydantic_validator(self):
        """BiomarkerExplanationItem must reject any key not in the 14 canonical list."""
        fabricated_keys = [
            "cortisol_level",
            "vocal_entropy",
            "dementia_index",
            "blood_pressure",
            "emotional_entropy",  # Removed legacy column
            "theta_wave_power",
            "cerebral_flow",
        ]
        for key in fabricated_keys:
            with pytest.raises(ValueError, match="not one of the 14 canonical"):
                BiomarkerExplanationItem(
                    biomarker_id=key,
                    observed_value=12.5,
                    unit="arb",
                    explanation="Fabricated clinical metric",
                )

    def test_anti_hallucination_discards_unmeasured_biomarkers(self):
        """If ground truth has only 2 biomarkers, any additional biomarker in LLM output is discarded."""
        llm_response = AnomalyExplanationResponse(
            summary="Multi-feature analysis",
            explanations=[
                BiomarkerExplanationItem(
                    biomarker_id="speech_rate",
                    observed_value=112.0,
                    unit="words/min",
                    explanation="Normal speech pacing.",
                ),
                BiomarkerExplanationItem(
                    biomarker_id="HNR",
                    observed_value=22.0,
                    unit="dB",
                    explanation="Hallucinated HNR explanation.",
                ),
                BiomarkerExplanationItem(
                    biomarker_id="syntactic_complexity",
                    observed_value=4.0,
                    unit="mean tree depth",
                    explanation="Hallucinated syntax depth.",
                ),
            ]
        )

        # Ground truth only measured speech_rate
        ground_truth = {"speech_rate": 112.0}
        sanitized = llm_response.filter_and_bind_ground_truth(ground_truth=ground_truth)

        # HNR and syntactic_complexity must be discarded
        assert len(sanitized.explanations) == 1
        assert sanitized.explanations[0].biomarker_id == "speech_rate"
        assert sanitized.explanations[0].observed_value == 112.0

    def test_anti_hallucination_rebinds_mutated_numerical_values(self):
        """When LLM hallucinates an altered number (e.g. 999.0 vs 114.0), ground truth is forcefully rebound."""
        llm_response = AnomalyExplanationResponse(
            summary="Deviation test",
            explanations=[
                BiomarkerExplanationItem(
                    biomarker_id="speech_rate",
                    observed_value=999.99,  # Hallucinated value
                    unit="words/min",
                    explanation="Speech rate altered by LLM.",
                ),
                BiomarkerExplanationItem(
                    biomarker_id="pause_frequency",
                    observed_value=-15.0,  # Physically impossible hallucinated value
                    unit="pauses/min",
                    explanation="Pause frequency altered.",
                ),
            ]
        )

        ground_truth = {
            "speech_rate": 114.0,
            "pause_frequency": 3.8,
        }
        sanitized = llm_response.filter_and_bind_ground_truth(ground_truth=ground_truth)

        assert len(sanitized.explanations) == 2
        exp_dict = {e.biomarker_id: e.observed_value for e in sanitized.explanations}
        # Verified: ground truth values 114.0 and 3.8 are restored
        assert exp_dict["speech_rate"] == 114.0
        assert exp_dict["pause_frequency"] == 3.8

    def test_anti_hallucination_enforces_canonical_units(self):
        """LLM-supplied units are overridden with canonical units."""
        llm_response = AnomalyExplanationResponse(
            summary="Unit test",
            explanations=[
                BiomarkerExplanationItem(
                    biomarker_id="jitter",
                    observed_value=0.012,
                    unit="seconds/frame",  # Incorrect unit from LLM
                    explanation="Jitter test.",
                ),
                BiomarkerExplanationItem(
                    biomarker_id="pitch_mean",
                    observed_value=180.0,
                    unit="RPM",  # Absurd unit from LLM
                    explanation="Pitch test.",
                ),
            ]
        )
        ground_truth = {"jitter": 0.012, "pitch_mean": 180.0}
        sanitized = llm_response.filter_and_bind_ground_truth(ground_truth=ground_truth)

        assert sanitized.explanations[0].unit == "%"
        assert sanitized.explanations[1].unit == "Hz"

    def test_anti_hallucination_enforces_ground_truth_risk_tier(self):
        """LLM cannot escalate or de-escalate verified session risk tier."""
        llm_response = AnomalyExplanationResponse(
            risk_tier="Red",  # LLM attempted to force Red tier
            summary="Risk tier tampering test",
            explanations=[],
        )
        # Ground truth verified tier is Yellow
        sanitized = llm_response.filter_and_bind_ground_truth(
            ground_truth={},
            enforce_risk_tier="Yellow",
        )
        assert sanitized.risk_tier == "Yellow", f"Expected Yellow, got {sanitized.risk_tier}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. OFFLINE DEGRADATION & FAULT TOLERANCE
# ─────────────────────────────────────────────────────────────────────────────

class TestOfflineGracefulDegradation:
    """
    Verify that all AI endpoints degrade gracefully to deterministic templates
    without throwing 500 errors when Nebius Token Factory is offline or unconfigured.
    """

    def test_explain_endpoint_falls_back_when_api_key_unset(self, client, auth_headers_user_a):
        """POST /api/ai/explain returns 200 with is_fallback=True when Nebius is unconfigured."""
        unconfigured_client = NebiusClient(api_key="")
        with patch("routes.ai.get_nebius_client", return_value=unconfigured_client):
            payload = {
                "biomarkers": {"speech_rate": 105.0, "pause_frequency": 4.2},
                "anomalies": ["speech_rate"],
                "risk_tier": "Yellow",
            }
            res = client.post("/api/ai/explain", json=payload, headers=auth_headers_user_a)
            assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
            data = res.json()
            assert data["is_fallback"] is True
            assert len(data["explanations"]) >= 1
            assert data["explanations"][0]["biomarker_id"] == "speech_rate"
            assert data["explanations"][0]["observed_value"] == 105.0

    def test_weekly_summary_falls_back_when_api_key_unset(self, client, auth_headers_user_a):
        """POST /api/ai/weekly-summary returns 200 with is_fallback=True when Nebius is unconfigured."""
        unconfigured_client = NebiusClient(api_key="")
        with patch("routes.ai.get_nebius_client", return_value=unconfigured_client):
            payload = {
                "sessions_data": [
                    {"speech_rate": 120.0, "risk_tier": "Green"},
                    {"speech_rate": 115.0, "risk_tier": "Green"},
                ]
            }
            res = client.post("/api/ai/weekly-summary", json=payload, headers=auth_headers_user_a)
            assert res.status_code == 200
            data = res.json()
            assert data["is_fallback"] is True
            assert len(data["key_observations"]) > 0
            assert "trend_analysis" in data

    def test_caregiver_summary_falls_back_when_api_key_unset(self, client, auth_headers_user_a):
        """POST /api/ai/caregiver-summary returns 200 with is_fallback=True when Nebius is unconfigured."""
        unconfigured_client = NebiusClient(api_key="")
        with patch("routes.ai.get_nebius_client", return_value=unconfigured_client):
            payload = {
                "biomarkers": {"speech_rate": 95.0, "pause_frequency": 6.0},
                "risk_tier": "Yellow",
                "patient_name": "Arthur",
            }
            res = client.post("/api/ai/caregiver-summary", json=payload, headers=auth_headers_user_a)
            assert res.status_code == 200
            data = res.json()
            assert data["is_fallback"] is True
            assert "Arthur" in data["plain_language_summary"]
            assert len(data["supportive_tips"]) >= 3

    def test_chat_endpoint_falls_back_when_api_key_unset(self, client, auth_headers_user_a):
        """POST /api/chat returns 200 with is_fallback=True when Nebius is unconfigured."""
        unconfigured_client = NebiusClient(api_key="")
        with patch("routes.chat.get_nebius_client", return_value=unconfigured_client):
            payload = {
                "messages": [{"role": "user", "content": "How do you calculate risk tiers?"}]
            }
            res = client.post("/api/chat", json=payload, headers=auth_headers_user_a)
            assert res.status_code == 200
            data = res.json()
            assert data["is_fallback"] is True
            assert "personal historical baseline" in data["reply"]

    @pytest.mark.asyncio
    async def test_nebius_client_timeout_triggers_deterministic_fallback(self):
        """When upstream HTTP request times out, NebiusClient falls back gracefully."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.post.side_effect = httpx.TimeoutException("Upstream connection timeout")

        client_inst = NebiusClient(api_key="valid-key", http_client=mock_http)
        res = await client_inst.explain_biomarkers(
            biomarkers={"speech_rate": 110.0},
            risk_tier="Green",
        )
        assert res.is_fallback is True
        assert res.explanations[0].observed_value == 110.0

    @pytest.mark.asyncio
    async def test_nebius_client_upstream_502_triggers_deterministic_fallback(self):
        """When Nebius returns 502 Bad Gateway, fallback returns without raising."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        req = httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")
        mock_http.post.return_value = httpx.Response(502, request=req, text="Bad Gateway")

        client_inst = NebiusClient(api_key="valid-key", http_client=mock_http)
        res = await client_inst.generate_weekly_summary(
            sessions_data=[{"speech_rate": 115.0, "risk_tier": "Green"}]
        )
        assert res.is_fallback is True
        assert len(res.key_observations) > 0

    @pytest.mark.asyncio
    async def test_nebius_client_corrupted_json_triggers_deterministic_fallback(self):
        """When Nebius returns unparseable text, fallback returns without raising."""
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        req = httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")
        mock_res_json = {
            "choices": [{"message": {"content": "<html><body>504 Gateway Timeout</body></html>"}}]
        }
        mock_http.post.return_value = httpx.Response(200, json=mock_res_json, request=req)

        client_inst = NebiusClient(api_key="valid-key", http_client=mock_http)
        res = await client_inst.explain_biomarkers(
            biomarkers={"speech_rate": 105.0},
            risk_tier="Green",
        )
        assert res.is_fallback is True
        assert res.explanations[0].observed_value == 105.0
