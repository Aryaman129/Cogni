import os
import sys
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure cognisafe-backend is in sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cognisafe-backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import database
from database import Base, get_db
from main import app
from auth import hash_password, create_token
from models import User, Session, CaregiverShare
from services.nebius_client import NebiusClient, get_nebius_client, SYSTEM_GUARDRAILS
from schemas.ai import (
    CANONICAL_BIOMARKERS,
    CANONICAL_BIOMARKER_UNITS,
    normalize_biomarker_name,
    BiomarkerExplanationItem,
    AnomalyExplanationResponse,
    WeeklySummaryResponse,
    CaregiverSummaryResponse,
    ChatResponse,
)

# ------------------------------------------------------------------------------
# Test Database & TestClient Fixtures
# ------------------------------------------------------------------------------
if getattr(database, "SessionLocal", None) and getattr(database.engine, "url", None) and str(database.engine.url) == "sqlite:///:memory:":
    TestingSessionLocal = database.SessionLocal
    test_engine = database.engine
else:
    TEST_DB_URL = "sqlite:///:memory:"
    test_engine = create_engine(
        TEST_DB_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    database.SessionLocal = TestingSessionLocal
    database.engine = test_engine


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture
def clean_db():
    """Clean database state for integration tests."""
    db = TestingSessionLocal()
    try:
        db.query(CaregiverShare).delete()
        db.query(Session).delete()
        db.query(User).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture
def test_user(clean_db):
    db = TestingSessionLocal()
    user = User(
        name="Test Patient",
        email="patient@example.com",
        password_hash=hash_password("password123"),
        dob="1960-01-01",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token({"sub": str(user.id)})
    db.close()
    return {"user_id": user.id, "email": user.email, "token": token}


@pytest.fixture
def other_user(clean_db):
    db = TestingSessionLocal()
    user = User(
        name="Other Patient",
        email="other@example.com",
        password_hash=hash_password("password123"),
        dob="1965-05-05",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token({"sub": str(user.id)})
    db.close()
    return {"user_id": user.id, "email": user.email, "token": token}


# ==============================================================================
# 1. Nebius Client Configuration Tests
# ==============================================================================

def test_default_client_configuration():
    """Verify default Nebius client configuration attributes."""
    with patch.dict(os.environ, {}, clear=True):
        c = NebiusClient(api_key=None, base_url=None, model_id=None, timeout_seconds=None)
        assert c.base_url == "https://api.tokenfactory.nebius.com/v1"
        assert c.model_id == "nvidia/Nemotron-3_5-Lightning"
        assert c.timeout_seconds == 30.0
        assert c.is_configured() is False


def test_custom_client_configuration_via_args():
    """Verify explicit configuration parameter overrides."""
    c = NebiusClient(
        api_key="neb-test-token-12345",
        base_url="https://custom.nebius.com/v1/",
        model_id="nvidia/Nemotron-3_5-Custom",
        timeout_seconds=45.0,
    )
    assert c.api_key == "neb-test-token-12345"
    assert c.base_url == "https://custom.nebius.com/v1"  # Stripped trailing slash
    assert c.model_id == "nvidia/Nemotron-3_5-Custom"
    assert c.timeout_seconds == 45.0
    assert c.is_configured() is True


def test_custom_client_configuration_via_env():
    """Verify configuration reading from environment variables."""
    env_vars = {
        "NEBIUS_API_KEY": "neb-env-key-999",
        "NEBIUS_BASE_URL": "https://env.nebius.ai/v1",
        "NEBIUS_MODEL_ID": "nvidia/Nemotron-3_5-Lightning",
        "NEBIUS_TIMEOUT_SECONDS": "20.5",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        c = NebiusClient()
        assert c.api_key == "neb-env-key-999"
        assert c.base_url == "https://env.nebius.ai/v1"
        assert c.model_id == "nvidia/Nemotron-3_5-Lightning"
        assert c.timeout_seconds == 20.5
        assert c.is_configured() is True


def test_is_configured_predicate(monkeypatch):
    """Verify is_configured handles empty, whitespace, and valid keys correctly."""
    assert NebiusClient(api_key="").is_configured() is False
    assert NebiusClient(api_key="   ").is_configured() is False
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert NebiusClient(api_key=None).is_configured() is False
    assert NebiusClient(api_key="valid-key").is_configured() is True


# ==============================================================================
# 2. Prompt Construction & System Guardrail Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_prompt_construction_and_ground_truth_binding():
    """
    Verify that prompt sent to Nebius contains exact ground-truth values and
    anti-hallucination system guardrails.
    """
    mock_response = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "explanations": [{
                        "biomarker_id": "speech_rate",
                        "observed_value": 115.0,
                        "unit": "words/min",
                        "status": "normal",
                        "explanation": "Speech rate is steady.",
                        "clinical_context": "Reflects normal motor speech."
                    }],
                    "summary": "Stable voice patterns.",
                    "recommendations": ["Continue regular tracking."]
                })
            }
        }]
    }

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    req = httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")
    mock_res = httpx.Response(200, json=mock_response, request=req)
    mock_http_client.post.return_value = mock_res

    client_instance = NebiusClient(
        api_key="test-key",
        http_client=mock_http_client,
    )

    ground_truth = {
        "speech_rate": 115.0,
        "pause_frequency": 3.2,
        "pitch_mean": 180.5,
    }

    res = await client_instance.explain_biomarkers(
        biomarkers=ground_truth,
        anomalies=["pause_frequency"],
        risk_tier="Green",
    )

    assert res.is_fallback is False
    # Verify mock HTTP client call inspection
    assert mock_http_client.post.called
    call_args, call_kwargs = mock_http_client.post.call_args
    assert call_args[0] == "https://api.tokenfactory.nebius.com/v1/chat/completions"

    sent_payload = call_kwargs["json"]
    assert sent_payload["model"] == "nvidia/Nemotron-3_5-Lightning"

    # Verify system guardrail
    system_msg = next(m["content"] for m in sent_payload["messages"] if m["role"] == "system")
    assert "CRITICAL CLINICAL & ANTI-HALLUCINATION RULES" in system_msg
    assert "NOT a diagnostic medical device" in system_msg
    assert "You must NEVER calculate, invent, extrapolate, or alter any biomarker numerical values" in system_msg

    # Verify user message contains ground truth values
    user_msg = next(m["content"] for m in sent_payload["messages"] if m["role"] == "user")
    assert "115.0" in user_msg
    assert "words/min" in user_msg
    assert "pause_frequency" in user_msg
    assert "Green" in user_msg


# ==============================================================================
# 3. Pydantic Response Parsing & Anti-Hallucination Rejection Tests
# ==============================================================================

def test_anti_hallucination_discards_fabricated_biomarkers():
    """
    Verify that any biomarker not in the canonical 14 is strictly rejected,
    and ungrounded metrics are removed.
    """
    raw_response = AnomalyExplanationResponse(
        summary="Test analysis",
        explanations=[
            BiomarkerExplanationItem(
                biomarker_id="speech_rate",
                observed_value=120.0,
                unit="words/min",
                explanation="Pacing is normal.",
            ),
            BiomarkerExplanationItem(
                biomarker_id="syntactic_complexity",
                observed_value=3.5,
                unit="mean tree depth",
                explanation="Grammar is complex.",
            ),
        ],
    )

    # If ground-truth only has speech_rate, syntactic_complexity must be discarded
    ground_truth = {"speech_rate": 120.0}
    sanitized = raw_response.filter_and_bind_ground_truth(ground_truth=ground_truth)

    assert len(sanitized.explanations) == 1
    assert sanitized.explanations[0].biomarker_id == "speech_rate"
    assert sanitized.explanations[0].observed_value == 120.0


def test_anti_hallucination_rebinds_deviated_biomarker_values():
    """
    If the LLM returns an altered or hallucinated numerical value (e.g. 999.0 vs 110.0),
    the validator re-binds the value strictly to verified ground-truth.
    """
    raw_response = AnomalyExplanationResponse(
        summary="Test analysis",
        explanations=[
            BiomarkerExplanationItem(
                biomarker_id="speech_rate",
                observed_value=999.0,  # Hallucinated value!
                unit="words/min",
                explanation="Speech rate altered by LLM.",
            )
        ],
    )

    ground_truth = {"speech_rate": 110.0}
    sanitized = raw_response.filter_and_bind_ground_truth(ground_truth=ground_truth)

    assert len(sanitized.explanations) == 1
    # Verified: value is forcefully restored to ground-truth 110.0
    assert sanitized.explanations[0].observed_value == 110.0


def test_anti_hallucination_discards_completely_fabricated_biomarker_keys():
    """Verify that invented metrics (e.g., cortisol_level, vocal_iq) cannot pass validation."""
    with pytest.raises(ValueError, match="not one of the 14 canonical"):
        BiomarkerExplanationItem(
            biomarker_id="cortisol_level",
            observed_value=12.5,
            unit="ug/dL",
            explanation="Fabricated cortisol measurement",
        )

    with pytest.raises(ValueError, match="not one of the 14 canonical"):
        BiomarkerExplanationItem(
            biomarker_id="brainwave_alpha",
            observed_value=8.2,
            unit="Hz",
            explanation="Fabricated EEG metric",
        )


def test_anti_hallucination_enforces_canonical_units():
    """Verify that canonical units are strictly bound to ground truth standards."""
    raw_response = AnomalyExplanationResponse(
        summary="Unit test",
        explanations=[
            BiomarkerExplanationItem(
                biomarker_id="jitter",
                observed_value=0.012,
                unit="arbitrary_units",
                explanation="Jitter test.",
            ),
            BiomarkerExplanationItem(
                biomarker_id="HNR",
                observed_value=18.5,
                unit="watts",
                explanation="HNR test.",
            ),
        ],
    )

    ground_truth = {"jitter": 0.012, "HNR": 18.5}
    sanitized = raw_response.filter_and_bind_ground_truth(ground_truth=ground_truth)

    assert sanitized.explanations[0].unit == "%"
    assert sanitized.explanations[1].unit == "dB"


def test_anti_hallucination_protects_risk_tier():
    """Verify that the verified risk tier cannot be overridden by LLM output."""
    raw_response = AnomalyExplanationResponse(
        risk_tier="Red",  # LLM attempted to change tier
        summary="Risk tier test",
        explanations=[],
    )
    sanitized = raw_response.filter_and_bind_ground_truth(
        ground_truth={},
        enforce_risk_tier="Green",  # Ground-truth verified tier
    )
    assert sanitized.risk_tier == "Green"


# ==============================================================================
# 4. Graceful Degradation Tests (Unconfigured / Offline / Error)
# ==============================================================================

@pytest.mark.asyncio
async def test_graceful_degradation_when_unconfigured():
    """
    When NEBIUS_API_KEY is not configured, all methods must return valid
    deterministic template responses with is_fallback=True, without raising 500.
    """
    client_instance = NebiusClient(api_key="")
    assert client_instance.is_configured() is False

    # 1. explain_biomarkers fallback
    explain_res = await client_instance.explain_biomarkers(
        biomarkers={"speech_rate": 95.0, "pause_frequency": 7.5},
        anomalies=["speech_rate"],
        risk_tier="Yellow",
        session_id=101,
    )
    assert explain_res.is_fallback is True
    assert explain_res.session_id == 101
    assert explain_res.risk_tier == "Yellow"
    assert len(explain_res.explanations) == 1
    assert explain_res.explanations[0].biomarker_id == "speech_rate"
    assert explain_res.explanations[0].observed_value == 95.0
    assert "words/min" in explain_res.explanations[0].unit
    assert len(explain_res.recommendations) > 0

    # 2. generate_weekly_summary fallback
    weekly_res = await client_instance.generate_weekly_summary(
        sessions_data=[
            {"speech_rate": 120.0, "risk_tier": "Green"},
            {"speech_rate": 118.0, "risk_tier": "Green"},
        ]
    )
    assert weekly_res.is_fallback is True
    assert len(weekly_res.key_observations) > 0
    assert "2 voice monitoring session" in weekly_res.key_observations[0]

    # 3. generate_caregiver_summary fallback
    caregiver_res = await client_instance.generate_caregiver_summary(
        biomarkers={"speech_rate": 88.0, "pause_frequency": 8.0},
        risk_tier="Yellow",
        patient_name="Grandpa Joe",
    )
    assert caregiver_res.is_fallback is True
    assert "Grandpa Joe" in caregiver_res.plain_language_summary
    assert len(caregiver_res.supportive_tips) >= 3

    # 4. chat fallback
    chat_res = await client_instance.chat(
        messages=[{"role": "user", "content": "How do you calculate risk tiers?"}]
    )
    assert chat_res.is_fallback is True
    assert "personal historical baseline" in chat_res.reply


@pytest.mark.asyncio
async def test_graceful_degradation_on_timeout():
    """Verify that httpx.TimeoutException triggers seamless fallback."""
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_http_client.post.side_effect = httpx.TimeoutException("Connection timed out after 30s")

    client_instance = NebiusClient(
        api_key="valid-key",
        http_client=mock_http_client,
    )

    res = await client_instance.explain_biomarkers(
        biomarkers={"speech_rate": 110.0},
        risk_tier="Green",
    )
    assert res.is_fallback is True
    assert res.explanations[0].observed_value == 110.0


@pytest.mark.asyncio
async def test_graceful_degradation_on_server_502():
    """Verify that upstream 502/500 errors trigger seamless fallback."""
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    req = httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")
    mock_response = httpx.Response(502, request=req, text="Bad Gateway")
    mock_http_client.post.return_value = mock_response

    client_instance = NebiusClient(
        api_key="valid-key",
        http_client=mock_http_client,
    )

    res = await client_instance.generate_caregiver_summary(
        biomarkers={"speech_rate": 115.0},
        patient_name="Alice",
    )
    assert res.is_fallback is True
    assert "Alice" in res.plain_language_summary


@pytest.mark.asyncio
async def test_graceful_degradation_on_malformed_json_response():
    """Verify that non-JSON response from LLM triggers graceful fallback."""
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    req = httpx.Request("POST", "https://api.tokenfactory.nebius.com/v1/chat/completions")
    mock_res_json = {
        "choices": [{"message": {"content": "This is raw non-json text that cannot be parsed."}}]
    }
    mock_res = httpx.Response(200, json=mock_res_json, request=req)
    mock_http_client.post.return_value = mock_res

    client_instance = NebiusClient(
        api_key="valid-key",
        http_client=mock_http_client,
    )

    res = await client_instance.explain_biomarkers(
        biomarkers={"speech_rate": 105.0},
        risk_tier="Green",
    )
    assert res.is_fallback is True
    assert res.explanations[0].observed_value == 105.0


# ==============================================================================
# 5. FastAPI Endpoints Authentication & Integration Tests
# ==============================================================================

def test_chat_endpoint_requires_jwt_auth():
    """POST /api/chat must reject unauthenticated requests with 401."""
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hello"}]})
    assert res.status_code == 401


def test_chat_endpoint_authenticated_success(test_user):
    """POST /api/chat with valid JWT bearer token returns 200 ChatResponse."""
    headers = {"Authorization": f"Bearer {test_user['token']}"}
    payload = {
        "messages": [{"role": "user", "content": "What is speech rate?"}],
        "system_prompt": "You are a helpful assistant.",
    }
    res = client.post("/api/chat", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert data["model"] == "nvidia/Nemotron-3_5-Lightning"
    assert "speech rate" in data["reply"].lower() or "biomarker" in data["reply"].lower()


def test_ai_explain_endpoint_requires_jwt_auth():
    """POST /api/ai/explain must reject unauthenticated requests with 401."""
    res = client.post("/api/ai/explain", json={"biomarkers": {"speech_rate": 110.0}})
    assert res.status_code == 401


def test_ai_explain_with_direct_biomarkers(test_user):
    """POST /api/ai/explain with valid biomarkers returns 200 AnomalyExplanationResponse."""
    headers = {"Authorization": f"Bearer {test_user['token']}"}
    payload = {
        "biomarkers": {
            "speech_rate": 90.0,
            "pause_frequency": 6.8,
            "HNR": 14.2,
        },
        "anomalies": ["speech_rate", "pause_frequency"],
        "risk_tier": "Yellow",
    }
    res = client.post("/api/ai/explain", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["risk_tier"] == "Yellow"
    assert len(data["explanations"]) == 2
    exps = {e["biomarker_id"]: e for e in data["explanations"]}
    assert "speech_rate" in exps
    assert exps["speech_rate"]["observed_value"] == 90.0
    assert exps["speech_rate"]["unit"] == "words/min"
    assert "pause_frequency" in exps
    assert exps["pause_frequency"]["observed_value"] == 6.8


def test_ai_explain_with_db_session(test_user):
    """
    POST /api/ai/explain loads ground truth from the user's persisted Session record
    and saves generated ai_insights back to the database.
    """
    db = TestingSessionLocal()
    session_record = Session(
        user_id=test_user["user_id"],
        risk_tier="Green",
        speech_rate=125.0,
        pause_frequency=3.0,
        pause_duration_mean=0.6,
        pitch_mean=185.0,
        hnr=22.0,
        semantic_coherence=0.88,
        mode="voice",
        duration_seconds=45.0,
    )
    db.add(session_record)
    db.commit()
    db.refresh(session_record)
    s_id = session_record.id
    db.close()

    headers = {"Authorization": f"Bearer {test_user['token']}"}
    payload = {"session_id": s_id}
    res = client.post("/api/ai/explain", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == s_id
    assert data["risk_tier"] == "Green"

    # Confirm that ai_insights was saved to DB
    db = TestingSessionLocal()
    updated_session = db.query(Session).filter(Session.id == s_id).first()
    assert updated_session.ai_insights is not None
    assert updated_session.ai_insights["risk_tier"] == "Green"
    db.close()


def test_ai_explain_cross_user_isolation(test_user, other_user):
    """User A cannot access or explain User B's session."""
    db = TestingSessionLocal()
    session_other = Session(
        user_id=other_user["user_id"],
        risk_tier="Red",
        speech_rate=80.0,
    )
    db.add(session_other)
    db.commit()
    db.refresh(session_other)
    s_id = session_other.id
    db.close()

    headers = {"Authorization": f"Bearer {test_user['token']}"}
    res = client.post("/api/ai/explain", json={"session_id": s_id}, headers=headers)
    assert res.status_code == 404


def test_ai_weekly_summary_endpoint(test_user):
    """POST /api/ai/weekly-summary aggregates user sessions and returns structured narrative."""
    db = TestingSessionLocal()
    for i in range(3):
        s = Session(
            user_id=test_user["user_id"],
            risk_tier="Green",
            speech_rate=115.0 + i,
            pause_frequency=3.0,
            recorded_at=datetime.utcnow() - timedelta(days=i),
        )
        db.add(s)
    db.commit()
    db.close()

    headers = {"Authorization": f"Bearer {test_user['token']}"}
    res = client.post("/api/ai/weekly-summary", json={}, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "overall_narrative" in data
    assert len(data["key_observations"]) > 0
    assert "trend_analysis" in data
    assert "recommendations" in data


def test_caregiver_summary_authenticated_user(test_user):
    """POST /api/ai/caregiver-summary accessible with JWT auth."""
    headers = {"Authorization": f"Bearer {test_user['token']}"}
    payload = {
        "biomarkers": {"speech_rate": 112.0, "pause_frequency": 3.5},
        "risk_tier": "Green",
        "patient_name": "Mom",
    }
    res = client.post("/api/ai/caregiver-summary", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "Mom" in data["plain_language_summary"]
    assert len(data["supportive_tips"]) >= 3


def test_caregiver_summary_with_valid_share_token(test_user):
    """
    Caregiver without JWT user token can access caregiver summary using a valid
    cryptographic CaregiverShare token.
    """
    db = TestingSessionLocal()
    # Create caregiver share
    share = CaregiverShare(
        patient_id=test_user["user_id"],
        share_token="secure_token_abc123xyz789",
        recipient_email="daughter@example.com",
        label="Daughter",
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=30),
        revoked=False,
    )
    db.add(share)

    # Create patient session
    s = Session(
        user_id=test_user["user_id"],
        risk_tier="Green",
        speech_rate=118.0,
        pause_frequency=3.2,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    s_id = s.id
    db.close()

    # Call without Authorization header, using share_token
    payload = {
        "share_token": "secure_token_abc123xyz789",
        "session_id": s_id,
    }
    res = client.post("/api/ai/caregiver-summary", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "plain_language_summary" in data
    assert len(data["observed_changes"]) > 0
    assert len(data["supportive_tips"]) > 0


def test_caregiver_summary_with_revoked_share_token(test_user):
    """Revoked share token must return 403 Forbidden."""
    db = TestingSessionLocal()
    share = CaregiverShare(
        patient_id=test_user["user_id"],
        share_token="revoked_token_111",
        created_at=datetime.utcnow(),
        revoked=True,
    )
    db.add(share)
    db.commit()
    db.close()

    payload = {"share_token": "revoked_token_111"}
    res = client.post("/api/ai/caregiver-summary", json=payload)
    assert res.status_code == 403


def test_caregiver_summary_with_expired_share_token(test_user):
    """Expired share token must return 410 Gone."""
    db = TestingSessionLocal()
    share = CaregiverShare(
        patient_id=test_user["user_id"],
        share_token="expired_token_222",
        created_at=datetime.utcnow() - timedelta(days=20),
        expires_at=datetime.utcnow() - timedelta(days=1),
        revoked=False,
    )
    db.add(share)
    db.commit()
    db.close()

    payload = {"share_token": "expired_token_222"}
    res = client.post("/api/ai/caregiver-summary", json=payload)
    assert res.status_code == 410
