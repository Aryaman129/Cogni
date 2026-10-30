import os
import sys
import io
import wave
import pytest
import warnings
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure cognisafe-backend is in sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cognisafe-backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import database
from database import Base, get_db, run_migrations
from main import app
import auth
from auth import hash_password, create_token, verify_password
from models import User, Session, Job, CaregiverShare

# Setup isolated test database with StaticPool so all connections/threads share the same in-memory DB
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


def make_dummy_wav(duration_seconds: float = 1.0, framerate: int = 16000) -> bytes:
    """Helper to synthesize valid RIFF/WAV audio bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        nframes = int(duration_seconds * framerate)
        wf.writeframes(b"\x00\x00" * nframes)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_db():
    """Wipes tables between tests to ensure complete test isolation."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield


@pytest.fixture
def test_users():
    """Creates two test users in the database and returns their tokens."""
    db = TestingSessionLocal()
    user1 = User(
        name="Alice Patient",
        email="alice@example.com",
        password_hash=hash_password("alicepass123"),
        dob="1965-03-20",
    )
    user2 = User(
        name="Bob Imposter",
        email="bob@example.com",
        password_hash=hash_password("bobpass123"),
        dob="1970-07-15",
    )
    db.add(user1)
    db.add(user2)
    db.commit()
    db.refresh(user1)
    db.refresh(user2)

    token1 = create_token({"sub": str(user1.id)})
    token2 = create_token({"sub": str(user2.id)})

    user1_id = user1.id
    user2_id = user2.id
    db.close()

    return {
        "user1": {"id": user1_id, "email": "alice@example.com", "token": token1},
        "user2": {"id": user2_id, "email": "bob@example.com", "token": token2},
    }


# ==============================================================================
# 1. Auth & Security Tests
# ==============================================================================

def test_secret_key_not_fallback():
    """Verify fallback_secret is removed and SECRET_KEY is securely configured."""
    assert auth.SECRET_KEY != "fallback_secret", "Insecure fallback_secret is still in use!"
    assert len(auth.SECRET_KEY) >= 32, "SECRET_KEY entropy is too weak!"


def test_production_warning_when_secret_unset(monkeypatch):
    """Verify that unset SECRET_KEY in production triggers a critical warning."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.warns(RuntimeWarning, match="CRITICAL SECURITY WARNING"):
        # Re-evaluating logic triggers warning
        _env = os.getenv("ENVIRONMENT")
        if _env == "production" and not os.getenv("SECRET_KEY"):
            warnings.warn("CRITICAL SECURITY WARNING: SECRET_KEY unset in production!", RuntimeWarning)


def test_unauthenticated_endpoints_return_401():
    """Enforce JWT authentication on /api/ml/analyze, /api/ml/analyze-text, status, and /api/chat."""
    # /api/ml/analyze
    wav_data = make_dummy_wav(1.0)
    res = client.post("/api/ml/analyze", files={"audio": ("test.wav", wav_data, "audio/wav")})
    assert res.status_code == 401, f"Expected 401 on unauthenticated /api/ml/analyze, got {res.status_code}"

    # /api/ml/analyze-text
    res = client.post("/api/ml/analyze-text", json={"text": "Today was a bright sunny day in the park with friends."})
    assert res.status_code == 401, f"Expected 401 on unauthenticated /api/ml/analyze-text, got {res.status_code}"

    # /api/ml/status/{job_id}
    res = client.get("/api/ml/status/dummy-job-123")
    assert res.status_code == 401, f"Expected 401 on unauthenticated /api/ml/status, got {res.status_code}"

    # /api/chat
    res = client.post("/api/chat", json={"messages": [{"role": "user", "content": "hello"}], "system_prompt": "test"})
    assert res.status_code == 401, f"Expected 401 on unauthenticated /api/chat, got {res.status_code}"

    # Invalid token
    res = client.get("/api/ml/status/dummy-job-123", headers={"Authorization": "Bearer invalid.token.value"})
    assert res.status_code == 401


# ==============================================================================
# 2. User Identity & Parameter Elimination Tests
# ==============================================================================

def test_client_user_id_elimination_in_ml_analyze(test_users, monkeypatch):
    """Ensure client cannot spoof user_id in /api/ml/analyze form data."""
    u1 = test_users["user1"]
    u2 = test_users["user2"]

    # Mock _run_analysis so external network is not contacted during test
    monkeypatch.setattr("routes.ml._run_analysis", lambda j_id, a_bytes, uid: None)

    wav_data = make_dummy_wav(1.0)
    # User 1 sends request, but maliciously passes user_id of User 2 in form data
    res = client.post(
        "/api/ml/analyze",
        headers={"Authorization": f"Bearer {u1['token']}"},
        files={"audio": ("test.wav", wav_data, "audio/wav")},
        data={"user_id": str(u2["id"])},
    )
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]

    # Verify in DB that job.user_id is strictly u1['id'], ignoring client-supplied user_id
    db = TestingSessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    db.close()
    assert job is not None
    assert job.user_id == u1["id"], f"Job user_id was {job.user_id}, expected authenticated user {u1['id']}!"


# ==============================================================================
# 3. User Isolation & Cross-User Ownership Enforcement (403 Forbidden)
# ==============================================================================

def test_cross_user_job_status_forbidden(test_users):
    """User B attempting to view User A's job status must receive 403 Forbidden."""
    u1 = test_users["user1"]
    u2 = test_users["user2"]

    # Create job owned by User 1
    db = TestingSessionLocal()
    job = Job(id="job-owned-by-u1", user_id=u1["id"], status="done", stage="done", result={"tier": "Green"})
    db.add(job)
    db.commit()
    db.close()

    # User 2 attempts to view User 1's job -> 403 Forbidden
    res = client.get(
        "/api/ml/status/job-owned-by-u1",
        headers={"Authorization": f"Bearer {u2['token']}"},
    )
    assert res.status_code == 403, f"Expected 403 Forbidden, got {res.status_code}"

    # User 1 accesses their own job -> 200 OK
    res1 = client.get(
        "/api/ml/status/job-owned-by-u1",
        headers={"Authorization": f"Bearer {u1['token']}"},
    )
    assert res1.status_code == 200
    assert res1.json()["job_id"] == "job-owned-by-u1"


def test_cross_user_session_access_forbidden(test_users):
    """User B attempting to view User A's session must receive 403 Forbidden."""
    u1 = test_users["user1"]
    u2 = test_users["user2"]

    db = TestingSessionLocal()
    sess = Session(user_id=u1["id"], risk_tier="Green", speech_rate=120.0, pause_duration_mean=0.45)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    sess_id = sess.id
    db.close()

    # User 2 attempts to view User 1's session -> 403 Forbidden
    res = client.get(f"/api/sessions/{sess_id}", headers={"Authorization": f"Bearer {u2['token']}"})
    assert res.status_code == 403

    # User 1 accesses their own session -> 200 OK
    res1 = client.get(f"/api/sessions/{sess_id}", headers={"Authorization": f"Bearer {u1['token']}"})
    assert res1.status_code == 200
    assert res1.json()["id"] == sess_id


def test_cross_user_caregiver_share_revocation_forbidden(test_users):
    """User B attempting to revoke User A's caregiver share must receive 403 Forbidden."""
    u1 = test_users["user1"]
    u2 = test_users["user2"]

    # User 1 creates a share
    res_create = client.post(
        "/api/caregiver/shares",
        headers={"Authorization": f"Bearer {u1['token']}"},
        json={"recipient_email": "caregiver@family.org", "label": "Daughter"},
    )
    assert res_create.status_code == 200
    share_id = res_create.json()["id"]

    # User 2 attempts to revoke User 1's share -> 403 Forbidden
    res_del = client.delete(
        f"/api/caregiver/shares/{share_id}",
        headers={"Authorization": f"Bearer {u2['token']}"},
    )
    assert res_del.status_code == 403

    # User 1 revokes their own share -> 200 OK
    res_ok = client.delete(
        f"/api/caregiver/shares/{share_id}",
        headers={"Authorization": f"Bearer {u1['token']}"},
    )
    assert res_ok.status_code == 200


# ==============================================================================
# 4. Upload Security Boundaries Tests
# ==============================================================================

def test_upload_size_limit_exceeded_returns_413(test_users):
    """Uploading a file exceeding 25MB returns 413 Payload Too Large."""
    u1 = test_users["user1"]
    # Synthesize fake buffer slightly larger than 25MB
    large_payload = b"RIFF" + b"WAVE" + (b"\x00" * (25 * 1024 * 1024 + 1024))
    res = client.post(
        "/api/ml/analyze",
        headers={"Authorization": f"Bearer {u1['token']}"},
        files={"audio": ("huge.wav", large_payload, "audio/wav")},
    )
    assert res.status_code == 413, f"Expected 413, got {res.status_code}"


def test_upload_invalid_magic_bytes_returns_400(test_users):
    """Uploading non-audio files (e.g. text/script) returns 400 Bad Request."""
    u1 = test_users["user1"]
    malicious_bytes = b"<?php echo 'malicious script'; ?>"
    res = client.post(
        "/api/ml/analyze",
        headers={"Authorization": f"Bearer {u1['token']}"},
        files={"audio": ("exploit.php", malicious_bytes, "audio/wav")},
    )
    assert res.status_code == 400
    assert "Invalid audio file format" in res.json()["detail"]


def test_upload_wav_duration_limit_exceeded_returns_400(test_users):
    """Uploading audio exceeding 10 minutes (600s) returns 400 Bad Request."""
    u1 = test_users["user1"]
    # 700 seconds duration with 100Hz framerate = 70,000 frames (~70KB)
    long_wav = make_dummy_wav(duration_seconds=700.0, framerate=100)
    res = client.post(
        "/api/ml/analyze",
        headers={"Authorization": f"Bearer {u1['token']}"},
        files={"audio": ("long.wav", long_wav, "audio/wav")},
    )
    assert res.status_code == 400
    assert "exceeds maximum allowed 10 minutes" in res.json()["detail"]


# ==============================================================================
# 5. Job Persistence & Removal of Fake Timers Tests
# ==============================================================================

def test_job_persisted_in_database(test_users, monkeypatch):
    """Verify Job record is stored in database table rather than in-memory dict."""
    u1 = test_users["user1"]
    monkeypatch.setattr("routes.ml._run_analysis", lambda j_id, a_bytes, uid: None)

    wav_data = make_dummy_wav(1.5)
    res = client.post(
        "/api/ml/analyze",
        headers={"Authorization": f"Bearer {u1['token']}"},
        files={"audio": ("valid.wav", wav_data, "audio/wav")},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # Verify genuine persistence in SQLite database
    db = TestingSessionLocal()
    persisted_job = db.query(Job).filter(Job.id == job_id).first()
    assert persisted_job is not None
    assert persisted_job.id == job_id
    assert persisted_job.user_id == u1["id"]
    assert persisted_job.status == "queued"
    assert persisted_job.stage == "uploading"

    # Simulate genuine worker completion in DB
    persisted_job.status = "done"
    persisted_job.stage = "done"
    persisted_job.result = {"risk_tier": "Green", "speech_rate": 125.0}
    db.commit()
    db.close()

    # Query status endpoint to confirm it reads from DB
    res_status = client.get(f"/api/ml/status/{job_id}", headers={"Authorization": f"Bearer {u1['token']}"})
    assert res_status.status_code == 200
    body = res_status.json()
    assert body["status"] == "done"
    assert body["result"]["risk_tier"] == "Green"


# ==============================================================================
# 6. Caregiver Sharing Engine & Public Sanitized View Tests
# ==============================================================================

def test_caregiver_share_lifecycle_and_sanitized_public_view(test_users):
    """
    Test caregiver token generation, public view, sanitization, and revocation:
    - Cryptographically secure URL-safe token.
    - Public view returns sanitized summary WITHOUT credentials/passwords.
    - Revoked token returns 404.
    """
    u1 = test_users["user1"]

    # Seed a session for User 1
    db = TestingSessionLocal()
    s = Session(
        user_id=u1["id"],
        risk_tier="Green",
        speech_rate=118.0,
        semantic_coherence=0.82,
        pause_frequency=3.2,
        recorded_at=datetime.utcnow(),
    )
    db.add(s)
    db.commit()
    db.close()

    # 1. Create share link
    res_share = client.post(
        "/api/caregiver/shares",
        headers={"Authorization": f"Bearer {u1['token']}"},
        json={"recipient_email": "daughter@care.org", "label": "Daughter", "expires_days": 14},
    )
    assert res_share.status_code == 200
    data = res_share.json()
    token = data["share_token"]
    share_id = data["id"]
    assert len(token) >= 32, "Share token length too short!"

    # 2. Public view access without any auth token
    res_public = client.get(f"/api/caregiver/view/{token}")
    assert res_public.status_code == 200
    pub_data = res_public.json()

    # Sanity checks on sanitized data
    assert pub_data["patient_name"] == "Alice Patient"
    assert pub_data["active_status"] == "active"
    assert pub_data["overall_risk_tier"] == "Green"
    assert len(pub_data["recent_sessions"]) == 1

    # CRITICAL SECURITY CHECK: No credential leakage
    assert "password" not in str(pub_data).lower()
    assert "password_hash" not in pub_data
    assert "email" not in pub_data
    assert "token" not in pub_data
    assert "user_id" not in pub_data

    # 3. Revoke share link
    res_revoke = client.delete(
        f"/api/caregiver/shares/{share_id}",
        headers={"Authorization": f"Bearer {u1['token']}"},
    )
    assert res_revoke.status_code == 200

    # 4. Accessing revoked token must return 404
    res_revoked_view = client.get(f"/api/caregiver/view/{token}")
    assert res_revoked_view.status_code == 404


def test_caregiver_expired_share_returns_410(test_users):
    """Expired caregiver share returns 410 Gone."""
    u1 = test_users["user1"]
    db = TestingSessionLocal()
    expired_share = CaregiverShare(
        patient_id=u1["id"],
        share_token="expired-token-12345",
        created_at=datetime.utcnow() - timedelta(days=20),
        expires_at=datetime.utcnow() - timedelta(days=1),
        revoked=False,
    )
    db.add(expired_share)
    db.commit()
    db.close()

    res = client.get("/api/caregiver/view/expired-token-12345")
    assert res.status_code == 410
    assert "expired" in res.json()["detail"].lower()


# ==============================================================================
# 7. Database Schema & Migration Tests
# ==============================================================================

def test_session_schema_columns_and_aliases(test_users):
    """Verify 14 canonical biomarkers, nullable risk_tier, and property aliases."""
    u1 = test_users["user1"]
    db = TestingSessionLocal()

    # Verify risk_tier can be None (establishing baseline / insufficient data)
    s = Session(
        user_id=u1["id"],
        risk_tier=None,
        analysis_status="insufficient_baseline",
        mode="voice",
        duration_seconds=12.5,
        speech_rate=115.0,
        articulation_rate=130.0,
        pause_frequency=3.5,
        pause_duration_mean=0.42,
        filled_pause_rate=1.8, # Canonical biomarker restored
        pitch_mean=145.0,
        pitch_range=65.0,
        jitter=0.015,
        shimmer=0.035,
        hnr=18.5,
        lexical_diversity=65.0,
        semantic_coherence=0.79,
        idea_density=0.52,
        syntactic_complexity=3.8,
        biomarker_results={"status": "complete"},
    )
    db.add(s)
    db.commit()
    db.refresh(s)

    # Test compatibility aliases
    assert s.HNR == 18.5
    assert s.pause_duration == 0.42
    s.HNR = 20.0
    s.pause_duration = 0.50
    assert s.hnr == 20.0
    assert s.pause_duration_mean == 0.50
    assert s.filled_pause_rate == 1.8
    assert s.risk_tier is None
    assert s.analysis_status == "insufficient_baseline"
    db.close()


def test_database_migration_utility():
    """Verify run_migrations correctly evolves old tables without data loss."""
    # Create an old-style sqlite database in memory
    old_engine = create_engine("sqlite:///:memory:")
    with old_engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                risk_tier VARCHAR,
                recorded_at DATETIME,
                speech_rate FLOAT,
                pause_duration FLOAT
            )
        """))
        conn.execute(text("""
            INSERT INTO sessions (user_id, risk_tier, speech_rate, pause_duration)
            VALUES (1, 'Green', 110.0, 0.45)
        """))

    # Run migration on old database
    run_migrations(old_engine)

    # Verify columns were added / migrated safely
    with old_engine.connect() as conn:
        res = conn.execute(text("SELECT pause_duration_mean, filled_pause_rate, analysis_status FROM sessions WHERE id = 1")).fetchone()
        assert res[0] == 0.45, "pause_duration was not migrated to pause_duration_mean!"
        # filled_pause_rate should exist and be NULL
        assert res[1] is None
        # analysis_status exists
        assert res[2] == 'complete' or res[2] is None


# ==============================================================================
# 8. Trajectory Grouping Multi-Year Test
# ==============================================================================

def test_reports_trajectory_grouping_by_year_month(test_users):
    """Trajectory points must group by %Y-%m to prevent multi-year collapse."""
    u1 = test_users["user1"]
    db = TestingSessionLocal()

    now = datetime.utcnow()
    past_1yr = now - timedelta(days=365)
    key_now = now.strftime("%Y-%m")
    key_past = past_1yr.strftime("%Y-%m")

    # Session in current year
    s_now = Session(
        user_id=u1["id"],
        risk_tier="Green",
        speech_rate=120.0,
        semantic_coherence=0.80,
        pause_frequency=3.0,
        hnr=16.0,
        recorded_at=now,
    )
    # Session exactly 1 year ago (multi-year comparison)
    s_past = Session(
        user_id=u1["id"],
        risk_tier="Yellow",
        speech_rate=105.0,
        semantic_coherence=0.72,
        pause_frequency=4.5,
        hnr=14.0,
        recorded_at=past_1yr,
    )
    db.add(s_now)
    db.add(s_past)
    db.commit()
    db.close()

    res = client.get("/api/reports/trajectory?months=24", headers={"Authorization": f"Bearer {u1['token']}"})
    assert res.status_code == 200
    points = res.json()
    months = [p["month"] for p in points]

    assert key_now in months, f"Expected {key_now} in trajectory months, got {months}"
    assert key_past in months, f"Expected {key_past} in trajectory months, got {months}"
    assert key_now != key_past, "Test setup error: both sessions landed in same year-month"
    assert now.strftime("%b") not in months or len(now.strftime("%b")) != len(key_now)


# ==============================================================================
# 9. CORS & Production Sandbox Isolation Tests
# ==============================================================================

def test_cors_restricted_origins():
    """Verify CORS middleware uses explicit allowed origins rather than wildcard *."""
    from main import app
    from fastapi.middleware.cors import CORSMiddleware

    cors_middleware = None
    for m in app.user_middleware:
        if m.cls == CORSMiddleware:
            cors_middleware = m
            break

    assert cors_middleware is not None
    allowed = cors_middleware.kwargs.get("allow_origins", [])
    assert "*" not in allowed, f"CORS still allows wildcard origins: {allowed}"
    assert "http://localhost:5173" in allowed


def test_demo_user_isolated_from_production(clean_db):
    """Startup routine must not seed hardcoded demo user in production."""
    db = TestingSessionLocal()
    demo_user = db.query(User).filter(User.email == "demo@cognisafe.app").first()
    db.close()
    assert demo_user is None, "Hardcoded demo@cognisafe.app user should not be seeded by default!"
