"""
tests/e2e/conftest.py
=====================
Shared fixtures, synthetic audio generators, JWT auth tokens, isolated SQLite database,
FastAPI TestClient, and mock services for CogniSafe E2E test suite.
"""

import os
import sys
import math
import struct
import wave
import io
import time
from datetime import datetime, timedelta, timezone
from typing import Generator, Dict, Any, List, Optional
from enum import Enum

import pytest
from pydantic import BaseModel, Field
from fastapi.testclient import TestClient

# Ensure cognisafe-backend and cognisafe-deploy are on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND_DIR = os.path.join(BASE_DIR, "cognisafe-backend")
DEPLOY_DIR = os.path.join(BASE_DIR, "cognisafe-deploy")

for d in (BACKEND_DIR, DEPLOY_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

# Set environment variables for testing before importing backend modules
TEST_SECRET_KEY = "test_super_secret_key_cognisafe_2026_test_e2e_suite_32bytes!"
os.environ["SECRET_KEY"] = TEST_SECRET_KEY
os.environ["ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "60"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["NEBIUS_BASE_URL"] = "https://api.tokenfactory.nebius.com/v1"
os.environ["NEBIUS_MODEL_ID"] = "nvidia/Nemotron-3_5-Lightning"
os.environ["NEBIUS_API_KEY"] = "test-nebius-key-12345"
os.environ["NEBIUS_TIMEOUT_SECONDS"] = "10.0"

from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base as BackendBase
from models import User, Session, Job, CaregiverShare

DBUser = User
DBSessionRecord = Session
DBJob = Job
DBCaregiverShare = CaregiverShare

# ─────────────────────────────────────────────────────────────────────────────
# 1. CANONICAL CONTRACT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_BIOMARKERS = [
    # 10 Acoustic
    "speech_rate",            # words/min
    "articulation_rate",      # words/min (speech duration only)
    "pause_frequency",        # pauses/min
    "pause_duration_mean",    # seconds (null if 0 pauses observed)
    "filled_pause_rate",      # filled pauses/min
    "pitch_mean",             # Hz (voiced frames only)
    "pitch_range",            # Hz (IQR or 5th-95th percentile)
    "jitter",                 # percentage (documented frame F0 perturbation proxy)
    "shimmer",                # percentage (documented voiced frame RMS proxy)
    "HNR",                    # dB (documented spectral HPSS proxy)
    # 4 Linguistic / NLP
    "lexical_diversity",      # MTLD (standard, all tokens included)
    "semantic_coherence",     # embedding cosine similarity (0.0-1.0)
    "idea_density",           # proposition ratio (0.0-1.0)
    "syntactic_complexity",   # mean dependency parse tree depth
]

class FeatureStatus(str, Enum):
    VALID = "valid"
    INSUFFICIENT_INPUT = "insufficient_input"
    EXTRACTION_FAILED = "extraction_failed"
    UNSUPPORTED_INPUT = "unsupported_input"
    MISSING_DEPENDENCY = "missing_dependency"
    INVALID_MEASUREMENT = "invalid_measurement"

class TypedFeatureResult(BaseModel):
    status: FeatureStatus
    value: Optional[float] = None
    unit: str
    is_proxy: bool = False
    method: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class NebiusStructuredResponse(BaseModel):
    summary: str
    clinical_explanation: str
    caregiver_note: str
    referenced_biomarkers: List[str]
    confidence_score: float = Field(..., ge=0.0, le=1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC AUDIO GENERATION FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_wav(
    duration_s: float = 3.0,
    sample_rate: int = 16000,
    freq_hz: float = 220.0,
    silence: bool = False,
    clipped: bool = False,
    with_pauses: bool = True
) -> bytes:
    """
    Generate valid PCM 16-bit mono RIFF/WAV binary stream.
    Zero external ML library dependencies; uses standard python wave/struct.
    """
    total_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        if silence:
            raw_data = b'\x00\x00' * total_samples
        elif clipped:
            # Peg samples to extreme max / min
            samples = []
            for i in range(total_samples):
                val = 32767 if (i // 100) % 2 == 0 else -32768
                samples.append(struct.pack('<h', val))
            raw_data = b''.join(samples)
        else:
            samples = []
            # Generate simulated harmonic speech tone with 300ms pause bursts
            pause_len = int(sample_rate * 0.3)
            speech_len = int(sample_rate * 0.7)
            cycle_len = pause_len + speech_len

            for i in range(total_samples):
                pos_in_cycle = i % cycle_len
                if with_pauses and pos_in_cycle < pause_len:
                    samples.append(struct.pack('<h', 0))
                else:
                    # Harmonic voice proxy: fundamental + 2 harmonics
                    t = float(i) / sample_rate
                    val = 0.6 * math.sin(2.0 * math.pi * freq_hz * t)
                    val += 0.3 * math.sin(2.0 * math.pi * (freq_hz * 2.0) * t)
                    val += 0.1 * math.sin(2.0 * math.pi * (freq_hz * 3.0) * t)
                    int_val = int(val * 24000)
                    samples.append(struct.pack('<h', int_val))
            raw_data = b''.join(samples)

        wf.writeframes(raw_data)

    buf.seek(0)
    return buf.read()

@pytest.fixture
def synthetic_speech_wav() -> bytes:
    """Standard 3-second harmonic speech WAV."""
    return create_synthetic_wav(duration_s=3.0, freq_hz=180.0, with_pauses=True)

@pytest.fixture
def synthetic_silent_wav() -> bytes:
    """Pure silent audio stream (all zeros)."""
    return create_synthetic_wav(duration_s=3.0, silence=True)

@pytest.fixture
def synthetic_clipped_wav() -> bytes:
    """Severely clipped / distorted audio stream."""
    return create_synthetic_wav(duration_s=3.0, clipped=True)

@pytest.fixture
def synthetic_short_wav() -> bytes:
    """Sub-minimum duration audio (<1.0 second)."""
    return create_synthetic_wav(duration_s=0.4, freq_hz=200.0)

@pytest.fixture
def corrupted_audio_bytes() -> bytes:
    """Corrupted WAV header / random truncated binary."""
    return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00" + b"\xff" * 64

@pytest.fixture
def oversized_audio_bytes() -> bytes:
    """Payload exceeding 25MB boundary limit."""
    return b"RIFF" + b"\x00" * (26 * 1024 * 1024)

# ─────────────────────────────────────────────────────────────────────────────
# 3. AUTHENTICATION & JWT FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def generate_jwt_token(
    user_id: int,
    secret_key: str = TEST_SECRET_KEY,
    expires_delta_minutes: int = 60,
    tamper_signature: bool = False
) -> str:
    """Generate a signed JWT token."""
    exp = datetime.now(timezone.utc) + timedelta(minutes=expires_delta_minutes)
    payload = {
        "sub": str(user_id),
        "exp": exp,
        "iat": datetime.now(timezone.utc)
    }
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    if tamper_signature:
        parts = token.split(".")
        tampered_sig = parts[2][:-4] + "wxyz"
        return f"{parts[0]}.{parts[1]}.{tampered_sig}"
    return token

@pytest.fixture
def auth_headers_user_a() -> Dict[str, str]:
    token = generate_jwt_token(user_id=101)
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def auth_headers_user_b() -> Dict[str, str]:
    token = generate_jwt_token(user_id=202)
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def expired_auth_headers() -> Dict[str, str]:
    token = generate_jwt_token(user_id=101, expires_delta_minutes=-10)
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def tampered_auth_headers() -> Dict[str, str]:
    token = generate_jwt_token(user_id=101, tamper_signature=True)
    return {"Authorization": f"Bearer {token}"}

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATABASE & FASTAPI TEST CLIENT FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

from sqlalchemy.pool import StaticPool

@pytest.fixture(scope="function")
def test_engine():
    """Create a pristine in-memory SQLite engine for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    BackendBase.metadata.create_all(bind=engine)
    return engine

@pytest.fixture(scope="function")
def test_db_session(test_engine):
    """Provide isolated db session with rollback / cleanup."""
    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = SessionMaker()
    try:
        yield db
    finally:
        db.close()

@pytest.fixture
def seed_user_and_sessions(test_db_session):
    """
    Factory fixture to seed users and session histories (preliminary vs mature).
    """
    def _seed(user_id: int, session_count: int, start_date: Optional[datetime] = None, drift: bool = False):
        user = test_db_session.query(DBUser).filter(DBUser.id == user_id).first()
        if not user:
            user = DBUser(
                id=user_id,
                name=f"Patient {user_id}",
                email=f"patient_{user_id}@cognisafe.test",
                password_hash="hashed_pw_test",
                dob="1960-01-01"
            )
            test_db_session.add(user)
            test_db_session.commit()

        base_time = start_date or (datetime.now(timezone.utc) - timedelta(days=session_count))
        created_sessions = []
        for i in range(session_count):
            session_time = base_time + timedelta(days=i)
            decay = (i * 0.05) if drift else 0.0
            s = DBSessionRecord(
                user_id=user_id,
                recorded_at=session_time,
                duration_seconds=30.0,
                mode="voice",
                analysis_status="complete" if session_count >= 5 else "insufficient_baseline",
                risk_tier="Green" if session_count >= 5 and not drift else ("Orange" if drift and i > 5 else None),
                speech_rate=max(60.0, 130.0 - (decay * 20.0)),
                articulation_rate=max(80.0, 150.0 - (decay * 15.0)),
                pause_frequency=min(12.0, 3.0 + (decay * 2.0)),
                pause_duration_mean=min(2.5, 0.45 + (decay * 0.2)),
                filled_pause_rate=min(5.0, 0.8 + (decay * 0.5)),
                pitch_mean=195.0,
                pitch_range=65.0,
                jitter=1.1 + (decay * 0.3),
                shimmer=4.2 + (decay * 0.5),
                hnr=max(8.0, 18.5 - decay),
                lexical_diversity=max(40.0, 72.0 - (decay * 5.0)),
                semantic_coherence=max(0.4, 0.82 - (decay * 0.08)),
                idea_density=max(0.3, 0.54 - (decay * 0.04)),
                syntactic_complexity=max(2.0, 4.2 - (decay * 0.3)),
                transcript="This is a standardized picture description test sentence.",
            )
            test_db_session.add(s)
            created_sessions.append(s)

        test_db_session.commit()
        return user, created_sessions

    return _seed

@pytest.fixture
def client(test_db_session):
    """
    FastAPI TestClient with overridden get_db dependency.
    """
    from main import app
    from database import get_db

    def _override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
