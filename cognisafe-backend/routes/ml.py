import uuid
import io
import wave
import threading
import httpx
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from database import get_db, SessionLocal
from models import User, Job
from dependencies import get_current_user

ml_router = APIRouter(prefix="/api/ml", tags=["ml"])

HF_BASE = "https://alamfarzann-cognisafe-ml.hf.space"
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB
MAX_DURATION_SECONDS = 600.0        # 10 minutes


def is_valid_audio_magic_bytes(data: bytes) -> bool:
    """Verifies audio magic bytes for WAV, MP3, OGG, FLAC, WebM, and M4A/AAC."""
    if len(data) < 4:
        return False

    # RIFF / WAV
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WAVE":
        return True

    # MP3 (ID3 tag or sync word)
    if data[:3] == b"ID3":
        return True
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True

    # OGG / Opus
    if data[:4] == b"OggS":
        return True

    # FLAC
    if data[:4] == b"fLaC":
        return True

    # WebM / EBML
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return True

    # M4A / MP4
    if len(data) >= 8 and (data[4:8] in (b"ftyp", b"moov")):
        return True

    # AAC ADTS header
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xF6) == 0xF0:
        return True

    return False


def validate_audio_file(audio_bytes: bytes, filename: str = ""):
    """Enforces upload limits, format/magic bytes, and duration boundaries."""
    if len(audio_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size ({len(audio_bytes)} bytes) exceeds 25MB maximum limit.",
        )

    if not is_valid_audio_magic_bytes(audio_bytes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid audio file format. Unrecognized audio header or magic bytes.",
        )

    # Check WAV duration if applicable
    if audio_bytes[:4] == b"RIFF" and len(audio_bytes) >= 12 and audio_bytes[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                framerate = wf.getframerate()
                nframes = wf.getnframes()
                if framerate > 0:
                    duration = nframes / float(framerate)
                    if duration > MAX_DURATION_SECONDS:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Audio duration ({duration:.1f}s) exceeds maximum allowed 10 minutes (600s).",
                        )
        except HTTPException:
            raise
        except Exception:
            # If wave parsing fails, let downstream pipeline validate or raise
            pass


def _run_analysis(job_id: str, audio_bytes: bytes, user_id: int):
    """
    Genuine asynchronous execution worker.
    Tracks state transitions directly in the persistent database Job record.
    Eliminates artificial sleep timers.
    """
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return

        job.status = "processing"
        job.stage = "processing"
        db.commit()

        with httpx.Client(timeout=360.0) as client:
            response = client.post(
                f"{HF_BASE}/analyze",
                files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
                data={"user_id": str(user_id)},
            )
            response.raise_for_status()
            result = response.json()

        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "done"
            job.stage = "done"
            job.result = result
            job.error = None
            db.commit()

    except httpx.TimeoutException:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.stage = "failed"
            job.error = "ML service timed out after 6 minutes."
            db.commit()
    except httpx.HTTPStatusError as e:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.stage = "failed"
            job.error = f"ML service error {e.response.status_code}: {e.response.text[:200]}"
            db.commit()
    except Exception as e:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "failed"
            job.stage = "failed"
            job.error = str(e)
            db.commit()
    finally:
        db.close()


# ── POST /api/ml/analyze (voice) ───────────────────────────────────────────────
@ml_router.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    db: DBSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Enforce streaming byte limits to prevent OOM
    chunks = []
    total_bytes = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    while True:
        chunk = await audio.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds 25MB maximum limit.",
            )
        chunks.append(chunk)

    audio_bytes = b"".join(chunks)

    # Validate audio header and duration
    validate_audio_file(audio_bytes, audio.filename or "")

    job_id = str(uuid.uuid4())

    # Create persistent Job in database
    job = Job(
        id=job_id,
        user_id=user.id,
        status="queued",
        stage="uploading",
        result=None,
        error=None,
    )
    db.add(job)
    db.commit()

    # Launch genuine execution worker
    thread = threading.Thread(
        target=_run_analysis,
        args=(job_id, audio_bytes, user.id),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


# ── GET /api/ml/status/{job_id} ───────────────────────────────────────────────
@ml_router.get("/status/{job_id}")
def get_status(
    job_id: str,
    db: DBSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")

    # Strict user ownership check: cross-user access returns 403 Forbidden
    if job.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to view this job.",
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "stage":  job.stage,
        "result": job.result if job.status == "done" else None,
        "error":  job.error if job.status == "failed" else None,
    }


# ── POST /api/ml/analyze-text (text mode for mute users) ─────────────────────
class TextAnalyzeRequest(BaseModel):
    text: str


@ml_router.post("/analyze-text")
async def analyze_text(
    req: TextAnalyzeRequest,
    user: User = Depends(get_current_user),
):
    """
    Proxy endpoint for text analysis.
    User identity is derived strictly from authenticated JWT token.
    """
    if not req.text or len(req.text.split()) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text too short. Please write at least 2-3 sentences."
        )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{HF_BASE}/analyze-text",
                json={"text": req.text, "user_id": str(user.id)},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Text analysis timed out. Please try again.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"ML service error {e.response.status_code}: {e.response.text[:200]}"
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# ── GET /api/ml/warmup ────────────────────────────────────────────────────────
@ml_router.get("/warmup")
async def warmup():
    """Call this before recording to wake up the HF Space."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.get(f"{HF_BASE}/health")
            return {"status": "warmed", "hf": r.json()}
    except Exception as e:
        return {"status": "warming", "detail": str(e)}