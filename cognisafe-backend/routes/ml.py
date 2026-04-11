import uuid
import threading
import httpx
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

ml_router = APIRouter(prefix="/api/ml", tags=["ml"])

HF_BASE = "https://alamfarzann-cognisafe-ml.hf.space"

# ── In-memory job store ────────────────────────────────────────────────────────
_jobs: dict = {}


def _run_analysis(job_id: str, audio_bytes: bytes, user_id: str):
    """Runs in a background thread. Updates _jobs[job_id] as each stage completes."""
    try:
        _set_stage(job_id, "transcribing")

        import time

        with httpx.Client(timeout=360.0) as client:
            stage_timer = _StageAdvancer(job_id)
            stage_timer.start()

            try:
                response = client.post(
                    f"{HF_BASE}/analyze",
                    files={"audio": ("recording.wav", audio_bytes, "audio/wav")},
                    data={"user_id": user_id},
                )
                stage_timer.stop()
                response.raise_for_status()
            except Exception:
                stage_timer.stop()
                raise

        result = response.json()
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["stage"]  = "done"
        _jobs[job_id]["result"] = result

    except httpx.TimeoutException:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"]  = "ML service timed out after 6 minutes."
    except httpx.HTTPStatusError as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"]  = f"ML service error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"]  = str(e)


def _set_stage(job_id: str, stage: str):
    if job_id in _jobs:
        _jobs[job_id]["stage"]  = stage
        _jobs[job_id]["status"] = "processing"


class _StageAdvancer(threading.Thread):
    STAGES = [
        ("transcribing", 20),
        ("acoustic",     15),
        ("nlp",          30),
        ("risk",          5),
    ]

    def __init__(self, job_id: str):
        super().__init__(daemon=True)
        self.job_id = job_id
        self._stop_event = threading.Event()

    def run(self):
        for stage, delay in self.STAGES:
            _set_stage(self.job_id, stage)
            if self._stop_event.wait(timeout=delay):
                return

    def stop(self):
        self._stop_event.set()


# ── POST /api/ml/analyze (voice) ───────────────────────────────────────────────
@ml_router.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
):
    audio_bytes = await audio.read()
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "status":     "queued",
        "stage":      "uploading",
        "result":     None,
        "error":      None,
        "created_at": datetime.utcnow().isoformat(),
        "user_id":    user_id,
    }

    thread = threading.Thread(
        target=_run_analysis,
        args=(job_id, audio_bytes, user_id),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


# ── GET /api/ml/status/{job_id} ───────────────────────────────────────────────
@ml_router.get("/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage":  job["stage"],
        "result": job["result"] if job["status"] == "done"   else None,
        "error":  job["error"]  if job["status"] == "failed" else None,
    }


# ── POST /api/ml/analyze-text (text mode for mute users) ─────────────────────
class TextAnalyzeRequest(BaseModel):
    text: str
    user_id: str = "demo_user"


@ml_router.post("/analyze-text")
async def analyze_text(req: TextAnalyzeRequest):
    """
    Proxy endpoint: forwards typed text to the HF Space /analyze-text endpoint.
    Used by mute users who cannot do voice sessions.
    Returns the same shape as /analyze so the frontend treats both identically.
    """
    if not req.text or len(req.text.split()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text too short. Please write at least 2-3 sentences."
        )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{HF_BASE}/analyze-text",
                json={"text": req.text, "user_id": req.user_id},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Text analysis timed out. Please try again.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"ML service error {e.response.status_code}: {e.response.text[:200]}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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