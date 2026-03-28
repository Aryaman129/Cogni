import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

ml_router = APIRouter(prefix="/api/ml", tags=["ml"])

HF_BASE = "https://alamfarzann-cognisafe-ml.hf.space"

@ml_router.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    user_id: str = Form(...),
):
    audio_bytes = await audio.read()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{HF_BASE}/analyze",
                files={"audio": ("recording.wav", audio_bytes, "audio/wav")},  # ← force .wav filename
                data={"user_id": user_id},
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="ML service timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"ML error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ML service unreachable: {str(e)}")


@ml_router.get("/health")
async def ml_health():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{HF_BASE}/health")
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))