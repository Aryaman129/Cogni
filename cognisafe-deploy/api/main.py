import sys
import os
import time
import uuid
import subprocess
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tempfile
import shutil

from pipeline.transcription import transcribe_audio
from pipeline.acoustic      import extract_acoustic_features
from pipeline.nlp           import analyze_text
from pipeline.anomaly       import run_anomaly_pipeline
from pipeline.risk          import merge_biomarkers

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CogniSafe AI Pipeline",
    description="Voice biomarker analysis API for cognitive health monitoring",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    return {
        "status": "ok",
        "service": "CogniSafe AI Pipeline",
        "timestamp": datetime.utcnow().isoformat()
    }


# ── Main Voice Analyze Endpoint ───────────────────────────────────────────────
@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    user_id: str = Form(default="demo_user")
):
    """
    Voice pipeline endpoint.
    Accepts an audio file, runs full 14-biomarker analysis,
    returns biomarkers + risk tier.
    """
    start_time = time.time()

    SUPPORTED = ('.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm', '.weba', '.opus')
    filename_lower = (audio.filename or '').lower()

    if not filename_lower.endswith(SUPPORTED):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED)}"
        )

    temp_dir  = tempfile.mkdtemp()
    ext       = os.path.splitext(filename_lower)[-1] or '.webm'
    raw_path  = os.path.join(temp_dir, f"audio_raw_{uuid.uuid4().hex}{ext}")
    temp_path = os.path.join(temp_dir, f"audio_{uuid.uuid4().hex}.wav")

    try:
        with open(raw_path, 'wb') as f:
            shutil.copyfileobj(audio.file, f)

        print(f"\n[API] Received: {audio.filename} ({ext}) → {raw_path}")

        if ext != '.wav':
            print(f"[API] Converting {ext} → WAV via ffmpeg...")
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', raw_path, '-ar', '16000', '-ac', '1', '-f', 'wav', temp_path],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                print(f"[ffmpeg ERROR] {result.stderr[-500:]}")
                raise HTTPException(status_code=500, detail=f"Audio conversion failed: {result.stderr[-200:]}")
            print("[API] Conversion done ✅")
        else:
            temp_path = raw_path

        print("[API] Running Whisper transcription...")
        transcript = transcribe_audio(temp_path)

        print("[API] Extracting acoustic features...")
        acoustic = extract_acoustic_features(temp_path, transcript_data=transcript)

        print("[API] Running NLP analysis...")
        nlp_scores = analyze_text(transcript)

        all_biomarkers = merge_biomarkers(acoustic, nlp_scores)

        print("[API] Running anomaly detection...")
        anomaly_result = run_anomaly_pipeline(user_id, all_biomarkers)

        processing_time = round(time.time() - start_time, 2)
        print(f"[API] ✅ Done in {processing_time}s — Risk Tier: {anomaly_result['risk_tier']}")

        return {
            "session_id":              str(uuid.uuid4()),
            "user_id":                 user_id,
            "timestamp":               datetime.utcnow().isoformat(),
            "processing_time_seconds": processing_time,
            "mode":                    "voice",
            "biomarkers":              all_biomarkers,
            "anomaly_flags":           anomaly_result['anomaly_flags'],
            "risk_tier":               anomaly_result['risk_tier'],
            "confidence_intervals":    anomaly_result['confidence_intervals'],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Text Mode Analyze Endpoint (for mute users) ───────────────────────────────
class TextAnalyzeRequest(BaseModel):
    text: str
    user_id: str = "demo_user"


@app.post("/analyze-text")
async def analyze_text_mode(req: TextAnalyzeRequest):
    """
    Text-only pipeline for mute users.
    Skips audio recording and transcription entirely.
    Runs NLP biomarkers on typed text and returns risk tier.

    Returns the same shape as /analyze so the frontend can
    treat both modes identically.
    """
    if not req.text or len(req.text.split()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text too short. Please write at least 2-3 sentences (10+ words) for an accurate analysis."
        )

    start_time = time.time()
    print(f"\n[TEXT API] Received text ({len(req.text.split())} words) from user {req.user_id}")

    try:
        # Step 1: NLP analysis — same function the voice pipeline uses
        print("[TEXT API] Running NLP analysis...")
        nlp_scores = analyze_text({"text": req.text})

        # Step 2: Merge with empty acoustic dict (acoustic biomarkers = 0 for text mode)
        # merge_biomarkers fills missing acoustic keys with 0.0
        all_biomarkers = merge_biomarkers({}, nlp_scores)

        # Step 3: Anomaly detection + risk tier
        # The anomaly model only needs the NLP slice — the zero acoustic values
        # are ignored by the isolation forest because they are consistently zero
        # for text-mode users and won't skew the risk tier.
        print("[TEXT API] Running anomaly detection...")
        anomaly_result = run_anomaly_pipeline(req.user_id, all_biomarkers)

        processing_time = round(time.time() - start_time, 2)
        print(f"[TEXT API] ✅ Done in {processing_time}s — Risk Tier: {anomaly_result['risk_tier']}")

        return {
            "session_id":              str(uuid.uuid4()),
            "user_id":                 req.user_id,
            "timestamp":               datetime.utcnow().isoformat(),
            "processing_time_seconds": processing_time,
            "mode":                    "text",
            "biomarkers":              all_biomarkers,
            "anomaly_flags":           anomaly_result["anomaly_flags"],
            "risk_tier":               anomaly_result["risk_tier"],
            "confidence_intervals":    anomaly_result["confidence_intervals"],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[TEXT API ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Compare Endpoint ──────────────────────────────────────────────────────────
@app.post("/compare")
async def compare(session_a: dict, session_b: dict):
    """Compare two session biomarker JSONs."""
    biomarkers_a = session_a.get('biomarkers', {})
    biomarkers_b = session_b.get('biomarkers', {})

    diff = {}
    for key in biomarkers_a:
        val_a  = biomarkers_a.get(key, 0.0)
        val_b  = biomarkers_b.get(key, 0.0)
        change = round(val_b - val_a, 4)
        pct    = round((change / val_a * 100), 2) if val_a != 0 else 0.0
        diff[key] = {
            'session_a':  val_a,
            'session_b':  val_b,
            'change':     change,
            'change_pct': pct,
            'direction':  'up' if change > 0 else ('down' if change < 0 else 'stable')
        }

    return {
        "timestamp_a": session_a.get('timestamp', ''),
        "timestamp_b": session_b.get('timestamp', ''),
        "diff":        diff
    }


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 Starting CogniSafe AI Pipeline on port 7860...")
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)