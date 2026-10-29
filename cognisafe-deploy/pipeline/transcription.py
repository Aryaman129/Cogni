"""
CogniSafe Whisper Audio Transcription Pipeline.
Extracts verbatim text, word timestamps, and pause events with explicit
error classification and complete model metadata.
"""

import logging
import os
import warnings
from typing import Optional, Dict, Any, List
import whisper

warnings.filterwarnings("ignore")
logger = logging.getLogger("cognisafe.transcription")

_MODEL = None
_CURRENT_MODEL_NAME = None


def get_whisper_model(model_name: str = "base"):
    """
    Lazy load Whisper model once and cache in memory.
    Prevents duplicate reloads across multiple pipeline calls.
    """
    global _MODEL, _CURRENT_MODEL_NAME
    if _MODEL is None or _CURRENT_MODEL_NAME != model_name:
        logger.info(f"[Whisper] Loading model '{model_name}'...")
        _MODEL = whisper.load_model(model_name)
        _CURRENT_MODEL_NAME = model_name
        logger.info(f"[Whisper] Model '{model_name}' loaded [OK]")
    return _MODEL


def transcribe_audio(
    audio_path: str,
    model_name: str = "base",
    language: str = "en",
    pause_threshold: float = 0.200,  # 200ms threshold for internal pauses
) -> Dict[str, Any]:
    """
    Transcribe audio file using OpenAI Whisper.
    Returns structured transcript data, word timestamps, pause events,
    explicit error classification, and pipeline metadata.
    """
    metadata = {
        "model_name": f"whisper-{model_name}",
        "engine": "openai-whisper",
        "language": language,
        "pause_threshold_seconds": pause_threshold,
    }

    if not os.path.exists(audio_path):
        logger.error(f"[Whisper ERROR] File not found: {audio_path}")
        return _fallback_transcript(
            status="file_not_found",
            error_message=f"Audio file not found: {audio_path}",
            metadata=metadata
        )

    try:
        model = get_whisper_model(model_name)

        # ── Step 1: Transcribe with word-level timestamps ────────────────────
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language=language
        )

        transcript_text = result.get('text', '').strip()

        # ── Step 2: Extract words with precise start/end timestamps ──────────
        words: List[Dict[str, Any]] = []
        segments = result.get('segments', [])
        for segment in segments:
            for word_info in segment.get('words', []):
                w_str = word_info.get('word', '').strip()
                if w_str:
                    words.append({
                        'word':  w_str,
                        'start': round(float(word_info.get('start', 0.0)), 3),
                        'end':   round(float(word_info.get('end', 0.0)), 3),
                    })

        # ── Step 3: Detect inter-word pauses (gap >= pause_threshold) ────────
        pause_events: List[Dict[str, Any]] = []
        for i in range(1, len(words)):
            gap = words[i]['start'] - words[i - 1]['end']
            if gap >= pause_threshold:
                pause_events.append({
                    'after_word':     words[i - 1]['word'],
                    'before_word':    words[i]['word'],
                    'duration':       round(float(gap), 3),
                    'start_time':     words[i - 1]['end'],
                })

        duration = segments[-1]['end'] if segments else 0.0
        status = "success" if transcript_text else "no_speech"

        logger.info(
            f"[Whisper] Transcribed {len(words)} words, "
            f"{len(pause_events)} pauses detected (status: {status})"
        )

        return {
            'text':         transcript_text,
            'words':        words,
            'pause_events': pause_events,
            'word_count':   len(words),
            'duration':     round(float(duration), 3),
            'status':       status,
            'error_message': None if status == "success" else "No speech detected in audio",
            'model_metadata': metadata,
        }

    except FileNotFoundError as e:
        err_str = str(e).lower()
        if "ffmpeg" in err_str or "winerror 2" in err_str:
            logger.error("[Whisper ERROR] FFmpeg is not installed or not in PATH")
            return _fallback_transcript(
                status="ffmpeg_missing",
                error_message="FFmpeg is not installed or not found in system PATH",
                metadata=metadata
            )
        logger.error(f"[Whisper ERROR] File not found: {e}")
        return _fallback_transcript(
            status="file_not_found",
            error_message=str(e),
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"[Whisper ERROR] Transcription inference failed: {e}", exc_info=True)
        return _fallback_transcript(
            status="inference_failed",
            error_message=str(e),
            metadata=metadata
        )


def _fallback_transcript(
    status: str,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Returns safe structured transcript dictionary on failure."""
    return {
        'text':           '',
        'words':          [],
        'pause_events':   [],
        'word_count':     0,
        'duration':       0.0,
        'status':         status,
        'error_message':  error_message,
        'model_metadata': metadata or {},
    }