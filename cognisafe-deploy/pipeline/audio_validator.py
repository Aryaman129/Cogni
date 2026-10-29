"""
Audio Validator and Safe Preflight Utilities for CogniSafe Pipeline.
Implements header magic bytes inspection, duration/silence/clipping checks,
and safe temporary file management without shell injection.
"""

from contextlib import contextmanager
from enum import Enum
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple, Generator
import numpy as np
from pydantic import BaseModel
import soundfile as sf
import librosa

logger = logging.getLogger("cognisafe.audio_validator")


class AudioValidationStatus(str, Enum):
    VALID = "valid"
    FILE_NOT_FOUND = "file_not_found"
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPT_FILE = "corrupt_file"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    SILENT = "silent"
    CLIPPED = "clipped"


class AudioValidationResult(BaseModel):
    is_valid: bool
    status: AudioValidationStatus
    detected_format: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    max_amplitude: Optional[float] = None
    clipping_ratio: Optional[float] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None


# Known audio magic byte signatures
MAGIC_SIGNATURES = [
    # Format, offset, signature bytes
    ("wav", 0, b"RIFF", 8, b"WAVE"),
    ("flac", 0, b"fLaC", None, None),
    ("ogg", 0, b"OggS", None, None),
    ("webm", 0, b"\x1a\x45\xdf\xa3", None, None),
    ("mp3", 0, b"ID3", None, None),
]

MP3_SYNC_WORDS = (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3")


def detect_audio_format_from_bytes(header: bytes) -> Optional[str]:
    """
    Inspect raw binary header bytes (at least 32 bytes) to identify audio format.
    Prevents file-extension spoofing (e.g. text file renamed .wav).
    """
    if len(header) < 4:
        return None

    # Check WAV/RIFF
    if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
        return "wav"

    # Check FLAC
    if header.startswith(b"fLaC"):
        return "flac"

    # Check OGG
    if header.startswith(b"OggS"):
        return "ogg"

    # Check WebM / MKV container
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "webm"

    # Check MP3 ID3 header
    if header.startswith(b"ID3"):
        return "mp3"

    # Check MP3 sync words without ID3 tag
    if len(header) >= 2 and header[:2] in MP3_SYNC_WORDS:
        return "mp3"

    # Check ISO Base Media File Format (M4A / AAC / MP4)
    if len(header) >= 12 and header[4:8] == b"ftyp":
        major_brand = header[8:12]
        if major_brand in (b"M4A ", b"isom", b"mp42", b"dash", b"3gp5", b"MSNV"):
            return "m4a"

    return None


def validate_audio_file(
    file_path: str,
    max_size_bytes: int = 25 * 1024 * 1024,  # 25 MB max
    min_duration: float = 0.5,               # 0.5s min
    max_duration: float = 600.0,             # 10 min max
    silence_threshold: float = 1e-4,         # absolute max amplitude below this = silent
    clipping_ratio_threshold: float = 0.05,  # >5% of samples at >= 0.99 = clipped
    allow_clipped_with_warning: bool = False
) -> AudioValidationResult:
    """
    Perform rigorous preflight validation on an audio file before running pipelines:
    1. File existence and non-zero size
    2. File size within upper security boundary (DoS protection)
    3. Binary magic bytes header check
    4. Successful audio decoding check
    5. Duration constraints check
    6. Total silence check (max(abs(y)) < threshold)
    7. Saturation / clipping check
    """
    path = Path(file_path)
    if not path.is_file():
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.FILE_NOT_FOUND,
            error_message=f"File not found: {file_path}"
        )

    size = path.stat().st_size
    if size == 0:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.EMPTY_FILE,
            file_size_bytes=0,
            error_message="Audio file is empty (0 bytes)"
        )

    if size > max_size_bytes:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.FILE_TOO_LARGE,
            file_size_bytes=size,
            error_message=f"File size {size} bytes exceeds maximum limit of {max_size_bytes} bytes"
        )

    # Magic bytes verification
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(64)
    except Exception as e:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.CORRUPT_FILE,
            file_size_bytes=size,
            error_message=f"Could not read file header: {e}"
        )

    detected_format = detect_audio_format_from_bytes(header_bytes)
    if detected_format is None:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.UNSUPPORTED_FORMAT,
            file_size_bytes=size,
            error_message="Unsupported or invalid audio file signature (magic bytes)"
        )

    # Audio decoding and signal metrics
    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
    except Exception as e:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.CORRUPT_FILE,
            detected_format=detected_format,
            file_size_bytes=size,
            error_message=f"Audio decoding failed: {e}"
        )

    if y is None or len(y) == 0:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.CORRUPT_FILE,
            detected_format=detected_format,
            file_size_bytes=size,
            duration_seconds=0.0,
            error_message="Decoded audio contains 0 samples"
        )

    duration = float(len(y)) / float(sr)
    max_amp = float(np.max(np.abs(y)))
    clipping_samples = np.sum(np.abs(y) >= 0.99)
    clipping_ratio = float(clipping_samples) / float(len(y))

    # Duration checks
    if duration < min_duration:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.TOO_SHORT,
            detected_format=detected_format,
            duration_seconds=round(duration, 3),
            sample_rate=sr,
            channels=1,
            max_amplitude=round(max_amp, 6),
            clipping_ratio=round(clipping_ratio, 6),
            file_size_bytes=size,
            error_message=f"Audio duration {duration:.2f}s is below minimum {min_duration}s"
        )

    if duration > max_duration:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.TOO_LONG,
            detected_format=detected_format,
            duration_seconds=round(duration, 3),
            sample_rate=sr,
            channels=1,
            max_amplitude=round(max_amp, 6),
            clipping_ratio=round(clipping_ratio, 6),
            file_size_bytes=size,
            error_message=f"Audio duration {duration:.2f}s exceeds maximum {max_duration}s"
        )

    # Total silence detection
    if max_amp < silence_threshold:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.SILENT,
            detected_format=detected_format,
            duration_seconds=round(duration, 3),
            sample_rate=sr,
            channels=1,
            max_amplitude=round(max_amp, 6),
            clipping_ratio=0.0,
            file_size_bytes=size,
            error_message=f"Total silence detected: peak amplitude {max_amp:.6f} < {silence_threshold}"
        )

    # Clipping detection
    if clipping_ratio > clipping_ratio_threshold and not allow_clipped_with_warning:
        return AudioValidationResult(
            is_valid=False,
            status=AudioValidationStatus.CLIPPED,
            detected_format=detected_format,
            duration_seconds=round(duration, 3),
            sample_rate=sr,
            channels=1,
            max_amplitude=round(max_amp, 6),
            clipping_ratio=round(clipping_ratio, 6),
            file_size_bytes=size,
            error_message=f"Severe clipping detected: {clipping_ratio*100:.1f}% samples >= 0.99"
        )

    return AudioValidationResult(
        is_valid=True,
        status=AudioValidationStatus.VALID,
        detected_format=detected_format,
        duration_seconds=round(duration, 3),
        sample_rate=sr,
        channels=1,
        max_amplitude=round(max_amp, 6),
        clipping_ratio=round(clipping_ratio, 6),
        file_size_bytes=size,
    )


@contextmanager
def safe_temp_audio_file(suffix: str = ".wav", prefix: str = "cognisafe_audio_") -> Generator[str, None, None]:
    """
    Context manager providing a safe, uniquely-named temporary audio file.
    Guarantees cleanup on block exit, mitigating leakages and race conditions.
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    try:
        yield temp_path
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {temp_path}: {e}")


def safe_convert_to_wav(input_path: str, output_path: str, target_sr: int = 16000, timeout: int = 60) -> bool:
    """
    Convert an arbitrary audio file to standard 16kHz mono WAV using ffmpeg.
    Uses argument list execution to prevent shell injection vulnerabilities.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i", str(input_path),
        "-ar", str(target_sr),
        "-ac", "1",
        "-f", "wav",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error(f"FFmpeg conversion failed: {result.stderr[-300:]}")
        return False
    return True
