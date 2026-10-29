"""
CogniSafe Acoustic Biomarkers Pipeline (10 Acoustic Biomarkers).
Adheres strictly to scientific definitions, eliminates silent 0.0 fallbacks,
and provides explicit typed feature contracts with transparent proxy labeling.
"""

import logging
import math
import re
from typing import Optional, Dict, Any, Union, List
import librosa
import numpy as np

from pipeline.schemas import (
    FeatureStatus,
    TypedFeatureResult,
    BiomarkerCollectionResult,
    CANONICAL_ACOUSTIC_BIOMARKERS,
)
from pipeline.audio_validator import validate_audio_file, AudioValidationStatus

logger = logging.getLogger("cognisafe.acoustic")


def extract_acoustic_features(
    audio_path: str,
    transcript_data: Optional[dict] = None,
    return_typed: bool = True,
) -> Union[BiomarkerCollectionResult, Dict[str, Any]]:
    """
    Extract the 10 canonical acoustic biomarkers from audio and transcript:
    1. speech_rate (words/min, total assessment duration)
    2. articulation_rate (words/min, speech-only duration via energy VAD)
    3. pause_frequency (internal gaps >= 200ms per minute)
    4. pause_duration_mean (mean duration of observed pauses; None if 0 observed)
    5. filled_pause_rate (token-aware regex disfluencies per minute)
    6. pitch_mean (mean F0 over voiced frames in Hz; None if unvoiced)
    7. pitch_range (robust IQR of F0 over voiced frames in Hz)
    8. jitter (frame F0 hop perturbation proxy)
    9. shimmer (voiced frame RMS perturbation proxy)
    10. HNR (spectral HPSS proxy evaluated on voiced speech segments)

    Eliminates all silent 0.0 fallbacks. Missing, unvoiced, or failed extractions
    consistently return value: None with explicit FeatureStatus.
    """
    # ── Step 0: Preflight Audio Validation ────────────────────────────────────
    val_res = validate_audio_file(audio_path, min_duration=0.5)
    if not val_res.is_valid:
        status = FeatureStatus.INSUFFICIENT_INPUT
        if val_res.status in (AudioValidationStatus.CORRUPT_FILE, AudioValidationStatus.UNSUPPORTED_FORMAT):
            status = FeatureStatus.EXTRACTION_FAILED
        elif val_res.status == AudioValidationStatus.FILE_NOT_FOUND:
            status = FeatureStatus.EXTRACTION_FAILED

        features = {
            name: TypedFeatureResult(
                status=status,
                value=None,
                unit=_get_unit(name),
                is_proxy=_is_proxy(name),
                method=_get_method(name),
                reason=val_res.error_message or val_res.status.value,
            )
            for name in CANONICAL_ACOUSTIC_BIOMARKERS
        }
        collection = BiomarkerCollectionResult(
            features=features,
            overall_status="insufficient_data" if status == FeatureStatus.INSUFFICIENT_INPUT else "failed",
            metadata={"validation_error": val_res.error_message, "validation_status": val_res.status.value}
        )
        return collection if return_typed else collection.to_dict()

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = float(len(y)) / float(sr)

        if duration < 0.5:
            return _insufficient_input_result(
                reason="duration_below_0.5s",
                return_typed=return_typed,
                metadata={"duration": duration}
            )

        # ── Step 1: Energy VAD and Pause Segmentation ─────────────────────────
        # librosa.effects.split uses top_db to isolate voiced/speech intervals
        intervals = librosa.effects.split(y, top_db=30)
        speech_samples = sum(end - start for start, end in intervals) if len(intervals) > 0 else 0
        speech_duration = float(speech_samples) / float(sr)

        # Internal pauses: gaps between speech intervals >= 200ms
        pauses = []
        for i in range(1, len(intervals)):
            gap = float(intervals[i][0] - intervals[i - 1][1]) / float(sr)
            if gap >= 0.200:  # 200 ms standard threshold
                pauses.append(gap)

        # ── Step 2: Pitch Tracking via pYIN ───────────────────────────────────
        # Note: C2 (~65.4 Hz) to C7 (~2093 Hz) covers full human phonation range
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
        )

        # Voiced frames require:
        # 1. voiced_flag is True
        # 2. voiced_probs >= 0.5 (excludes spurious tracking in white/ambient noise)
        # 3. Finite, positive F0 estimate
        if voiced_flag is not None and voiced_probs is not None:
            valid_voiced_mask = (voiced_flag == True) & (voiced_probs >= 0.5) & (~np.isnan(f0)) & (f0 > 0)
            voiced_f0 = f0[valid_voiced_mask]
        else:
            valid_voiced_mask = np.zeros(0, dtype=bool)
            voiced_f0 = np.array([])

        has_sufficient_voicing = len(voiced_f0) >= 5

        # ── Step 3: Compute Each Biomarker ────────────────────────────────────
        features: Dict[str, TypedFeatureResult] = {}

        # 1. pitch_mean
        if has_sufficient_voicing:
            pitch_mean_val = round(float(np.mean(voiced_f0)), 4)
            conf = float(np.mean(voiced_probs[valid_voiced_mask])) if voiced_probs is not None else 1.0
            features['pitch_mean'] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=pitch_mean_val,
                unit="Hz",
                is_proxy=False,
                method="pyin_voiced_f0_mean",
                confidence=round(conf, 4),
                metadata={"voiced_frames": int(len(voiced_f0))}
            )
        else:
            features['pitch_mean'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="Hz",
                is_proxy=False,
                method="pyin_voiced_f0_mean",
                reason="unvoiced_or_insufficient_voiced_frames",
                metadata={"voiced_frames": int(len(voiced_f0))}
            )

        # 2. pitch_range (Robust IQR over voiced frames)
        if has_sufficient_voicing:
            p25, p75 = np.percentile(voiced_f0, [25, 75])
            p5, p95 = np.percentile(voiced_f0, [5, 95])
            pitch_range_iqr = round(float(p75 - p25), 4)
            features['pitch_range'] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=pitch_range_iqr,
                unit="Hz",
                is_proxy=False,
                method="pyin_voiced_f0_iqr",
                metadata={
                    "iqr": pitch_range_iqr,
                    "p5_p95_range": round(float(p95 - p5), 4),
                    "p25": round(float(p25), 4),
                    "p75": round(float(p75), 4),
                }
            )
        else:
            features['pitch_range'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="Hz",
                is_proxy=False,
                method="pyin_voiced_f0_iqr",
                reason="unvoiced_or_insufficient_voiced_frames"
            )

        # 3. jitter (Frame F0 hop perturbation proxy)
        if has_sufficient_voicing:
            periods = 1.0 / (voiced_f0 + 1e-9)
            mean_period = float(np.mean(periods))
            if mean_period > 0:
                jitter_val = round(float(np.mean(np.abs(np.diff(periods))) / mean_period), 6)
                features['jitter'] = TypedFeatureResult(
                    status=FeatureStatus.VALID,
                    value=jitter_val,
                    unit="ratio",
                    is_proxy=True,
                    method="frame_f0_hop_perturbation_proxy",
                    metadata={"description": "Hop-level period perturbation proxy from pyin"}
                )
            else:
                features['jitter'] = TypedFeatureResult(
                    status=FeatureStatus.INVALID_MEASUREMENT,
                    value=None,
                    unit="ratio",
                    is_proxy=True,
                    method="frame_f0_hop_perturbation_proxy",
                    reason="mean_period_zero"
                )
        else:
            features['jitter'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="ratio",
                is_proxy=True,
                method="frame_f0_hop_perturbation_proxy",
                reason="unvoiced_or_insufficient_voiced_frames"
            )

        # 4. shimmer (Voiced frame RMS perturbation proxy)
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        min_len = min(len(rms), len(valid_voiced_mask))
        voiced_rms = rms[:min_len][valid_voiced_mask[:min_len]]

        if len(voiced_rms) >= 5:
            mean_rms = float(np.mean(voiced_rms))
            if mean_rms > 1e-6:
                shimmer_val = round(float(np.mean(np.abs(np.diff(voiced_rms))) / mean_rms), 6)
                features['shimmer'] = TypedFeatureResult(
                    status=FeatureStatus.VALID,
                    value=shimmer_val,
                    unit="ratio",
                    is_proxy=True,
                    method="voiced_frame_rms_perturbation_proxy",
                    metadata={"description": "Voiced-frame RMS amplitude perturbation proxy"}
                )
            else:
                features['shimmer'] = TypedFeatureResult(
                    status=FeatureStatus.INSUFFICIENT_INPUT,
                    value=None,
                    unit="ratio",
                    is_proxy=True,
                    method="voiced_frame_rms_perturbation_proxy",
                    reason="negligible_voiced_energy"
                )
        else:
            features['shimmer'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="ratio",
                is_proxy=True,
                method="voiced_frame_rms_perturbation_proxy",
                reason="unvoiced_or_insufficient_voiced_frames"
            )

        # 5. HNR (Spectral HPSS proxy evaluated on voiced speech segments)
        if len(intervals) > 0 and speech_duration >= 0.3:
            y_speech = np.concatenate([y[start:end] for start, end in intervals])
            harmonics = librosa.effects.harmonic(y_speech)
            noise = y_speech - harmonics
            h_power = float(np.sum(harmonics ** 2))
            n_power = float(np.sum(noise ** 2))
            if n_power < 1e-12:
                hnr_val = 50.0  # Cap at high harmonic ratio
            else:
                hnr_val = round(float(10.0 * np.log10((h_power + 1e-9) / (n_power + 1e-9))), 4)

            features['HNR'] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=hnr_val,
                unit="dB",
                is_proxy=True,
                method="spectral_hpss_voiced_hnr_proxy",
                metadata={"description": "Harmonic-percussive separation proxy on speech intervals"}
            )
        else:
            features['HNR'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="dB",
                is_proxy=True,
                method="spectral_hpss_voiced_hnr_proxy",
                reason="insufficient_speech_segments"
            )

        # ── Step 4: Transcript-Dependent Features ──────────────────────────────
        words = transcript_data.get('words', []) if transcript_data else []
        text = transcript_data.get('text', '') if transcript_data else ''

        if len(words) > 0:
            word_count = len(words)
        elif text.strip():
            word_count = len([w for w in text.strip().split() if any(c.isalnum() for c in w)])
        else:
            word_count = 0

        # 6. speech_rate (Words / total assessment duration)
        if word_count > 0 and duration > 0:
            speech_rate_val = round(float(word_count / duration) * 60.0, 4)
            features['speech_rate'] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=speech_rate_val,
                unit="words/min",
                is_proxy=False,
                method="whisper_word_count_total_duration",
                metadata={"word_count": word_count, "total_duration": round(duration, 3)}
            )
        else:
            features['speech_rate'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="words/min",
                is_proxy=False,
                method="whisper_word_count_total_duration",
                reason="empty_speech_no_words_detected",
                metadata={"word_count": word_count}
            )

        # 7. articulation_rate (Words / speech-only duration via VAD)
        if word_count > 0 and speech_duration >= 0.3:
            art_rate_val = round(float(word_count / speech_duration) * 60.0, 4)
            features['articulation_rate'] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=art_rate_val,
                unit="words/min",
                is_proxy=False,
                method="whisper_word_count_speech_vad_duration",
                metadata={"word_count": word_count, "speech_duration": round(speech_duration, 3)}
            )
        else:
            features['articulation_rate'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="words/min",
                is_proxy=False,
                method="whisper_word_count_speech_vad_duration",
                reason="insufficient_speech_duration_or_no_words",
                metadata={"word_count": word_count, "speech_duration": round(speech_duration, 3)}
            )

        # 8. pause_frequency (Internal pauses >= 200ms per minute)
        pause_freq_val = round(float(len(pauses) / duration) * 60.0, 4)
        features['pause_frequency'] = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=pause_freq_val,
            unit="pauses/min",
            is_proxy=False,
            method="energy_vad_internal_pauses_ge_200ms",
            metadata={"pause_count": len(pauses), "duration": round(duration, 3)}
        )

        # 9. pause_duration_mean (Mean duration of observed pauses; None if 0 observed)
        if len(pauses) > 0:
            mean_pause_dur = round(float(np.mean(pauses)), 4)
            features['pause_duration_mean'] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=mean_pause_dur,
                unit="seconds",
                is_proxy=False,
                method="mean_internal_pause_duration",
                metadata={"observed_pauses": len(pauses)}
            )
        else:
            features['pause_duration_mean'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="seconds",
                is_proxy=False,
                method="mean_internal_pause_duration",
                reason="no_pauses_observed",
                metadata={"observed_pauses": 0}
            )

        # 10. filled_pause_rate (Token-aware regex disfluency matching)
        if text.strip() and duration > 0:
            disfluency_matches = re.findall(r"\b(uh|um|er|ah)\b", text, flags=re.IGNORECASE)
            fp_count = len(disfluency_matches)
            fp_rate_val = round(float(fp_count / duration) * 60.0, 4)
            features['filled_pause_rate'] = TypedFeatureResult(
                status=FeatureStatus.VALID,
                value=fp_rate_val,
                unit="pauses/min",
                is_proxy=False,
                method="token_aware_regex_disfluency",
                metadata={"filled_pause_count": fp_count, "matches": disfluency_matches}
            )
        else:
            features['filled_pause_rate'] = TypedFeatureResult(
                status=FeatureStatus.INSUFFICIENT_INPUT,
                value=None,
                unit="pauses/min",
                is_proxy=False,
                method="token_aware_regex_disfluency",
                reason="empty_transcript_or_invalid_duration"
            )

        collection = BiomarkerCollectionResult(
            features=features,
            overall_status="complete",
            metadata={"duration": round(duration, 3), "speech_duration": round(speech_duration, 3)}
        )
        return collection if return_typed else collection.to_dict()

    except Exception as e:
        logger.error(f"[Acoustic ERROR] Exception during feature extraction: {e}", exc_info=True)
        return _extraction_failed_result(reason=str(e), return_typed=return_typed)


def _get_unit(name: str) -> str:
    units = {
        "speech_rate": "words/min",
        "articulation_rate": "words/min",
        "pause_frequency": "pauses/min",
        "pause_duration_mean": "seconds",
        "filled_pause_rate": "pauses/min",
        "pitch_mean": "Hz",
        "pitch_range": "Hz",
        "jitter": "ratio",
        "shimmer": "ratio",
        "HNR": "dB",
    }
    return units.get(name, "")


def _is_proxy(name: str) -> bool:
    return name in ("jitter", "shimmer", "HNR")


def _get_method(name: str) -> str:
    methods = {
        "speech_rate": "whisper_word_count_total_duration",
        "articulation_rate": "whisper_word_count_speech_vad_duration",
        "pause_frequency": "energy_vad_internal_pauses_ge_200ms",
        "pause_duration_mean": "mean_internal_pause_duration",
        "filled_pause_rate": "token_aware_regex_disfluency",
        "pitch_mean": "pyin_voiced_f0_mean",
        "pitch_range": "pyin_voiced_f0_iqr",
        "jitter": "frame_f0_hop_perturbation_proxy",
        "shimmer": "voiced_frame_rms_perturbation_proxy",
        "HNR": "spectral_hpss_voiced_hnr_proxy",
    }
    return methods.get(name, "unknown")


def _insufficient_input_result(reason: str, return_typed: bool, metadata: Optional[dict] = None):
    features = {
        name: TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit=_get_unit(name),
            is_proxy=_is_proxy(name),
            method=_get_method(name),
            reason=reason,
        )
        for name in CANONICAL_ACOUSTIC_BIOMARKERS
    }
    collection = BiomarkerCollectionResult(
        features=features,
        overall_status="insufficient_data",
        metadata=metadata
    )
    return collection if return_typed else collection.to_dict()


def _extraction_failed_result(reason: str, return_typed: bool):
    features = {
        name: TypedFeatureResult(
            status=FeatureStatus.EXTRACTION_FAILED,
            value=None,
            unit=_get_unit(name),
            is_proxy=_is_proxy(name),
            method=_get_method(name),
            reason=reason,
        )
        for name in CANONICAL_ACOUSTIC_BIOMARKERS
    }
    collection = BiomarkerCollectionResult(
        features=features,
        overall_status="failed",
        metadata={"error": reason}
    )
    return collection if return_typed else collection.to_dict()