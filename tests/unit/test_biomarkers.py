"""
Comprehensive Unit Tests for CogniSafe Biomarkers and Audio Validation Pipeline.
Verifies all 14 biomarkers (10 acoustic + 4 NLP), typed contracts, audio preflight,
silence/clipping/corrupt handling, and absence of silent 0.0 fallbacks.
"""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
import pytest
import soundfile as sf

# Ensure cognisafe-deploy is in Python path for test discovery
deploy_path = str(Path(__file__).resolve().parent.parent.parent / "cognisafe-deploy")
if deploy_path not in sys.path:
    sys.path.insert(0, deploy_path)

from pipeline.schemas import (
    FeatureStatus,
    TypedFeatureResult,
    BiomarkerCollectionResult,
    CANONICAL_ACOUSTIC_BIOMARKERS,
    CANONICAL_NLP_BIOMARKERS,
    CANONICAL_BIOMARKERS,
)
from pipeline.audio_validator import (
    AudioValidationStatus,
    AudioValidationResult,
    detect_audio_format_from_bytes,
    validate_audio_file,
    safe_temp_audio_file,
)
from pipeline.acoustic import extract_acoustic_features
from pipeline.nlp import (
    analyze_text,
    compute_mtld,
    compute_semantic_coherence,
    compute_idea_density,
    compute_syntactic_complexity,
    get_spacy_model,
)
from pipeline.transcription import transcribe_audio, _fallback_transcript


# ══════════════════════════════════════════════════════════════════════════════
# 1. Schemas & Typed Feature Result Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemasAndContracts:
    """Verify TypedFeatureResult, FeatureStatus enum, and BiomarkerCollectionResult."""

    def test_canonical_biomarker_counts(self):
        assert len(CANONICAL_ACOUSTIC_BIOMARKERS) == 10
        assert len(CANONICAL_NLP_BIOMARKERS) == 4
        assert len(CANONICAL_BIOMARKERS) == 14
        assert "speech_rate" in CANONICAL_ACOUSTIC_BIOMARKERS
        assert "pause_duration_mean" in CANONICAL_ACOUSTIC_BIOMARKERS
        assert "filled_pause_rate" in CANONICAL_ACOUSTIC_BIOMARKERS
        assert "lexical_diversity" in CANONICAL_NLP_BIOMARKERS
        assert "syntactic_complexity" in CANONICAL_NLP_BIOMARKERS

    def test_feature_status_enum_values(self):
        expected_statuses = {
            "valid",
            "insufficient_input",
            "extraction_failed",
            "unsupported_input",
            "missing_dependency",
            "invalid_measurement",
        }
        actual_statuses = {status.value for status in FeatureStatus}
        assert expected_statuses == actual_statuses

    def test_typed_feature_result_valid(self):
        res = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=135.5,
            unit="words/min",
            is_proxy=False,
            method="test_method",
            confidence=0.95,
        )
        assert res.status == FeatureStatus.VALID
        assert res.value == 135.5
        assert res["value"] == 135.5
        assert res.get("unit") == "words/min"
        assert res.is_proxy is False
        dump = res.to_dict()
        assert dump["value"] == 135.5
        assert dump["status"] == "valid"

    def test_typed_feature_result_enforces_none_on_insufficient_or_failed(self):
        # Even if a caller attempts to supply a numeric value with INSUFFICIENT_INPUT,
        # the model validator must force value to None to prevent silent falsification
        res = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=0.0,
            unit="Hz",
            is_proxy=False,
            method="test_method",
            reason="unvoiced",
        )
        assert res.value is None
        assert res.status == FeatureStatus.INSUFFICIENT_INPUT

    def test_biomarker_collection_result_operations(self):
        feat1 = TypedFeatureResult(
            status=FeatureStatus.VALID,
            value=120.0,
            unit="words/min",
            is_proxy=False,
            method="method_1",
        )
        feat2 = TypedFeatureResult(
            status=FeatureStatus.INSUFFICIENT_INPUT,
            value=None,
            unit="seconds",
            is_proxy=False,
            method="method_2",
            reason="no_pauses",
        )
        collection = BiomarkerCollectionResult(
            features={"speech_rate": feat1, "pause_duration_mean": feat2},
            overall_status="partial",
        )
        assert len(collection) == 2
        assert "speech_rate" in collection
        assert collection["speech_rate"].value == 120.0
        assert collection["pause_duration_mean"].value is None
        assert collection.valid_count == 1
        assert collection.insufficient_count == 1
        assert collection.failed_count == 0

        flat = collection.to_flat_dict()
        assert flat == {"speech_rate": 120.0, "pause_duration_mean": None}


# ══════════════════════════════════════════════════════════════════════════════
# 2. Audio Validator Tests (Preflight, Magic Bytes, Silence, Clipping)
# ══════════════════════════════════════════════════════════════════════════════

class TestAudioValidator:
    """Verify audio preflight: magic bytes, duration, silence, clipping, and temp files."""

    def test_magic_bytes_detection_valid_wav(self):
        # Standard RIFF/WAVE header
        header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
        assert detect_audio_format_from_bytes(header) == "wav"

    def test_magic_bytes_detection_flac_ogg_mp3_webm(self):
        assert detect_audio_format_from_bytes(b"fLaC\x00\x00\x00\x22") == "flac"
        assert detect_audio_format_from_bytes(b"OggS\x00\x02\x00\x00") == "ogg"
        assert detect_audio_format_from_bytes(b"ID3\x03\x00\x00\x00") == "mp3"
        assert detect_audio_format_from_bytes(b"\xff\xfb\x90\x64\x00") == "mp3"
        assert detect_audio_format_from_bytes(b"\x1a\x45\xdf\xa3\x93") == "webm"

    def test_magic_bytes_rejects_corrupted_or_text_file(self):
        assert detect_audio_format_from_bytes(b"This is a text file") is None
        assert detect_audio_format_from_bytes(b"\x00\x00\x00\x00") is None
        assert detect_audio_format_from_bytes(b"") is None

    def test_validate_nonexistent_file(self):
        res = validate_audio_file("nonexistent_audio_path_12345.wav")
        assert res.is_valid is False
        assert res.status == AudioValidationStatus.FILE_NOT_FOUND

    def test_validate_empty_file(self):
        with safe_temp_audio_file(suffix=".wav") as temp_p:
            res = validate_audio_file(temp_p)
            assert res.is_valid is False
            assert res.status == AudioValidationStatus.EMPTY_FILE

    def test_validate_total_silence_audio(self):
        sr = 16000
        silent = np.zeros(sr * 2, dtype=np.float32)
        with safe_temp_audio_file(suffix=".wav") as temp_p:
            sf.write(temp_p, silent, sr)
            res = validate_audio_file(temp_p)
            assert res.is_valid is False
            assert res.status == AudioValidationStatus.SILENT
            assert res.max_amplitude == 0.0

    def test_validate_severe_clipping_audio(self):
        sr = 16000
        # Create heavily saturated signal (square wave at 1.0)
        clipped = np.ones(sr * 2, dtype=np.float32)
        with safe_temp_audio_file(suffix=".wav") as temp_p:
            sf.write(temp_p, clipped, sr)
            res = validate_audio_file(temp_p)
            assert res.is_valid is False
            assert res.status == AudioValidationStatus.CLIPPED
            assert res.clipping_ratio > 0.05

    def test_validate_too_short_audio(self):
        sr = 16000
        short_sig = 0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 0.2, int(sr * 0.2)))
        with safe_temp_audio_file(suffix=".wav") as temp_p:
            sf.write(temp_p, short_sig.astype(np.float32), sr)
            res = validate_audio_file(temp_p, min_duration=0.5)
            assert res.is_valid is False
            assert res.status == AudioValidationStatus.TOO_SHORT

    def test_validate_valid_audio(self):
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 220 * t)
        with safe_temp_audio_file(suffix=".wav") as temp_p:
            sf.write(temp_p, sig.astype(np.float32), sr)
            res = validate_audio_file(temp_p)
            assert res.is_valid is True
            assert res.status == AudioValidationStatus.VALID
            assert res.duration_seconds == pytest.approx(2.0, abs=0.05)
            assert res.sample_rate == 16000


# ══════════════════════════════════════════════════════════════════════════════
# 3. Acoustic Biomarkers Tests (10 Biomarkers)
# ══════════════════════════════════════════════════════════════════════════════

class TestAcousticBiomarkers:
    """Verify all 10 acoustic biomarkers on valid, silent, unvoiced, and edge-case audio."""

    @pytest.fixture
    def synthetic_voiced_audio(self):
        """Generate 3s synthetic voiced speech with fundamental F0 ~ 150 Hz and a pause."""
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        f0 = 150.0
        # Voiced signal with fundamental and first two harmonics
        sig = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)
        # Introduce a clean 400ms pause in the middle
        sig[int(sr * 1.2):int(sr * 1.6)] = 0.0

        with safe_temp_audio_file(suffix=".wav") as path:
            sf.write(path, sig.astype(np.float32), sr)
            yield path

    def test_all_10_acoustic_biomarkers_valid_speech(self, synthetic_voiced_audio):
        transcript = {
            "text": "The patient describes the picture clearly and concisely.",
            "words": [
                {"word": "The", "start": 0.1, "end": 0.3},
                {"word": "patient", "start": 0.35, "end": 0.7},
                {"word": "describes", "start": 0.75, "end": 1.15},
                {"word": "the", "start": 1.7, "end": 1.9},
                {"word": "picture", "start": 1.95, "end": 2.3},
                {"word": "clearly", "start": 2.35, "end": 2.65},
                {"word": "and", "start": 2.68, "end": 2.78},
                {"word": "concisely", "start": 2.8, "end": 2.95},
            ]
        }
        res = extract_acoustic_features(synthetic_voiced_audio, transcript_data=transcript)
        assert res.overall_status == "complete"
        assert len(res) == 10

        # Check all 10 canonical features are present
        for name in CANONICAL_ACOUSTIC_BIOMARKERS:
            assert name in res
            feat = res[name]
            assert isinstance(feat, TypedFeatureResult)

        # 1. pitch_mean: fundamental frequency near 150 Hz
        assert res["pitch_mean"].status == FeatureStatus.VALID
        assert res["pitch_mean"].value == pytest.approx(150.0, abs=15.0)

        # 2. pitch_range: robust IQR in Hz
        assert res["pitch_range"].status == FeatureStatus.VALID
        assert res["pitch_range"].value is not None
        assert res["pitch_range"].value >= 0.0

        # 3. jitter: hop perturbation proxy (labeled transparently)
        assert res["jitter"].status == FeatureStatus.VALID
        assert res["jitter"].is_proxy is True
        assert res["jitter"].value is not None

        # 4. shimmer: voiced frame RMS proxy (labeled transparently)
        assert res["shimmer"].status == FeatureStatus.VALID
        assert res["shimmer"].is_proxy is True
        assert res["shimmer"].value is not None

        # 5. HNR: spectral HPSS proxy (labeled transparently)
        assert res["HNR"].status == FeatureStatus.VALID
        assert res["HNR"].is_proxy is True
        assert res["HNR"].value > 0.0

        # 6. speech_rate: 8 words / 3s * 60 = 160 wpm
        assert res["speech_rate"].status == FeatureStatus.VALID
        assert res["speech_rate"].value == pytest.approx(160.0, abs=10.0)

        # 7. articulation_rate: words / speech duration
        assert res["articulation_rate"].status == FeatureStatus.VALID
        assert res["articulation_rate"].value >= res["speech_rate"].value

        # 8. pause_frequency: internal pauses per min
        assert res["pause_frequency"].status == FeatureStatus.VALID
        assert res["pause_frequency"].value > 0.0

        # 9. pause_duration_mean: valid pause observed
        assert res["pause_duration_mean"].status == FeatureStatus.VALID
        assert res["pause_duration_mean"].value == pytest.approx(0.4, abs=0.15)

        # 10. filled_pause_rate: 0 disfluencies in this transcript
        assert res["filled_pause_rate"].status == FeatureStatus.VALID
        assert res["filled_pause_rate"].value == 0.0

    def test_silent_audio_returns_none_for_all_biomarkers(self):
        sr = 16000
        silent = np.zeros(sr * 2, dtype=np.float32)
        with safe_temp_audio_file(suffix=".wav") as path:
            sf.write(path, silent, sr)
            res = extract_acoustic_features(path)

        assert res.overall_status == "insufficient_data"
        for name in CANONICAL_ACOUSTIC_BIOMARKERS:
            feat = res[name]
            assert feat.status == FeatureStatus.INSUFFICIENT_INPUT
            assert feat.value is None, f"Biomarker {name} must be None on silent audio, got {feat.value}"

    def test_unvoiced_pure_noise_pitch_biomarkers(self):
        """Random Gaussian noise has no phonation; pitch, jitter, and shimmer must be None."""
        sr = 16000
        noise = np.random.normal(0, 0.05, sr * 2).astype(np.float32)
        with safe_temp_audio_file(suffix=".wav") as path:
            sf.write(path, noise, sr)
            res = extract_acoustic_features(path)

        # Unvoiced frames should result in INSUFFICIENT_INPUT and None value for pitch
        assert res["pitch_mean"].status == FeatureStatus.INSUFFICIENT_INPUT
        assert res["pitch_mean"].value is None
        assert res["pitch_range"].status == FeatureStatus.INSUFFICIENT_INPUT
        assert res["pitch_range"].value is None
        assert res["jitter"].status == FeatureStatus.INSUFFICIENT_INPUT
        assert res["jitter"].value is None

    def test_zero_pauses_pause_duration_mean_is_none_not_zero(self):
        """When 0 pauses are observed, pause_duration_mean must return None (not 0.0)."""
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 150.0 * t)  # continuous 2s tone, NO pause
        transcript = {"text": "continuous speech", "words": [{"word": "word", "start": 0.1, "end": 1.9}]}
        with safe_temp_audio_file(suffix=".wav") as path:
            sf.write(path, sig.astype(np.float32), sr)
            res = extract_acoustic_features(path, transcript_data=transcript)

        assert res["pause_frequency"].status == FeatureStatus.VALID
        assert res["pause_frequency"].value == 0.0
        assert res["pause_duration_mean"].status == FeatureStatus.INSUFFICIENT_INPUT
        assert res["pause_duration_mean"].value is None

    def test_empty_speech_speech_rate_is_none(self):
        """When transcript is empty or no words detected, speech_rate must return None."""
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 150.0 * t)
        with safe_temp_audio_file(suffix=".wav") as path:
            sf.write(path, sig.astype(np.float32), sr)
            res = extract_acoustic_features(path, transcript_data={"text": "", "words": []})

        assert res["speech_rate"].status == FeatureStatus.INSUFFICIENT_INPUT
        assert res["speech_rate"].value is None
        assert res["articulation_rate"].status == FeatureStatus.INSUFFICIENT_INPUT
        assert res["articulation_rate"].value is None

    def test_token_aware_filled_pause_rate(self):
        """Verify token-aware regex disfluency matching without false substring hits."""
        sr = 16000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 150.0 * t)
        # "Um", "uh", and "er" are fillers. "umbrella", "other", and "herb" are NOT fillers!
        transcript = {
            "text": "Um, the umbrella was over there, uh, with another herb, er, right?",
            "words": [{"word": "word", "start": 0.1, "end": 1.9}]
        }
        with safe_temp_audio_file(suffix=".wav") as path:
            sf.write(path, sig.astype(np.float32), sr)
            res = extract_acoustic_features(path, transcript_data=transcript)

        feat = res["filled_pause_rate"]
        assert feat.status == FeatureStatus.VALID
        # Exactly 3 matches: "Um", "uh", "er" (2.0s duration -> 3 / 2.0 * 60 = 90.0 per min)
        assert feat.metadata["filled_pause_count"] == 3
        assert feat.value == pytest.approx(90.0, abs=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. NLP Biomarkers Tests (4 Linguistic Biomarkers)
# ══════════════════════════════════════════════════════════════════════════════

class TestNLPBiomarkers:
    """Verify all 4 NLP biomarkers on standard, single-sentence, and short texts."""

    @pytest.fixture(scope="class")
    def nlp_doc_fixture(self):
        nlp = get_spacy_model()
        text = (
            "The old clock in the tower strikes midnight with a deep resonant sound. "
            "A gentle breeze moves through the dark cobblestone streets. "
            "The nocturnal animals begin their watchful journey under the quiet moonlight."
        )
        return nlp(text), text

    def test_all_4_nlp_biomarkers_valid_text(self, nlp_doc_fixture):
        doc, text = nlp_doc_fixture
        res = analyze_text({"text": text})
        assert res.overall_status == "complete"
        assert len(res) == 4

        for name in CANONICAL_NLP_BIOMARKERS:
            assert name in res
            feat = res[name]
            assert feat.status == FeatureStatus.VALID
            assert feat.value is not None

        # 1. lexical_diversity (MTLD): standard score for diverse paragraph is > 20.0
        assert res["lexical_diversity"].value > 20.0
        assert res["lexical_diversity"].unit == "MTLD"

        # 2. semantic_coherence: cosine similarity between 0.0 and 1.0
        assert 0.0 <= res["semantic_coherence"].value <= 1.0
        assert res["semantic_coherence"].unit == "cosine_similarity"

        # 3. idea_density: proposition ratio between 0.0 and 1.0
        assert 0.2 <= res["idea_density"].value <= 0.8
        assert res["idea_density"].unit == "ratio"
        assert res["idea_density"].is_proxy is True

        # 4. syntactic_complexity: average dependency parse depth >= 2
        assert res["syntactic_complexity"].value >= 2.0
        assert res["syntactic_complexity"].unit == "depth"

    def test_semantic_coherence_single_sentence_returns_none(self):
        """Single sentence cannot establish inter-sentence coherence; must return None (not 1.0)."""
        res = analyze_text("This is an isolated single sentence with no second sentence.")
        feat = res["semantic_coherence"]
        assert feat.status == FeatureStatus.INSUFFICIENT_INPUT
        assert feat.value is None
        assert "fewer_than_2_sentences" in feat.reason

    def test_lexical_diversity_insufficient_tokens_returns_none(self):
        """MTLD requires at least 20 tokens; short inputs must return None."""
        short_text = "The dog barked at the cat."
        res = analyze_text(short_text)
        feat = res["lexical_diversity"]
        assert feat.status == FeatureStatus.INSUFFICIENT_INPUT
        assert feat.value is None
        assert "insufficient_tokens" in feat.reason

    def test_idea_density_excludes_auxiliary_verbs(self):
        """Auxiliary verbs ('is', 'has', 'was') should not count as propositions."""
        nlp = get_spacy_model()
        doc = nlp("The musician was playing a violin while people were dancing happily.")
        feat = compute_idea_density(doc)
        assert feat.status == FeatureStatus.VALID
        # "playing", "dancing", "musician", "violin", "happily" are propositions; "was" and "were" are AUX
        propositions = feat.metadata["sample_propositions"]
        assert "was" not in propositions
        assert "were" not in propositions

    def test_syntactic_complexity_depth_guard_and_cycle_safety(self):
        """Verify recursion safety and depth cap on highly repetitive deep chains."""
        nlp = get_spacy_model()
        # Repetitive conjunction chain
        deep_sentence = "I know that you think that he said that she thought that we knew that they saw it."
        doc = nlp(deep_sentence)
        feat = compute_syntactic_complexity(doc, max_depth_guard=20)
        assert feat.status == FeatureStatus.VALID
        assert feat.value <= 20.0

    def test_empty_or_whitespace_text_handling(self):
        for empty_val in ("", "   ", None, {}):
            res = analyze_text(empty_val)
            assert res.overall_status == "insufficient_data"
            for name in CANONICAL_NLP_BIOMARKERS:
                assert res[name].status == FeatureStatus.INSUFFICIENT_INPUT
                assert res[name].value is None


# ══════════════════════════════════════════════════════════════════════════════
# 5. Transcription Pipeline Tests (Error Classification & Metadata)
# ══════════════════════════════════════════════════════════════════════════════

class TestTranscriptionPipeline:
    """Verify transcription error classification and model metadata."""

    def test_transcribe_nonexistent_file(self):
        res = transcribe_audio("completely_missing_audio_path_xyz.wav")
        assert res["status"] == "file_not_found"
        assert res["word_count"] == 0
        assert res["text"] == ""
        assert "whisper-base" in res["model_metadata"]["model_name"]
        assert res["model_metadata"]["language"] == "en"

    def test_fallback_transcript_schema(self):
        fallback = _fallback_transcript(
            status="ffmpeg_missing",
            error_message="FFmpeg not found in PATH"
        )
        assert fallback["status"] == "ffmpeg_missing"
        assert fallback["error_message"] == "FFmpeg not found in PATH"
        assert fallback["words"] == []
        assert fallback["pause_events"] == []
        assert fallback["word_count"] == 0
