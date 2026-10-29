"""
Biomarker Schemas and Typed Feature Contracts for CogniSafe Pipeline.
Preserves the 14 canonical biomarkers and enforces explicit validity/error typing.
"""

from enum import Enum
import math
from typing import Optional, Dict, Any, List, Iterator, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class FeatureStatus(str, Enum):
    """Explicit status enum for all biomarker extractions."""
    VALID = "valid"
    INSUFFICIENT_INPUT = "insufficient_input"
    EXTRACTION_FAILED = "extraction_failed"
    UNSUPPORTED_INPUT = "unsupported_input"
    MISSING_DEPENDENCY = "missing_dependency"
    INVALID_MEASUREMENT = "invalid_measurement"


CANONICAL_ACOUSTIC_BIOMARKERS: List[str] = [
    "speech_rate",            # words/min (total duration)
    "articulation_rate",      # words/min (speech-only duration via VAD)
    "pause_frequency",        # pauses/min (internal gaps >= 200ms)
    "pause_duration_mean",    # seconds (mean over valid pauses, None if 0 observed)
    "filled_pause_rate",      # disfluencies/min (token-aware regex matching)
    "pitch_mean",             # Hz (voiced frames mean)
    "pitch_range",            # Hz (robust range, IQR or 5th-95th percentile)
    "jitter",                 # ratio (frame F0 hop perturbation proxy)
    "shimmer",                # ratio (voiced frame RMS perturbation proxy)
    "HNR",                    # dB (spectral HPSS proxy on voiced segments)
]

CANONICAL_NLP_BIOMARKERS: List[str] = [
    "lexical_diversity",      # MTLD (standard bidirectional, all alpha tokens)
    "semantic_coherence",     # cosine similarity (consecutive + global sentence embeddings)
    "idea_density",           # ratio (proposition POS heuristic excluding auxiliaries)
    "syntactic_complexity",   # depth (iterative parse tree depth with cycle guard)
]

CANONICAL_BIOMARKERS: List[str] = CANONICAL_ACOUSTIC_BIOMARKERS + CANONICAL_NLP_BIOMARKERS


class TypedFeatureResult(BaseModel):
    """
    Standard typed contract for every biomarker measurement.
    Eliminates silent 0.0 fallbacks: invalid, unvoiced, or failed measurements
    must have status != VALID and value is None.
    """
    status: FeatureStatus
    value: Optional[float] = None
    unit: str
    is_proxy: bool = False
    method: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if math.isnan(v) or math.isinf(v):
                return None
        return v

    @model_validator(mode="after")
    def enforce_status_value_consistency(self) -> "TypedFeatureResult":
        # If status is not VALID, value must be None to prevent silent falsification
        if self.status != FeatureStatus.VALID and self.value is not None:
            self.value = None
        # If status is VALID but value is None (and reason is not explicitly intentional), flag as invalid
        if self.status == FeatureStatus.VALID and self.value is None:
            if not self.reason:
                self.status = FeatureStatus.INVALID_MEASUREMENT
                self.reason = "null_value_marked_as_valid"
        return self

    def __getitem__(self, item: str) -> Any:
        """Allow subscript access for backward compatibility with dict-based consumers."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        """Allow dict-like get access."""
        if hasattr(self, item):
            val = getattr(self, item)
            return val if val is not None else default
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard dictionary format."""
        return self.model_dump()


class BiomarkerCollectionResult(BaseModel):
    """
    Container for multiple extracted biomarkers (e.g. 10 acoustic, 4 NLP, or all 14).
    Supports dict-like subscripting and iteration while offering typing and summary stats.
    """
    features: Dict[str, TypedFeatureResult] = Field(default_factory=dict)
    overall_status: str = "complete"  # "complete", "insufficient_data", "partial", "failed"
    metadata: Optional[Dict[str, Any]] = None

    def __getitem__(self, key: str) -> TypedFeatureResult:
        return self.features[key]

    def __setitem__(self, key: str, value: TypedFeatureResult):
        self.features[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.features

    def __iter__(self) -> Iterator[str]:
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def get(self, key: str, default: Any = None) -> Any:
        return self.features.get(key, default)

    def keys(self):
        return self.features.keys()

    def values(self):
        return self.features.values()

    def items(self):
        return self.features.items()

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.model_dump() for k, v in self.features.items()}

    def to_flat_dict(self) -> Dict[str, Optional[float]]:
        """Return simple {biomarker_name: numeric_value_or_None} map."""
        return {k: v.value for k, v in self.features.items()}

    @property
    def valid_count(self) -> int:
        return sum(1 for f in self.features.values() if f.status == FeatureStatus.VALID and f.value is not None)

    @property
    def insufficient_count(self) -> int:
        return sum(1 for f in self.features.values() if f.status == FeatureStatus.INSUFFICIENT_INPUT)

    @property
    def failed_count(self) -> int:
        return sum(1 for f in self.features.values() if f.status in (
            FeatureStatus.EXTRACTION_FAILED,
            FeatureStatus.UNSUPPORTED_INPUT,
            FeatureStatus.MISSING_DEPENDENCY,
            FeatureStatus.INVALID_MEASUREMENT
        ))
