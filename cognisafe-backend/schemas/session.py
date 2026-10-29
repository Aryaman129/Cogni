from pydantic import BaseModel, Field, root_validator
from typing import Optional, List, Dict, Any
from datetime import datetime

class SessionCreate(BaseModel):
    risk_tier:            Optional[str]   = None
    analysis_status:      Optional[str]   = "complete"
    mode:                 Optional[str]   = "voice"
    duration_seconds:     Optional[float] = None

    # 14 Canonical Biomarkers
    speech_rate:          Optional[float] = None
    articulation_rate:    Optional[float] = None
    pause_frequency:      Optional[float] = None
    pause_duration_mean:  Optional[float] = None
    pause_duration:       Optional[float] = None # Legacy alias
    filled_pause_rate:    Optional[float] = None
    pitch_mean:           Optional[float] = None
    pitch_range:          Optional[float] = None
    jitter:               Optional[float] = None
    shimmer:              Optional[float] = None
    hnr:                  Optional[float] = None
    HNR:                  Optional[float] = None # Uppercase alias
    lexical_diversity:    Optional[float] = None
    semantic_coherence:   Optional[float] = None
    idea_density:         Optional[float] = None
    syntactic_complexity: Optional[float] = None

    # Anomaly & Metadata
    has_anomaly:          Optional[bool]  = False
    anomaly_flags:        Optional[str]   = None
    transcript:           Optional[str]   = None
    biomarker_results:    Optional[Dict[str, Any]] = None
    ai_insights:          Optional[Dict[str, Any]] = None

    @root_validator(pre=True)
    def normalize_aliases(cls, values):
        if not isinstance(values, dict):
            return values
        # Handle pause_duration -> pause_duration_mean
        if values.get("pause_duration_mean") is None and values.get("pause_duration") is not None:
            values["pause_duration_mean"] = values.get("pause_duration")
        elif values.get("pause_duration") is None and values.get("pause_duration_mean") is not None:
            values["pause_duration"] = values.get("pause_duration_mean")

        # Handle HNR -> hnr
        if values.get("hnr") is None and values.get("HNR") is not None:
            values["hnr"] = values.get("HNR")
        elif values.get("HNR") is None and values.get("hnr") is not None:
            values["HNR"] = values.get("hnr")

        # Discard phantom column emotional_entropy if present
        values.pop("emotional_entropy", None)
        return values


class SessionResponse(BaseModel):
    id:                   int
    risk_tier:            Optional[str]   = None
    analysis_status:      Optional[str]   = "complete"
    mode:                 Optional[str]   = "voice"
    duration_seconds:     Optional[float] = None
    recorded_at:          datetime

    # 14 Canonical Biomarkers
    speech_rate:          Optional[float] = None
    articulation_rate:    Optional[float] = None
    pause_frequency:      Optional[float] = None
    pause_duration_mean:  Optional[float] = None
    pause_duration:       Optional[float] = None
    filled_pause_rate:    Optional[float] = None
    pitch_mean:           Optional[float] = None
    pitch_range:          Optional[float] = None
    jitter:               Optional[float] = None
    shimmer:              Optional[float] = None
    hnr:                  Optional[float] = None
    HNR:                  Optional[float] = None
    lexical_diversity:    Optional[float] = None
    semantic_coherence:   Optional[float] = None
    idea_density:         Optional[float] = None
    syntactic_complexity: Optional[float] = None

    # Anomaly & Metadata
    has_anomaly:          Optional[bool]  = None
    anomaly_flags:        Optional[str]   = None
    transcript:           Optional[str]   = None
    biomarker_results:    Optional[Dict[str, Any]] = None
    ai_insights:          Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

    @root_validator(pre=True)
    def populate_response_aliases(cls, values):
        # Support both ORM object and dict
        if hasattr(values, "pause_duration_mean"):
            p_mean = getattr(values, "pause_duration_mean", None)
            h_val = getattr(values, "hnr", None)
            # if converting from ORM, attributes will be accessed
            return values
        if isinstance(values, dict):
            if values.get("pause_duration_mean") is not None and values.get("pause_duration") is None:
                values["pause_duration"] = values.get("pause_duration_mean")
            if values.get("hnr") is not None and values.get("HNR") is None:
                values["HNR"] = values.get("hnr")
        return values


class HistoryItem(BaseModel):
    date:               str
    status:             str
    risk_tier:          Optional[str]   = None
    session_id:         int
    semantic_coherence: Optional[float] = None
    speech_rate:        Optional[float] = None
    pause_frequency:    Optional[float] = None


class TodayResponse(BaseModel):
    recorded:   bool
    risk_tier:  Optional[str] = None
    session_id: Optional[int] = None