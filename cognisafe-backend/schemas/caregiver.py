from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

class CaregiverShareCreate(BaseModel):
    recipient_email: Optional[str] = None
    label:           Optional[str] = None
    expires_days:    Optional[int] = 30 # Default 30 days validity

class CaregiverShareResponse(BaseModel):
    id:              int
    patient_id:      int
    share_token:     str
    recipient_email: Optional[str] = None
    label:           Optional[str] = None
    created_at:      datetime
    expires_at:      Optional[datetime] = None
    revoked:         bool
    permissions:     str
    share_url:       Optional[str] = None

    class Config:
        from_attributes = True

class CaregiverSessionSummary(BaseModel):
    session_id:         int
    date:               str
    risk_tier:          Optional[str] = None
    speech_rate:        Optional[float] = None
    semantic_coherence: Optional[float] = None
    pause_frequency:    Optional[float] = None

class CaregiverPublicView(BaseModel):
    patient_name:            str
    active_status:           str # e.g. "active", "monitoring_current"
    last_session_date:       Optional[str] = None
    overall_risk_tier:       Optional[str] = None
    total_sessions_monitored: int
    average_coherence:       Optional[float] = None
    average_speech_rate:     Optional[float] = None
    average_pause_frequency: Optional[float] = None
    recent_sessions:         List[CaregiverSessionSummary]
