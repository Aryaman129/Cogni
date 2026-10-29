import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
from sqlalchemy.orm import Session as DBSession

from database import get_db
from models import User, Session, CaregiverShare
from dependencies import get_current_user
from auth import decode_token
from services.nebius_client import get_nebius_client
from schemas.ai import (
    ExplainRequest,
    AnomalyExplanationResponse,
    WeeklySummaryRequest,
    WeeklySummaryResponse,
    CaregiverSummaryRequest,
    CaregiverSummaryResponse,
    CANONICAL_BIOMARKERS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])


def _extract_biomarkers_from_session(s: Session) -> Dict[str, Any]:
    """Helper to convert Session ORM model into verified ground-truth biomarker dictionary."""
    biomarkers = {
        "speech_rate": s.speech_rate,
        "articulation_rate": s.articulation_rate,
        "pause_frequency": s.pause_frequency,
        "pause_duration_mean": s.pause_duration_mean,
        "filled_pause_rate": s.filled_pause_rate,
        "pitch_mean": s.pitch_mean,
        "pitch_range": s.pitch_range,
        "jitter": s.jitter,
        "shimmer": s.shimmer,
        "HNR": s.hnr,
        "lexical_diversity": s.lexical_diversity,
        "semantic_coherence": s.semantic_coherence,
        "idea_density": s.idea_density,
        "syntactic_complexity": s.syntactic_complexity,
    }
    # If the session model stores a detailed biomarker_results JSON, merge it
    if s.biomarker_results and isinstance(s.biomarker_results, dict):
        for k, v in s.biomarker_results.items():
            if k not in biomarkers or biomarkers[k] is None:
                biomarkers[k] = v
    return biomarkers


# ------------------------------------------------------------------------------
# 1. POST /api/ai/explain
# ------------------------------------------------------------------------------
@router.post("/explain", response_model=AnomalyExplanationResponse)
async def explain_anomalies(
    req: ExplainRequest,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Generate contextual explanations for detected biomarker anomalies.
    Binds output strictly to registered ground-truth measurements.
    """
    client = get_nebius_client()
    biomarkers: Dict[str, Any] = {}
    risk_tier = req.risk_tier
    anomalies = req.anomalies or []
    baseline = req.baseline or {}

    # If session_id is provided, verify ownership and load registered ground-truth measurements
    if req.session_id is not None:
        session_obj = db.query(Session).filter(
            Session.id == req.session_id,
            Session.user_id == user.id,
        ).first()

        if not session_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {req.session_id} not found or does not belong to authenticated user.",
            )

        biomarkers = _extract_biomarkers_from_session(session_obj)
        if risk_tier is None:
            risk_tier = session_obj.risk_tier

        # Parse stored anomaly flags if not explicitly provided
        if not anomalies and session_obj.anomaly_flags:
            try:
                flags = json.loads(session_obj.anomaly_flags)
                if isinstance(flags, list):
                    anomalies = flags
                elif isinstance(flags, dict):
                    anomalies = list(flags.keys())
            except Exception:
                anomalies = [a.strip() for a in str(session_obj.anomaly_flags).split(",") if a.strip()]

    # If biomarkers are provided in request body, merge/override
    if req.biomarkers:
        biomarkers.update(req.biomarkers)

    if not biomarkers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No biomarker measurements provided or found in session.",
        )

    explanation_res = await client.explain_biomarkers(
        biomarkers=biomarkers,
        anomalies=anomalies,
        baseline=baseline,
        risk_tier=risk_tier,
        session_id=req.session_id,
    )

    # Persist AI insights into session record if session exists
    if req.session_id is not None:
        try:
            s_to_update = db.query(Session).filter(Session.id == req.session_id).first()
            if s_to_update:
                s_to_update.ai_insights = explanation_res.model_dump()
                db.commit()
        except Exception as e:
            logger.warning("Failed to persist ai_insights into session %s: %s", req.session_id, e)

    return explanation_res


# ------------------------------------------------------------------------------
# 2. POST /api/ai/weekly-summary
# ------------------------------------------------------------------------------
@router.post("/weekly-summary", response_model=WeeklySummaryResponse)
async def weekly_summary(
    req: WeeklySummaryRequest,
    user: User = Depends(get_current_user),
    db: DBSession = Depends(get_db),
):
    """
    Generate longitudinal trend summaries and weekly report narrative sections.
    """
    client = get_nebius_client()
    sessions_data = req.sessions_data

    # If sessions_data is not passed directly, retrieve from user's persisted sessions
    if not sessions_data:
        q = db.query(Session).filter(Session.user_id == user.id)

        if req.session_ids:
            q = q.filter(Session.id.in_(req.session_ids))

        if req.start_date:
            try:
                start_dt = datetime.fromisoformat(req.start_date.replace("Z", "+00:00"))
                q = q.filter(Session.recorded_at >= start_dt)
            except Exception:
                pass

        if req.end_date:
            try:
                end_dt = datetime.fromisoformat(req.end_date.replace("Z", "+00:00"))
                q = q.filter(Session.recorded_at <= end_dt)
            except Exception:
                pass

        db_sessions = q.order_by(Session.recorded_at.desc()).limit(14).all()

        sessions_data = []
        for s in db_sessions:
            s_dict = _extract_biomarkers_from_session(s)
            s_dict["id"] = s.id
            s_dict["risk_tier"] = s.risk_tier
            s_dict["recorded_at"] = s.recorded_at.isoformat() if s.recorded_at else None
            sessions_data.append(s_dict)

    summary_res = await client.generate_weekly_summary(
        sessions_data=sessions_data or [],
        start_date=req.start_date,
        end_date=req.end_date,
    )
    return summary_res


# ------------------------------------------------------------------------------
# 3. POST /api/ai/caregiver-summary
# ------------------------------------------------------------------------------
@router.post("/caregiver-summary", response_model=CaregiverSummaryResponse)
async def caregiver_summary(
    req: CaregiverSummaryRequest,
    db: DBSession = Depends(get_db),
    authorization: Optional[str] = Header(None),
):
    """
    Generate plain-language translation of clinical metrics for family members.
    Supports either JWT bearer auth or active CaregiverShare share token.
    """
    client = get_nebius_client()
    target_user_id: Optional[int] = None
    patient_name = req.patient_name or "your loved one"

    # Option A: Caregiver Share Token authorization
    if req.share_token:
        share = db.query(CaregiverShare).filter(
            CaregiverShare.share_token == req.share_token
        ).first()

        if not share:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Caregiver share link not found.",
            )
        if share.revoked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Caregiver share link has been revoked.",
            )
        if share.expires_at and datetime.utcnow() > share.expires_at:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Caregiver share link has expired.",
            )

        target_user_id = share.patient_id
        # Optionally personalize patient name from User record
        patient = db.query(User).filter(User.id == target_user_id).first()
        if patient and patient.name:
            patient_name = patient.name

    # Option B: Authenticated User authorization
    elif authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1].strip()
        payload = decode_token(token)
        if not payload or not payload.get("sub"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            target_user_id = int(payload.get("sub"))
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Malformed token subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        patient = db.query(User).filter(User.id == target_user_id).first()
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        if patient.name and req.patient_name == "your loved one":
            patient_name = patient.name
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required via Bearer JWT token or valid share_token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    biomarkers = req.biomarkers or {}
    risk_tier = req.risk_tier

    # If session_id is provided, verify session belongs to target patient
    if req.session_id is not None:
        session_obj = db.query(Session).filter(
            Session.id == req.session_id,
            Session.user_id == target_user_id,
        ).first()

        if not session_obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {req.session_id} not found for this patient.",
            )

        extracted = _extract_biomarkers_from_session(session_obj)
        extracted.update(biomarkers)
        biomarkers = extracted
        if risk_tier is None:
            risk_tier = session_obj.risk_tier

    caregiver_res = await client.generate_caregiver_summary(
        biomarkers=biomarkers,
        risk_tier=risk_tier,
        patient_name=patient_name,
    )
    return caregiver_res
