import secrets
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from database import get_db
from models import User, Session, CaregiverShare
from schemas import (
    CaregiverShareCreate,
    CaregiverShareResponse,
    CaregiverSessionSummary,
    CaregiverPublicView,
)
from dependencies import get_current_user

router = APIRouter(prefix="/api/caregiver", tags=["caregiver"])


@router.post("/shares", response_model=CaregiverShareResponse)
def create_share(
    req: CaregiverShareCreate,
    db: DBSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Creates a cryptographically secure, revocable sharing link for family/caregivers.
    Only the authenticated patient can generate shares for their own profile.
    """
    token = secrets.token_urlsafe(32)
    expires_at = None
    if req.expires_days and req.expires_days > 0:
        expires_at = datetime.utcnow() + timedelta(days=req.expires_days)

    share = CaregiverShare(
        patient_id=user.id,
        share_token=token,
        recipient_email=req.recipient_email,
        label=req.label,
        created_at=datetime.utcnow(),
        expires_at=expires_at,
        revoked=False,
        permissions="view_summary",
    )
    db.add(share)
    db.commit()
    db.refresh(share)

    validate_fn = getattr(CaregiverShareResponse, "model_validate", getattr(CaregiverShareResponse, "from_orm", None))
    resp = validate_fn(share)
    resp.share_url = f"/caregiver/view/{token}"
    return resp


@router.get("/shares", response_model=List[CaregiverShareResponse])
def list_shares(
    db: DBSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Lists all active (unrevoked) shares created by the authenticated patient."""
    shares = (
        db.query(CaregiverShare)
        .filter(CaregiverShare.patient_id == user.id, CaregiverShare.revoked == False)
        .order_by(CaregiverShare.created_at.desc())
        .all()
    )
    result = []
    validate_fn = getattr(CaregiverShareResponse, "model_validate", getattr(CaregiverShareResponse, "from_orm", None))
    for s in shares:
        item = validate_fn(s)
        item.share_url = f"/caregiver/view/{s.share_token}"
        result.append(item)
    return result


@router.delete("/shares/{share_id}")
def revoke_share(
    share_id: int,
    db: DBSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Revokes a caregiver share link.
    Strictly verifies ownership: cross-user revocation returns 403 Forbidden.
    """
    share = db.query(CaregiverShare).filter(CaregiverShare.id == share_id).first()
    if not share:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Caregiver share not found.",
        )

    if share.patient_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not own this caregiver share.",
        )

    share.revoked = True
    db.commit()
    return {"message": "Share link revoked successfully", "id": share_id}


@router.get("/view/{token}", response_model=CaregiverPublicView)
def public_caregiver_view(
    token: str,
    db: DBSession = Depends(get_db),
):
    """
    Public caregiver view endpoint.
    Validates token, expiration, and revocation status.
    Returns a sanitized monitoring summary without credentials, hashes, or emails.
    """
    share = (
        db.query(CaregiverShare)
        .filter(CaregiverShare.share_token == token)
        .first()
    )
    if not share or share.revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or revoked share link.",
        )

    if share.expires_at and datetime.utcnow() > share.expires_at:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Share link has expired.",
        )

    patient = db.query(User).filter(User.id == share.patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient record not found.",
        )

    all_sessions = (
        db.query(Session)
        .filter(Session.user_id == patient.id)
        .order_by(Session.recorded_at.desc())
        .all()
    )

    recent_sessions = all_sessions[:10]
    total_count = len(all_sessions)

    # Compute averages
    avg_coherence = None
    avg_speech = None
    avg_pause = None
    overall_risk = None

    if recent_sessions:
        coherences = [s.semantic_coherence for s in recent_sessions if s.semantic_coherence is not None]
        speeches = [s.speech_rate for s in recent_sessions if s.speech_rate is not None]
        pauses = [s.pause_frequency for s in recent_sessions if s.pause_frequency is not None]

        if coherences:
            avg_coherence = round(sum(coherences) / len(coherences), 2)
        if speeches:
            avg_speech = round(sum(speeches) / len(speeches), 1)
        if pauses:
            avg_pause = round(sum(pauses) / len(pauses), 2)

        # Mode risk tier
        risk_counts = {}
        for s in recent_sessions:
            if s.risk_tier:
                risk_counts[s.risk_tier] = risk_counts.get(s.risk_tier, 0) + 1
        if risk_counts:
            overall_risk = max(risk_counts, key=risk_counts.get)
        else:
            overall_risk = "Establishing Baseline"

    # Active status: whether session recorded in last 7 days
    now = datetime.utcnow()
    last_session_date = None
    active_status = "idle"

    if recent_sessions:
        last_s = recent_sessions[0]
        last_session_date = last_s.recorded_at.isoformat()
        if (now - last_s.recorded_at).days <= 7:
            active_status = "active"

    summaries = [
        CaregiverSessionSummary(
            session_id=s.id,
            date=s.recorded_at.isoformat(),
            risk_tier=s.risk_tier,
            speech_rate=s.speech_rate,
            semantic_coherence=s.semantic_coherence,
            pause_frequency=s.pause_frequency,
        )
        for s in recent_sessions
    ]

    return CaregiverPublicView(
        patient_name=patient.name,
        active_status=active_status,
        last_session_date=last_session_date,
        overall_risk_tier=overall_risk,
        total_sessions_monitored=total_count,
        average_coherence=avg_coherence,
        average_speech_rate=avg_speech,
        average_pause_frequency=avg_pause,
        recent_sessions=summaries,
    )
