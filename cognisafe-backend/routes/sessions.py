from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession
from datetime import datetime, timedelta
from typing import List, Optional
from database import get_db
from models import User, Session
from schemas import SessionCreate, SessionResponse, HistoryItem, TodayResponse
from dependencies import get_current_user

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

def risk_to_status(risk: Optional[str]) -> str:
    if not risk:
        return "info"
    return {"Green": "good", "Yellow": "warn", "Orange": "warn", "Red": "bad"}.get(risk, "good")

@router.post("", response_model=SessionResponse)
def save_session(
    req:  SessionCreate,
    db:   DBSession = Depends(get_db),
    user: User      = Depends(get_current_user)
):
    data = req.dict(exclude_unset=True)

    # Normalize aliases for DB storage
    if "pause_duration" in data and "pause_duration_mean" not in data:
        data["pause_duration_mean"] = data.pop("pause_duration")
    elif "pause_duration" in data:
        data.pop("pause_duration")

    if "HNR" in data and "hnr" not in data:
        data["hnr"] = data.pop("HNR")
    elif "HNR" in data:
        data.pop("HNR")

    data.pop("emotional_entropy", None)

    session = Session(user_id=user.id, **data)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/today", response_model=TodayResponse)
def get_today(
    db:   DBSession = Depends(get_db),
    user: User      = Depends(get_current_user)
):
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    session = (
        db.query(Session)
        .filter(Session.user_id == user.id, Session.recorded_at >= today_start)
        .order_by(Session.recorded_at.desc())
        .first()
    )
    if not session:
        return TodayResponse(recorded=False)
    return TodayResponse(
        recorded=True,
        risk_tier=session.risk_tier,
        session_id=session.id
    )

@router.get("/latest", response_model=Optional[SessionResponse])
def get_latest(
    db:   DBSession = Depends(get_db),
    user: User      = Depends(get_current_user)
):
    session = (
        db.query(Session)
        .filter(Session.user_id == user.id)
        .order_by(Session.recorded_at.desc())
        .first()
    )
    return session

@router.get("/history", response_model=List[HistoryItem])
def get_history(
    months: int       = 1,
    db:     DBSession = Depends(get_db),
    user:   User      = Depends(get_current_user)
):
    since = datetime.utcnow() - timedelta(days=30 * months)
    sessions = (
        db.query(Session)
        .filter(Session.user_id == user.id, Session.recorded_at >= since)
        .order_by(Session.recorded_at)
        .all()
    )
    return [
        HistoryItem(
            date=s.recorded_at.isoformat(),
            status=risk_to_status(s.risk_tier),
            risk_tier=s.risk_tier,
            session_id=s.id,
            semantic_coherence=s.semantic_coherence,
            speech_rate=s.speech_rate,
            pause_frequency=s.pause_frequency,
        )
        for s in sessions
    ]

@router.get("/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: int,
    db:         DBSession = Depends(get_db),
    user:       User      = Depends(get_current_user)
):
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to view this session"
        )
    return session
