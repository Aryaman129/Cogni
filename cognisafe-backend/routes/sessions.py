from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from datetime import datetime, timedelta
from typing import List
from database import get_db
from models import User, Session
from schemas import SessionCreate, SessionResponse, HistoryItem, TodayResponse
from dependencies import get_current_user

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

def risk_to_status(risk: str) -> str:
    return {"Green": "good", "Yellow": "warn", "Red": "bad"}.get(risk, "good")

@router.post("", response_model=SessionResponse)
def save_session(
    req:  SessionCreate,
    db:   DBSession = Depends(get_db),
    user: User      = Depends(get_current_user)
):
    session = Session(user_id=user.id, **req.dict())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/today", response_model=TodayResponse)
def get_today(
    db:   DBSession = Depends(get_db),
    user: User      = Depends(get_current_user)
):
    return TodayResponse(recorded=False)
    return TodayResponse(
        recorded=True,
        risk_tier=session.risk_tier,
        session_id=session.id
    )

@router.get("/latest", response_model=SessionResponse)
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
    if not session:
        raise HTTPException(status_code=404, detail="No sessions found")
    return session

@router.get("/history", response_model=List[HistoryItem])
def get_history(
    months: int      = 1,
    db:     DBSession = Depends(get_db),
    user:   User     = Depends(get_current_user)
):
    since    = datetime.utcnow() - timedelta(days=30 * months)
    sessions = (
        db.query(Session)
        .filter(Session.user_id == user.id, Session.recorded_at >= since)
        .order_by(Session.recorded_at)
        .all()
    )
    return [
        HistoryItem(
            date=s.recorded_at.strftime("%Y-%m-%d"),
            status=risk_to_status(s.risk_tier),
            risk_tier=s.risk_tier,
            session_id=s.id,
        )
        for s in sessions
    ]