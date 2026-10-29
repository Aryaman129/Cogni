from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class CaregiverShare(Base):
    __tablename__ = "caregiver_shares"

    id              = Column(Integer, primary_key=True, index=True)
    patient_id      = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    share_token     = Column(String(128), unique=True, index=True, nullable=False)
    recipient_email = Column(String(255), nullable=True)
    label           = Column(String(255), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at      = Column(DateTime, nullable=True)
    revoked         = Column(Boolean, default=False, nullable=False)
    permissions     = Column(String(64), default="view_summary", nullable=False)

    patient = relationship("User", back_populates="caregiver_shares")
