from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Job(Base):
    __tablename__ = "jobs"

    id         = Column(String, primary_key=True, index=True) # UUID
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    status     = Column(String, nullable=False, default="queued") # queued, processing, done, failed
    stage      = Column(String, nullable=False, default="uploading") # uploading, transcribing, acoustic, nlp, risk, done, failed
    result     = Column(JSON, nullable=True)
    error      = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="jobs")
