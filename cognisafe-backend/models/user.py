from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String, nullable=False)
    email         = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    dob           = Column(String, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    sessions         = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    jobs             = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    caregiver_shares = relationship("CaregiverShare", back_populates="patient", cascade="all, delete-orphan")