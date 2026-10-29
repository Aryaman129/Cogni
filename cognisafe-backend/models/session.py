from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Session(Base):
    __tablename__ = "sessions"

    id                   = Column(Integer, primary_key=True, index=True)
    user_id              = Column(Integer, ForeignKey("users.id"), nullable=False)
    risk_tier            = Column(String, nullable=True) # Nullable for establishing_baseline / insufficient_data
    recorded_at          = Column(DateTime, default=datetime.utcnow)
    duration_seconds     = Column(Float, nullable=True)
    mode                 = Column(String, default="voice") # "voice" | "text"
    analysis_status      = Column(String, default="complete", nullable=True) # complete, insufficient_baseline, insufficient_data, analysis_failed

    # 14 Canonical Biomarkers (10 Acoustic, 4 Linguistic)
    speech_rate          = Column(Float, nullable=True)
    articulation_rate    = Column(Float, nullable=True)
    pause_frequency      = Column(Float, nullable=True)
    pause_duration_mean  = Column(Float, nullable=True) # Canonical name
    filled_pause_rate    = Column(Float, nullable=True) # Restored canonical biomarker
    pitch_mean           = Column(Float, nullable=True)
    pitch_range          = Column(Float, nullable=True)
    jitter               = Column(Float, nullable=True)
    shimmer              = Column(Float, nullable=True)
    hnr                  = Column(Float, nullable=True) # DB column hnr, aliased as HNR
    lexical_diversity    = Column(Float, nullable=True)
    semantic_coherence   = Column(Float, nullable=True)
    idea_density         = Column(Float, nullable=True)
    syntactic_complexity = Column(Float, nullable=True)

    # Anomaly & Metadata
    has_anomaly          = Column(Boolean, default=False)
    anomaly_flags        = Column(String, nullable=True)
    transcript           = Column(Text, nullable=True)
    biomarker_results    = Column(JSON, nullable=True)
    ai_insights          = Column(JSON, nullable=True)

    user = relationship("User", back_populates="sessions")

    # Compatibility aliases
    @property
    def HNR(self):
        return self.hnr

    @HNR.setter
    def HNR(self, value):
        self.hnr = value

    @property
    def pause_duration(self):
        return self.pause_duration_mean

    @pause_duration.setter
    def pause_duration(self, value):
        self.pause_duration_mean = value