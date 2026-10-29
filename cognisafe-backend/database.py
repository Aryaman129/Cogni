from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cognisafe.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def run_migrations(target_engine=None):
    """
    Safely evolves existing database tables to match the updated schema
    without losing existing session or user records.
    """
    eng = target_engine or engine
    # First ensure all tables are created if not present
    Base.metadata.create_all(bind=eng)

    inspector = inspect(eng)
    existing_tables = inspector.get_table_names()

    if "sessions" in existing_tables:
        existing_cols = {col["name"] for col in inspector.get_columns("sessions")}
        with eng.begin() as conn:
            # 1. Add filled_pause_rate if missing
            if "filled_pause_rate" not in existing_cols:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN filled_pause_rate FLOAT"))
            
            # 2. Check pause_duration -> pause_duration_mean
            if "pause_duration_mean" not in existing_cols:
                if "pause_duration" in existing_cols:
                    try:
                        conn.execute(text("ALTER TABLE sessions RENAME COLUMN pause_duration TO pause_duration_mean"))
                    except Exception:
                        conn.execute(text("ALTER TABLE sessions ADD COLUMN pause_duration_mean FLOAT"))
                        conn.execute(text("UPDATE sessions SET pause_duration_mean = pause_duration WHERE pause_duration_mean IS NULL"))
                else:
                    conn.execute(text("ALTER TABLE sessions ADD COLUMN pause_duration_mean FLOAT"))

            # 3. Add analysis_status if missing
            if "analysis_status" not in existing_cols:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN analysis_status VARCHAR DEFAULT 'complete'"))

            # 4. Add mode if missing
            if "mode" not in existing_cols:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN mode VARCHAR DEFAULT 'voice'"))

            # 5. Add duration_seconds if missing
            if "duration_seconds" not in existing_cols:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN duration_seconds FLOAT"))

            # 6. Add biomarker_results if missing
            if "biomarker_results" not in existing_cols:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN biomarker_results JSON"))

            # 7. Add transcript if missing
            if "transcript" not in existing_cols:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN transcript TEXT"))

            # 8. Add ai_insights if missing
            if "ai_insights" not in existing_cols:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN ai_insights JSON"))