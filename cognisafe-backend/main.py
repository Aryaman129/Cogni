import os
from dotenv import load_dotenv

# Load local environment configuration (.env)
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base, run_migrations
from models import User, Session, Job, CaregiverShare  # noqa
from routes import (
    auth_router,
    sessions_router,
    users_router,
    reports_router,
    chat_router,
    caregiver_router,
)
from routes.ml import ml_router

# Initialize database schema and run column migrations
run_migrations(engine)

app = FastAPI(title="CogniSafe API", version="1.0.0")

# Restrict CORS origins via ALLOWED_ORIGINS environment variable
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

@app.on_event("startup")
def init_sandbox_mode():
    """
    Isolate demo fixtures cleanly to non-production sandbox mode.
    Never seeds hardcoded demo users in production.
    """
    enable_sandbox = os.getenv("ENABLE_SANDBOX_DEMO", "0").lower() in ("1", "true", "yes")
    is_prod = os.getenv("ENVIRONMENT", "").lower() in ("production", "prod")

    if enable_sandbox and not is_prod:
        from database import SessionLocal
        from auth import hash_password
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.email == "sandbox.demo@cognisafe.app").first()
            if not existing:
                demo_user = User(
                    name="Sandbox Demo User",
                    email="sandbox.demo@cognisafe.app",
                    password_hash=hash_password("sandbox1234"),
                    dob="1970-01-01",
                )
                db.add(demo_user)
                db.commit()
        finally:
            db.close()

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(users_router)
app.include_router(reports_router)
app.include_router(caregiver_router)
app.include_router(ml_router)
app.include_router(chat_router)

@app.get("/")
def root():
    return {"status": "CogniSafe API running", "version": "1.0.0"}