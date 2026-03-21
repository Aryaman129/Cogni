from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from models import User, Session  # noqa
from routes import auth_router, sessions_router, users_router, reports_router

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CogniSafe API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(users_router)
app.include_router(reports_router)

@app.get("/")
def root():
    return {"status": "CogniSafe API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}