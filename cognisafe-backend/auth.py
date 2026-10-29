from datetime import datetime, timedelta
from typing import Optional
import os
import secrets
import warnings
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

# Require explicit SECRET_KEY; generate cryptographically secure random secret if unset and warn in production
_env_secret = os.getenv("SECRET_KEY")
_environment = os.getenv("ENVIRONMENT", os.getenv("ENV", "development")).lower()

if not _env_secret:
    if _environment in ("production", "prod"):
        warnings.warn(
            "CRITICAL SECURITY WARNING: SECRET_KEY environment variable is unset in production! "
            "Generating temporary random ephemeral key. All sessions will invalidate on restart.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Generate cryptographically secure random secret for local development/testing
    SECRET_KEY = secrets.token_hex(32)
else:
    SECRET_KEY = _env_secret

ALGORITHM  = os.getenv("ALGORITHM", "HS256")
EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=EXPIRE_MIN)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
