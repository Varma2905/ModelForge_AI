import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bcrypt
import jwt

logger = logging.getLogger("regression_studio.auth")

JWT_ALGORITHM = "HS256"
DEFAULT_SECRET = "dev-insecure-secret-change-me-before-deploying"
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", DEFAULT_SECRET)

if JWT_SECRET_KEY == DEFAULT_SECRET:
    logger.warning(
        "JWT_SECRET_KEY is using the insecure default value. "
        "Set a real secret in backend/.env before deploying."
    )

ACCESS_TOKEN_EXPIRE_HOURS = 24
REMEMBER_ME_EXPIRE_DAYS = 30


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: str, email: str, remember_me: bool = False) -> str:
    now = datetime.now(timezone.utc)
    expires_delta = (
        timedelta(days=REMEMBER_ME_EXPIRE_DAYS)
        if remember_me
        else timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


class TokenError(Exception):
    """Raised when a JWT is missing, malformed, or expired."""


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise TokenError("Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise TokenError("Invalid authentication token.")
