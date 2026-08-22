import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

import jwt

from app.auth.security import JWT_ALGORITHM, JWT_SECRET_KEY

END_USER_TOKEN_EXPIRE_HOURS = 24
END_USER_TOKEN_TYPE = "baas_end_user"


def generate_project_id() -> str:
    return "proj_" + secrets.token_urlsafe(16)


def generate_key_pair() -> Tuple[str, str]:
    """Returns (public_key, secret_key). Uses `secrets` (cryptographically
    secure) rather than the str(uuid.uuid4())[:8] pattern used elsewhere in
    this codebase for non-secret internal IDs — that pattern is fine for an
    internal ID, wrong for an API secret."""
    return "pk_live_" + secrets.token_urlsafe(24), "sk_live_" + secrets.token_urlsafe(32)


class EndUserTokenError(Exception):
    """Raised when a BaaS end-user JWT is missing, malformed, expired, or —
    critically — a Studio session token presented where an end-user token is
    expected (or vice versa)."""


def create_end_user_token(user_id: str, project_id: str, email: str) -> str:
    """Deliberately NOT a call to auth/security.py's create_access_token —
    the extra project_id + typ claims are what make a Studio session JWT and
    a BaaS end-user JWT mutually non-replayable against each other, even
    though both are signed with the same JWT_SECRET_KEY."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "project_id": project_id,
        "typ": END_USER_TOKEN_TYPE,
        "iat": now,
        "exp": now + timedelta(hours=END_USER_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_end_user_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise EndUserTokenError("Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise EndUserTokenError("Invalid authentication token.")

    if payload.get("typ") != END_USER_TOKEN_TYPE:
        raise EndUserTokenError("Invalid authentication token.")
    return payload
