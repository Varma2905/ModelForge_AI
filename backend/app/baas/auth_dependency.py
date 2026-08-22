from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.security import verify_password
from app.baas.security import EndUserTokenError, decode_end_user_token
from app.database.mongodb import db_client

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_project_by_public_key(
    x_public_key: Optional[str] = Header(None, alias="X-Public-Key")
) -> dict:
    """Resolves a project from the public key alone — no secret needed. This
    is what makes reads and end-user signup/login usable directly from a
    third-party browser, matching the master spec's 'public key = client-
    safe' framing.

    The header is Optional (not FastAPI's `Header(...)` required form) so a
    missing key is rejected by *this function* as a normal HTTPException —
    consistent with the rest of the app's {success,error} envelope — rather
    than by FastAPI's request-validation layer, which returns its own
    unenveloped {"detail": [...]} shape before this function ever runs."""
    if not x_public_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key.")
    key_doc = await db_client.find_one("baas_api_keys", {"public_key": x_public_key})
    if not key_doc or key_doc.get("disabled"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or disabled API key.")
    project = await db_client.find_one("baas_projects", {"_id": key_doc["project_id"]})
    if not project:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or disabled API key.")
    return {"project_id": key_doc["project_id"], "key_id": key_doc["_id"]}


async def get_current_project(
    x_public_key: Optional[str] = Header(None, alias="X-Public-Key"),
    x_secret_key: Optional[str] = Header(None, alias="X-Secret-Key"),
) -> dict:
    """Requires both keys — gates writes. Never exposed to a browser per the
    master spec ('secret key MUST NEVER be exposed to frontend JavaScript');
    meant to be called from the consuming app's own backend. See the
    docstring on get_current_project_by_public_key for why these headers are
    Optional rather than FastAPI-required."""
    if not x_public_key or not x_secret_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API credentials.")
    key_doc = await db_client.find_one("baas_api_keys", {"public_key": x_public_key})
    if (
        not key_doc
        or key_doc.get("disabled")
        or not verify_password(x_secret_key, key_doc["secret_hash"])
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")
    project = await db_client.find_one("baas_projects", {"_id": key_doc["project_id"]})
    if not project:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials.")
    return {"project_id": key_doc["project_id"], "key_id": key_doc["_id"]}


async def get_current_end_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """Structurally parallel to app/auth/dependencies.py::get_current_user."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    try:
        payload = decode_end_user_token(credentials.credentials)
    except EndUserTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    end_user = await db_client.find_one(
        "baas_end_users", {"_id": payload.get("sub"), "project_id": payload.get("project_id")}
    )
    if not end_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account no longer exists.")

    return end_user
