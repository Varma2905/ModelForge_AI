import asyncio
import logging
import os
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.datasets.ingest import ingest_source
from app.datasets.providers import gdrive_sessions
from app.datasets.providers.google_drive_provider import GoogleDriveProvider
from app.datasets.providers.kaggle_provider import KaggleCredentialsError, KaggleProvider
from app.integrations import store as integrations_store
from app.utils.response import ok

logger = logging.getLogger("regression_studio.dataset_source_routes")

router = APIRouter(prefix="/datasets", tags=["Dataset Sources"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001")

_NOT_CONFIGURED = "isn't configured on this server."
_KAGGLE_NOT_CONNECTED = "Connect your Kaggle account first."


# --- Config visibility: booleans only, never secrets ---
@router.get("/sources/status")
async def dataset_sources_status(current_user: dict = Depends(get_current_user)):
    return ok({
        "google_drive_configured": GoogleDriveProvider().is_configured(),
    })


# --- Kaggle ---
# Kaggle has no third-party OAuth flow — every user connects their OWN
# personal API token (username + key, from kaggle.com/settings/api), which
# is encrypted and stored server-side against their platform user_id (see
# app/integrations/store.py). Never shared across users, never sent to the
# frontend after the initial connect call.
class KaggleConnectRequest(BaseModel):
    username: str
    key: str


@router.post("/kaggle/connect")
async def connect_kaggle(request: KaggleConnectRequest, current_user: dict = Depends(get_current_user)):
    username = request.username.strip()
    key = request.key.strip()
    if not username or not key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Both Kaggle username and API key are required."
        )

    provider = KaggleProvider()
    try:
        await asyncio.to_thread(provider.validate_credentials, username, key)
    except KaggleCredentialsError as e:
        # Status varies by category so the frontend (and anyone reading network
        # logs) can tell "your key is wrong" apart from "Kaggle is having
        # issues" apart from "our request was malformed" — never blanket
        # everything as one generic message. NONE of these are 401: that
        # status is reserved for "the platform JWT itself is invalid" —
        # api-service.ts's shared unwrap() calls clearToken() and bounces to
        # /login on ANY 401 response, regardless of which endpoint sent it.
        # Using 401 here (this endpoint's original bug) made every failed
        # "Connect Kaggle" attempt log the user out of the whole app.
        category_status = {
            "invalid_credentials": status.HTTP_400_BAD_REQUEST,
            "forbidden": status.HTTP_400_BAD_REQUEST,
            "bad_request": status.HTTP_502_BAD_GATEWAY,
            "rate_limited": status.HTTP_429_TOO_MANY_REQUESTS,
            "kaggle_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
            "unknown": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }
        raise HTTPException(
            status_code=category_status.get(e.category, status.HTTP_400_BAD_REQUEST), detail=str(e)
        )

    await integrations_store.save_connection(
        user_id=current_user["_id"],
        provider="kaggle",
        credentials={"username": username, "key": key},
        provider_user_id=username,
    )
    return ok({"connected": True, "kaggle_username": username}, message="Kaggle account connected.")


@router.get("/kaggle/status")
async def kaggle_status(current_user: dict = Depends(get_current_user)):
    connection = await integrations_store.get_connection(current_user["_id"], "kaggle")
    if not connection:
        return ok({"connected": False, "kaggle_username": None})
    return ok({"connected": True, "kaggle_username": connection.get("provider_user_id")})


@router.post("/kaggle/disconnect")
async def disconnect_kaggle(current_user: dict = Depends(get_current_user)):
    await integrations_store.delete_connection(current_user["_id"], "kaggle")
    return ok({"connected": False}, message="Kaggle account disconnected.")


async def _get_kaggle_credentials(user_id: str) -> Dict[str, str]:
    connection = await integrations_store.get_connection(user_id, "kaggle")
    if not connection:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_KAGGLE_NOT_CONNECTED)
    return connection["credentials"]


class KaggleResolveRequest(BaseModel):
    dataset_ref: str


@router.post("/kaggle/resolve")
async def resolve_kaggle_dataset(
    request: KaggleResolveRequest, current_user: dict = Depends(get_current_user)
):
    creds = await _get_kaggle_credentials(current_user["_id"])
    provider = KaggleProvider()

    try:
        dataset_ref = provider.parse_ref(request.dataset_ref)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        files = await asyncio.to_thread(provider.list_files, creds["username"], creds["key"], dataset_ref)
    except Exception as e:
        logger.warning(f"Kaggle list_files failed for '{dataset_ref}': {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach Kaggle for dataset '{dataset_ref}'. Check the reference and try again.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset '{dataset_ref}' has no CSV/Excel/JSON/Parquet files to import.",
        )

    return ok({"dataset_ref": dataset_ref, "files": files})


class KaggleImportRequest(BaseModel):
    dataset_ref: str
    file_name: str


@router.post("/kaggle/import")
async def import_kaggle_dataset(
    request: KaggleImportRequest, current_user: dict = Depends(get_current_user)
):
    creds = await _get_kaggle_credentials(current_user["_id"])
    provider = KaggleProvider()

    try:
        source = await asyncio.to_thread(
            provider.download_file, creds["username"], creds["key"], request.dataset_ref, request.file_name
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.warning(f"Kaggle download failed for '{request.dataset_ref}'/{request.file_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to download the selected Kaggle file.",
        )

    try:
        result = await ingest_source(source, current_user["_id"], name_override=request.file_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return ok(result, message="Dataset imported from Kaggle.")


# --- Google Drive ---
@router.get("/google-drive/auth-url")
async def google_drive_auth_url(current_user: dict = Depends(get_current_user)):
    provider = GoogleDriveProvider()
    if not provider.is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Google Drive import {_NOT_CONFIGURED}",
        )

    flow = provider.build_flow()
    state = gdrive_sessions.create_pending_flow(current_user["_id"], flow)
    auth_url, _ = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", state=state, prompt="consent"
    )
    return ok({"auth_url": auth_url})


@router.get("/google-drive/callback")
async def google_drive_callback(code: str, state: str):
    # Deliberately NOT behind get_current_user — this is a top-level browser
    # redirect from Google carrying no Authorization header. user_id is
    # recovered from the pending-flow entry stashed under `state`.
    pending = gdrive_sessions.pop_pending_flow(state)
    if not pending:
        return RedirectResponse(f"{FRONTEND_URL}/new/upload?gdrive_error=expired_or_invalid")

    try:
        pending["flow"].fetch_token(code=code)
    except Exception as e:
        logger.warning(f"Google Drive OAuth token exchange failed: {e}")
        return RedirectResponse(f"{FRONTEND_URL}/new/upload?gdrive_error=auth_failed")

    session_token = gdrive_sessions.create_session(pending["user_id"], pending["flow"].credentials)
    return RedirectResponse(f"{FRONTEND_URL}/new/upload?gdrive_session={session_token}")


@router.get("/google-drive/files")
async def list_google_drive_files(session: str, current_user: dict = Depends(get_current_user)):
    entry = gdrive_sessions.get_session(session, current_user["_id"])
    if not entry:
        raise HTTPException(
            # NOT 401 — see the matching comment on the Kaggle connect route
            # above. A dead Google Drive session is a third-party connection
            # problem for an already-authenticated platform user, not a
            # platform-session problem; 401 here was clearing the user's
            # platform JWT and bouncing them to /login on an expired Drive
            # session.
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Drive session expired. Please reconnect.",
        )

    provider = GoogleDriveProvider()
    try:
        files = await asyncio.to_thread(provider.list_files, entry["credentials"])
    except Exception as e:
        logger.warning(f"Google Drive list_files failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Could not list Google Drive files."
        )

    return ok({"files": files})


class GoogleDriveImportRequest(BaseModel):
    session: str
    file_id: str
    file_name: str


@router.post("/google-drive/import")
async def import_google_drive_file(
    request: GoogleDriveImportRequest, current_user: dict = Depends(get_current_user)
):
    entry = gdrive_sessions.get_session(request.session, current_user["_id"])
    if not entry:
        raise HTTPException(
            # NOT 401 — see the matching comment on the Kaggle connect route
            # above. A dead Google Drive session is a third-party connection
            # problem for an already-authenticated platform user, not a
            # platform-session problem; 401 here was clearing the user's
            # platform JWT and bouncing them to /login on an expired Drive
            # session.
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Drive session expired. Please reconnect.",
        )

    provider = GoogleDriveProvider()
    try:
        source = await asyncio.to_thread(
            provider.download_file, entry["credentials"], request.file_id, request.file_name
        )
    except Exception as e:
        logger.warning(f"Google Drive download failed for file {request.file_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Failed to download the selected Google Drive file."
        )

    try:
        result = await ingest_source(source, current_user["_id"], name_override=request.file_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return ok(result, message="Dataset imported from Google Drive.")
