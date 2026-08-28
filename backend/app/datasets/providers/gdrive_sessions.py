import logging
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger("regression_studio.datasets.gdrive_sessions")

# In-memory ONLY — never written to disk or a database. This is what lets
# Google Drive OAuth work without a durable secret store: access/refresh
# tokens live only for the lifetime of this process, are keyed behind our
# own opaque tokens, and are never sent to the frontend.
_PENDING_FLOWS: Dict[str, Dict[str, Any]] = {}   # state -> {user_id, flow, created_at}
_ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}  # session -> {user_id, credentials, expires_at}

_STATE_TTL_SEC = 600      # time allowed to complete the Google consent redirect
_SESSION_TTL_SEC = 1800   # time allowed to browse/import files after connecting


def _purge_expired() -> None:
    now = time.time()
    expired_states = [s for s, v in _PENDING_FLOWS.items() if now - v["created_at"] > _STATE_TTL_SEC]
    for s in expired_states:
        del _PENDING_FLOWS[s]
    expired_sessions = [s for s, v in _ACTIVE_SESSIONS.items() if now > v["expires_at"]]
    for s in expired_sessions:
        del _ACTIVE_SESSIONS[s]


def create_pending_flow(user_id: str, flow: Any) -> str:
    _purge_expired()
    state = uuid.uuid4().hex
    _PENDING_FLOWS[state] = {"user_id": user_id, "flow": flow, "created_at": time.time()}
    return state


def pop_pending_flow(state: str) -> Optional[Dict[str, Any]]:
    """Single-use: removes the entry so a state token can't be replayed."""
    _purge_expired()
    return _PENDING_FLOWS.pop(state, None)


def create_session(user_id: str, credentials: Any) -> str:
    _purge_expired()
    session_token = uuid.uuid4().hex
    _ACTIVE_SESSIONS[session_token] = {
        "user_id": user_id,
        "credentials": credentials,
        "expires_at": time.time() + _SESSION_TTL_SEC,
    }
    return session_token


def get_session(session_token: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Returns None (and evicts) if expired, or if the session belongs to a
    different user — prevents session-token guessing from letting one user
    read another's Drive files."""
    _purge_expired()
    entry = _ACTIVE_SESSIONS.get(session_token)
    if not entry or entry["user_id"] != user_id:
        return None
    return entry
