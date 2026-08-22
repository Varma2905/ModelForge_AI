import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from app.auth.dependencies import get_current_user
from app.auth.security import hash_password, verify_password
from app.baas.auth_dependency import (
    get_current_end_user,
    get_current_project,
    get_current_project_by_public_key,
)
from app.baas.schema_builder import SchemaProposalError, propose_schema
from app.baas.security import create_end_user_token, generate_key_pair, generate_project_id
from app.database.mongodb import db_client
from app.services.query_service import validate_identifier
from app.utils.response import ok

logger = logging.getLogger("regression_studio.baas_routes")

# ── Rate limiting ────────────────────────────────────────────────────────────
# /data/* and /end-users/* are genuinely public-facing (gated only by a key
# that's expected to sit in client-side JS for the read/auth surface) — same
# in-memory sliding-window pattern already used in ai_routes.py, keyed by
# project_id instead of user_id.
_RATE_LIMIT_WINDOW_SEC = 60
_RATE_LIMIT_MAX_REQUESTS = 60
_rate_limit_state: Dict[str, deque] = defaultdict(deque)


def _check_rate_limit(project_id: str) -> None:
    now = time.time()
    q = _rate_limit_state[project_id]
    while q and now - q[0] > _RATE_LIMIT_WINDOW_SEC:
        q.popleft()
    if len(q) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please slow down.",
        )
    q.append(now)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Request/response models ──────────────────────────────────────────────────
class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class CreateKeyRequest(BaseModel):
    label: Optional[str] = Field(None, max_length=100)


class ProposeSchemaRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000)


class ColumnDef(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    type: Literal["string", "number", "boolean", "date"]


class CreateTableRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    columns: List[ColumnDef] = Field(..., min_length=1)


class EndUserSignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class EndUserLoginRequest(BaseModel):
    email: EmailStr
    password: str


def _to_project_response(p: dict) -> dict:
    return {"id": p["_id"], "name": p["name"], "created_at": p["created_at"]}


def _to_key_response(k: dict) -> dict:
    return {
        "id": k["_id"],
        "public_key": k["public_key"],
        "label": k.get("label"),
        "disabled": k.get("disabled", False),
        "created_at": k["created_at"],
    }


def _to_table_response(t: dict) -> dict:
    return {"id": t["_id"], "name": t["name"], "columns": t["columns"], "created_at": t["created_at"]}


def _to_record_response(doc: dict) -> dict:
    return {
        "id": doc["_id"],
        **doc.get("data", {}),
        "created_at": doc.get("created_at"),
        "updated_at": doc.get("updated_at"),
    }


def _to_end_user_response(u: dict) -> dict:
    return {"id": u["_id"], "email": u["email"], "created_at": u["created_at"]}


async def _get_owned_project(project_id: str, user_id: str) -> dict:
    project = await db_client.find_one("baas_projects", {"_id": project_id})
    if not project or project.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Project with ID {project_id} not found."
        )
    return project


async def _cascade_delete_project(project_id: str) -> None:
    """A BaaS project's tables/records/keys/end-users have no other consumer
    to preserve (unlike data_sources.py's deliberate non-cascade on delete,
    where an imported dataset has independent analysis-history value)."""
    for collection in ("baas_tables", "baas_records", "baas_api_keys", "baas_end_users"):
        docs = await db_client.find_many(collection, {"project_id": project_id})
        for doc in docs:
            await db_client.delete_one(collection, {"_id": doc["_id"]})
    await db_client.delete_one("baas_projects", {"_id": project_id})


# ══════════════════════════════════════════════════════════════════════════
# Studio-JWT-gated project management — registered directly on the main app,
# keeps the existing strict CORS policy.
# ══════════════════════════════════════════════════════════════════════════
baas_management_router = APIRouter(prefix="/baas", tags=["Backend as a Service — Management"])


@baas_management_router.post("/projects")
async def create_project(request: CreateProjectRequest, current_user: dict = Depends(get_current_user)):
    now = _now()
    project_doc = {
        "_id": generate_project_id(),
        "user_id": current_user["_id"],
        "name": request.name.strip(),
        "created_at": now,
    }
    await db_client.insert_one("baas_projects", project_doc)

    public_key, secret_key = generate_key_pair()
    key_doc = {
        "_id": str(ObjectId()),
        "project_id": project_doc["_id"],
        "public_key": public_key,
        "secret_hash": hash_password(secret_key),
        "label": "Default",
        "disabled": False,
        "created_at": now,
    }
    await db_client.insert_one("baas_api_keys", key_doc)

    return ok(
        {"project": _to_project_response(project_doc), "public_key": public_key, "secret_key": secret_key},
        message="Project created. Save the secret key now — it will not be shown again.",
    )


@baas_management_router.get("/projects")
async def list_projects(current_user: dict = Depends(get_current_user)):
    projects = await db_client.find_many("baas_projects", {"user_id": current_user["_id"]})
    sorted_projects = sorted(projects, key=lambda p: p.get("created_at", ""), reverse=True)
    return ok([_to_project_response(p) for p in sorted_projects])


@baas_management_router.get("/projects/{project_id}")
async def get_project(project_id: str, current_user: dict = Depends(get_current_user)):
    project = await _get_owned_project(project_id, current_user["_id"])
    return ok(_to_project_response(project))


@baas_management_router.delete("/projects/{project_id}")
async def delete_project(project_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_project(project_id, current_user["_id"])
    await _cascade_delete_project(project_id)
    return ok({"deleted": True}, message="Project and all its data deleted.")


@baas_management_router.post("/projects/{project_id}/keys")
async def create_key(
    project_id: str, request: CreateKeyRequest, current_user: dict = Depends(get_current_user)
):
    await _get_owned_project(project_id, current_user["_id"])
    public_key, secret_key = generate_key_pair()
    key_doc = {
        "_id": str(ObjectId()),
        "project_id": project_id,
        "public_key": public_key,
        "secret_hash": hash_password(secret_key),
        "label": request.label or "Untitled key",
        "disabled": False,
        "created_at": _now(),
    }
    await db_client.insert_one("baas_api_keys", key_doc)
    return ok(
        {"key": _to_key_response(key_doc), "secret_key": secret_key},
        message="Key created. Save the secret key now — it will not be shown again.",
    )


@baas_management_router.get("/projects/{project_id}/keys")
async def list_keys(project_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_project(project_id, current_user["_id"])
    keys = await db_client.find_many("baas_api_keys", {"project_id": project_id})
    sorted_keys = sorted(keys, key=lambda k: k.get("created_at", ""), reverse=True)
    return ok([_to_key_response(k) for k in sorted_keys])


@baas_management_router.post("/projects/{project_id}/keys/{key_id}/revoke")
async def revoke_key(project_id: str, key_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_project(project_id, current_user["_id"])
    key_doc = await db_client.find_one("baas_api_keys", {"_id": key_id})
    if not key_doc or key_doc.get("project_id") != project_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Key with ID {key_id} not found.")
    await db_client.update_one("baas_api_keys", {"_id": key_id}, {"$set": {"disabled": True}})
    return ok({"revoked": True})


@baas_management_router.post("/projects/{project_id}/tables/propose-schema")
async def propose_table_schema(
    project_id: str, request: ProposeSchemaRequest, current_user: dict = Depends(get_current_user)
):
    await _get_owned_project(project_id, current_user["_id"])
    try:
        columns = await propose_schema(request.description)
    except SchemaProposalError as e:
        status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE if e.category == "not_configured" else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=status_code, detail=str(e))
    return ok({"columns": columns})


@baas_management_router.post("/projects/{project_id}/tables")
async def create_table(
    project_id: str, request: CreateTableRequest, current_user: dict = Depends(get_current_user)
):
    await _get_owned_project(project_id, current_user["_id"])
    table_name = validate_identifier(request.name, "table name").lower()

    existing = await db_client.find_many("baas_tables", {"project_id": project_id})
    if any(t["name"] == table_name for t in existing):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"A table named '{table_name}' already exists in this project."
        )

    seen = set()
    columns = []
    for col in request.columns:
        col_name = validate_identifier(col.name, "column name").lower()
        if col_name in seen or col_name == "id":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Duplicate or reserved column name: '{col_name}'."
            )
        seen.add(col_name)
        columns.append({"name": col_name, "type": col.type})

    table_doc = {
        "_id": str(ObjectId()),
        "project_id": project_id,
        "name": table_name,
        "columns": columns,
        "created_at": _now(),
    }
    await db_client.insert_one("baas_tables", table_doc)
    return ok(_to_table_response(table_doc), message="Table created.")


@baas_management_router.get("/projects/{project_id}/tables")
async def list_project_tables(project_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_project(project_id, current_user["_id"])
    tables = await db_client.find_many("baas_tables", {"project_id": project_id})
    return ok([_to_table_response(t) for t in tables])


# ══════════════════════════════════════════════════════════════════════════
# Public/secret-key-authed SDK surface — mounted on a separate CORS-open
# sub-app at /baas in main.py (see Decision #6). No "/baas" prefix here —
# the mount already adds it.
# ══════════════════════════════════════════════════════════════════════════
baas_public_router = APIRouter(tags=["Backend as a Service — Public API"])


async def _get_project_table(project_id: str, table_name: str) -> dict:
    table = await db_client.find_one("baas_tables", {"project_id": project_id, "name": table_name.lower()})
    if not table:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Table '{table_name}' not found in this project.")
    return table


def _coerce_value(value: Any, col_type: str) -> Any:
    if value is None:
        return None
    try:
        if col_type == "number" and not isinstance(value, bool):
            return float(value)
        if col_type == "boolean":
            if isinstance(value, str):
                return value.strip().lower() in ("true", "1", "yes")
            return bool(value)
        return value
    except (TypeError, ValueError):
        return value


def _validate_record_body(table: dict, body: Dict[str, Any]) -> Dict[str, Any]:
    known_columns = {c["name"]: c["type"] for c in table["columns"]}
    unknown = [k for k in body.keys() if k not in known_columns]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown column(s): {unknown}. Defined columns: {list(known_columns.keys())}.",
        )
    return {k: _coerce_value(v, known_columns[k]) for k, v in body.items()}


async def _get_owned_record(project_id: str, table_name: str, record_id: str) -> dict:
    record = await db_client.find_one("baas_records", {"_id": record_id})
    if (
        not record
        or record.get("project_id") != project_id
        or record.get("table_name") != table_name.lower()
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    return record


@baas_public_router.get("/data/{table_name}")
async def list_records(
    table_name: str,
    limit: int = Query(50, ge=1, le=200),
    project: dict = Depends(get_current_project_by_public_key),
):
    _check_rate_limit(project["project_id"])
    await _get_project_table(project["project_id"], table_name)
    records = await db_client.find_many(
        "baas_records", {"project_id": project["project_id"], "table_name": table_name.lower()}
    )
    sorted_records = sorted(records, key=lambda r: r.get("created_at", ""), reverse=True)
    return ok({"records": [_to_record_response(r) for r in sorted_records[:limit]], "total_count": len(sorted_records)})


@baas_public_router.get("/data/{table_name}/{record_id}")
async def get_record(
    table_name: str, record_id: str, project: dict = Depends(get_current_project_by_public_key)
):
    _check_rate_limit(project["project_id"])
    await _get_project_table(project["project_id"], table_name)
    record = await _get_owned_record(project["project_id"], table_name, record_id)
    return ok(_to_record_response(record))


@baas_public_router.post("/data/{table_name}")
async def create_record(
    table_name: str, body: Dict[str, Any] = Body(default_factory=dict), project: dict = Depends(get_current_project)
):
    _check_rate_limit(project["project_id"])
    table = await _get_project_table(project["project_id"], table_name)
    data = _validate_record_body(table, body)
    now = _now()
    record_doc = {
        "_id": str(ObjectId()),
        "project_id": project["project_id"],
        "table_name": table_name.lower(),
        "data": data,
        "created_at": now,
        "updated_at": now,
    }
    await db_client.insert_one("baas_records", record_doc)
    return ok(_to_record_response(record_doc), message="Record created.")


@baas_public_router.patch("/data/{table_name}/{record_id}")
async def update_record(
    table_name: str,
    record_id: str,
    body: Dict[str, Any] = Body(default_factory=dict),
    project: dict = Depends(get_current_project),
):
    _check_rate_limit(project["project_id"])
    table = await _get_project_table(project["project_id"], table_name)
    record = await _get_owned_record(project["project_id"], table_name, record_id)
    updates = _validate_record_body(table, body)
    merged_data = {**record.get("data", {}), **updates}
    now = _now()
    await db_client.update_one("baas_records", {"_id": record_id}, {"$set": {"data": merged_data, "updated_at": now}})
    record["data"] = merged_data
    record["updated_at"] = now
    return ok(_to_record_response(record), message="Record updated.")


@baas_public_router.delete("/data/{table_name}/{record_id}")
async def delete_record(table_name: str, record_id: str, project: dict = Depends(get_current_project)):
    _check_rate_limit(project["project_id"])
    await _get_project_table(project["project_id"], table_name)
    await _get_owned_record(project["project_id"], table_name, record_id)
    await db_client.delete_one("baas_records", {"_id": record_id})
    return ok({"deleted": True}, message="Record deleted.")


@baas_public_router.post("/end-users/signup")
async def end_user_signup(
    request: EndUserSignupRequest, project: dict = Depends(get_current_project_by_public_key)
):
    _check_rate_limit(project["project_id"])
    normalized_email = request.email.lower()
    existing = await db_client.find_one(
        "baas_end_users", {"project_id": project["project_id"], "email": normalized_email}
    )
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")

    user_doc = {
        "_id": str(ObjectId()),
        "project_id": project["project_id"],
        "email": normalized_email,
        "password_hash": hash_password(request.password),
        "created_at": _now(),
    }
    user_id = await db_client.insert_one("baas_end_users", user_doc)
    user_doc["_id"] = user_id
    token = create_end_user_token(user_id=user_id, project_id=project["project_id"], email=normalized_email)
    return ok({"user": _to_end_user_response(user_doc), "token": token}, message="Account created successfully.")


@baas_public_router.post("/end-users/login")
async def end_user_login(request: EndUserLoginRequest, project: dict = Depends(get_current_project_by_public_key)):
    _check_rate_limit(project["project_id"])
    normalized_email = request.email.lower()
    user = await db_client.find_one(
        "baas_end_users", {"project_id": project["project_id"], "email": normalized_email}
    )
    if not user or not verify_password(request.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    token = create_end_user_token(user_id=user["_id"], project_id=project["project_id"], email=user["email"])
    return ok({"user": _to_end_user_response(user), "token": token}, message="Logged in successfully.")


@baas_public_router.get("/end-users/me")
async def end_user_me(current_end_user: dict = Depends(get_current_end_user)):
    return ok(_to_end_user_response(current_end_user))
