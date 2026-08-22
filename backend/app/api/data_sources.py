import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from app.connectors import ConnectorError, get_connector
from app.database.mongodb import db_client
from app.services.query_service import validate_mongo_query_spec, validate_select_sql
from app.utils.crypto import decrypt_secret, encrypt_secret
from app.utils.db_errors import raise_http_from_connector_error
from app.utils.response import ok, sanitize_floats

logger = logging.getLogger("regression_studio.data_sources")

router = APIRouter(prefix="/data-sources", tags=["Data Sources"])

DEFAULT_IMPORT_ROWS = 10_000
MAX_IMPORT_ROWS = 100_000
DEFAULT_PREVIEW_ROWS = 100
MAX_PREVIEW_ROWS = 500


# ── Request/response models ──────────────────────────────────────────────
class SQLConnectionInput(BaseModel):
    engine: Literal["mysql", "postgresql"]
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_enabled: bool = False
    connection_timeout_sec: int = Field(10, ge=1, le=60)


class MongoConnectionInput(BaseModel):
    engine: Literal["mongodb"] = "mongodb"
    uri: str
    database: str
    connection_timeout_sec: int = Field(10, ge=1, le=60)


ConnectionInput = Union[SQLConnectionInput, MongoConnectionInput]


class TestConnectionRequest(BaseModel):
    connection: ConnectionInput = Field(discriminator="engine")


class CreateDataSourceRequest(BaseModel):
    name: str
    connection: ConnectionInput = Field(discriminator="engine")


class DataSourceSummary(BaseModel):
    id: str
    name: str
    engine: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: str
    username: Optional[str] = None
    ssl_enabled: bool = False
    created_at: str
    updated_at: str
    last_tested_at: Optional[str] = None
    last_test_status: Optional[str] = None


class ImportTableRequest(BaseModel):
    database: str
    table: Optional[str] = None
    row_limit: Optional[int] = Field(None, ge=1, le=MAX_IMPORT_ROWS)
    dataset_name: Optional[str] = None
    custom_sql: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    sort: Optional[List[Tuple[str, int]]] = None
    fields: Optional[List[str]] = None


_MONGO_HOST_RE = re.compile(r"^mongodb(?:\+srv)?://(?:[^:@/]+(?::[^@/]*)?@)?([^/?]+)")
_MONGO_USER_RE = re.compile(r"^mongodb(?:\+srv)?://([^:@/]+)(?::[^@/]*)?@")


def _parse_mongo_uri_for_display(uri: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort, display-only extraction — never used for the actual
    connection. Returns (host, username); port is intentionally omitted
    since +srv URIs have no single port and replica sets have several."""
    host_match = _MONGO_HOST_RE.match(uri)
    user_match = _MONGO_USER_RE.match(uri)
    host = host_match.group(1) if host_match else None
    username = user_match.group(1) if user_match else None
    return host, username


async def _get_owned_data_source(data_source_id: str, user_id: str) -> dict:
    source = await db_client.find_one("data_sources", {"_id": data_source_id})
    if not source or source.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source with ID {data_source_id} not found.",
        )
    return source


def _connector_config_from_input(connection: ConnectionInput) -> Dict[str, Any]:
    return connection.model_dump()


def _connector_config_from_saved(source: dict) -> Dict[str, Any]:
    secret = json.loads(decrypt_secret(source["encrypted_secret"]))
    if source["engine"] == "mongodb":
        return {
            "uri": secret["uri"],
            "database": source["database"],
            "connection_timeout_sec": source.get("connection_timeout_sec", 10),
        }
    return {
        "host": source["host"],
        "port": source["port"],
        "database": source["database"],
        "username": source["username"],
        "password": secret["password"],
        "ssl_enabled": source.get("ssl_enabled", False),
        "connection_timeout_sec": source.get("connection_timeout_sec", 10),
    }


def _to_summary(source: dict) -> DataSourceSummary:
    return DataSourceSummary(
        id=source["_id"],
        name=source["name"],
        engine=source["engine"],
        host=source.get("host"),
        port=source.get("port"),
        database=source["database"],
        username=source.get("username"),
        ssl_enabled=source.get("ssl_enabled", False),
        created_at=source["created_at"],
        updated_at=source["updated_at"],
        last_tested_at=source.get("last_tested_at"),
        last_test_status=source.get("last_test_status"),
    )


def _now() -> str:
    return pd.Timestamp.now().isoformat()


# ── Endpoints ─────────────────────────────────────────────────────────────
@router.post("/test")
async def test_connection(request: TestConnectionRequest, current_user: dict = Depends(get_current_user)):
    """Ephemeral connection test — never persists anything, regardless of outcome."""
    connector = get_connector(request.connection.engine, _connector_config_from_input(request.connection))
    try:
        version = await connector.test_connection()
    except ConnectorError as e:
        raise_http_from_connector_error(e)
        return  # unreachable, satisfies type checkers
    finally:
        await connector.close()
    return ok({"status": "ok", "server_version": version})


@router.post("")
async def create_data_source(request: CreateDataSourceRequest, current_user: dict = Depends(get_current_user)):
    conn = request.connection
    now = _now()

    if isinstance(conn, MongoConnectionInput) or conn.engine == "mongodb":
        host, username = _parse_mongo_uri_for_display(conn.uri)
        secret = json.dumps({"uri": conn.uri})
        doc = {
            "_id": str(ObjectId()),
            "user_id": current_user["_id"],
            "name": request.name,
            "engine": "mongodb",
            "host": host,
            "port": None,
            "database": conn.database,
            "username": username,
            "ssl_enabled": False,
            "connection_timeout_sec": conn.connection_timeout_sec,
            "encrypted_secret": encrypt_secret(secret),
            "created_at": now,
            "updated_at": now,
            "last_tested_at": None,
            "last_test_status": None,
            "last_test_error_category": None,
        }
    else:
        secret = json.dumps({"password": conn.password})
        doc = {
            "_id": str(ObjectId()),
            "user_id": current_user["_id"],
            "name": request.name,
            "engine": conn.engine,
            "host": conn.host,
            "port": conn.port,
            "database": conn.database,
            "username": conn.username,
            "ssl_enabled": conn.ssl_enabled,
            "connection_timeout_sec": conn.connection_timeout_sec,
            "encrypted_secret": encrypt_secret(secret),
            "created_at": now,
            "updated_at": now,
            "last_tested_at": None,
            "last_test_status": None,
            "last_test_error_category": None,
        }

    data_source_id = await db_client.insert_one("data_sources", doc)
    doc["_id"] = data_source_id
    return ok(_to_summary(doc).model_dump(), message="Data source saved.")


@router.get("")
async def list_data_sources(current_user: dict = Depends(get_current_user)):
    sources = await db_client.find_many("data_sources", {"user_id": current_user["_id"]})
    sorted_sources = sorted(sources, key=lambda s: s.get("created_at", ""), reverse=True)
    return ok([_to_summary(s).model_dump() for s in sorted_sources])


@router.get("/{data_source_id}")
async def get_data_source(data_source_id: str, current_user: dict = Depends(get_current_user)):
    source = await _get_owned_data_source(data_source_id, current_user["_id"])
    return ok(_to_summary(source).model_dump())


@router.post("/{data_source_id}/test")
async def test_saved_data_source(data_source_id: str, current_user: dict = Depends(get_current_user)):
    source = await _get_owned_data_source(data_source_id, current_user["_id"])
    connector = get_connector(source["engine"], _connector_config_from_saved(source))
    now = _now()
    try:
        version = await connector.test_connection()
    except ConnectorError as e:
        await db_client.update_one(
            "data_sources",
            {"_id": data_source_id},
            {"$set": {"last_tested_at": now, "last_test_status": "failed", "last_test_error_category": e.category}},
        )
        raise_http_from_connector_error(e)
        return
    finally:
        await connector.close()

    await db_client.update_one(
        "data_sources",
        {"_id": data_source_id},
        {"$set": {"last_tested_at": now, "last_test_status": "ok", "last_test_error_category": None}},
    )
    return ok({"status": "ok", "server_version": version})


@router.delete("/{data_source_id}")
async def delete_data_source(data_source_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_data_source(data_source_id, current_user["_id"])
    await db_client.delete_one("data_sources", {"_id": data_source_id})
    # Only the saved connection is removed — datasets/models already
    # imported from it are left untouched (analysis history is never
    # deleted implicitly).
    return ok({"deleted": True}, message="Data source disconnected.")


@router.get("/{data_source_id}/databases")
async def list_databases(data_source_id: str, current_user: dict = Depends(get_current_user)):
    source = await _get_owned_data_source(data_source_id, current_user["_id"])
    connector = get_connector(source["engine"], _connector_config_from_saved(source))
    try:
        databases = await connector.list_databases()
    except ConnectorError as e:
        raise_http_from_connector_error(e)
        return
    finally:
        await connector.close()
    return ok({"databases": databases})


@router.get("/{data_source_id}/tables")
async def list_tables(
    data_source_id: str, database: str = Query(...), current_user: dict = Depends(get_current_user)
):
    source = await _get_owned_data_source(data_source_id, current_user["_id"])
    connector = get_connector(source["engine"], _connector_config_from_saved(source))
    try:
        tables = await connector.list_tables(database)
    except ConnectorError as e:
        raise_http_from_connector_error(e)
        return
    finally:
        await connector.close()
    return ok(
        {
            "tables": [
                {
                    "name": t.name,
                    "row_count_estimate": t.row_count_estimate,
                    "columns": [
                        {"name": c.name, "type": c.type, "nullable": c.nullable, "is_pk": c.is_pk}
                        for c in t.columns
                    ],
                }
                for t in tables
            ]
        }
    )


@router.get("/{data_source_id}/preview")
async def preview_table(
    data_source_id: str,
    database: str = Query(...),
    table: str = Query(...),
    limit: int = Query(DEFAULT_PREVIEW_ROWS, ge=1, le=MAX_PREVIEW_ROWS),
    current_user: dict = Depends(get_current_user),
):
    source = await _get_owned_data_source(data_source_id, current_user["_id"])
    connector = get_connector(source["engine"], _connector_config_from_saved(source))
    try:
        result, stats = await connector.preview_table(database, table, limit)
    except ConnectorError as e:
        raise_http_from_connector_error(e)
        return
    finally:
        await connector.close()
    return ok(
        {
            "columns": result.columns,
            "rows": result.rows,
            "row_count_returned": result.row_count_returned,
            "sampled": True,  # stats/preview are computed over the previewed rows only, never a full scan
            "column_stats": [
                {
                    "column": s.column,
                    "dtype": s.dtype,
                    "null_count": s.null_count,
                    "min": s.min,
                    "max": s.max,
                    "mean": s.mean,
                }
                for s in stats
            ],
        }
    )


@router.post("/{data_source_id}/import")
async def import_data_source_table(
    data_source_id: str, request: ImportTableRequest, current_user: dict = Depends(get_current_user)
):
    source = await _get_owned_data_source(data_source_id, current_user["_id"])
    connector = get_connector(source["engine"], _connector_config_from_saved(source))

    limit = min(request.row_limit or DEFAULT_IMPORT_ROWS, MAX_IMPORT_ROWS)

    try:
        if request.custom_sql:
            if source["engine"] == "mongodb":
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "custom_sql is not supported for MongoDB sources.")
            validated_sql = validate_select_sql(request.custom_sql)
            result = await connector.import_query(request.database, {"sql": validated_sql}, limit)
        elif source["engine"] == "mongodb" and (request.filter or request.sort or request.fields):
            if not request.table:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "table (collection) is required.")
            filter_, sort, fields = validate_mongo_query_spec(request.filter, request.sort, request.fields)
            result = await connector.import_query(
                request.database,
                {"collection": request.table, "filter": filter_, "sort": sort, "fields": fields},
                limit,
            )
        else:
            if not request.table:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "table is required.")
            result = await connector.import_table(request.database, request.table, limit)
    except ConnectorError as e:
        raise_http_from_connector_error(e)
        return
    finally:
        await connector.close()

    df = pd.DataFrame(result.rows, columns=result.columns)
    column_names = df.columns.tolist()
    data_types = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    row_count = len(df)
    col_count = len(column_names)
    missing_values = int(df.isna().sum().sum())
    # df.where(pd.notnull(df), None) does NOT reliably turn NaN into None on
    # float-dtype columns (a documented pandas quirk — None assigned into a
    # float column is coerced back to NaN to preserve dtype), and Starlette's
    # JSONResponse rejects raw NaN outright. sanitize_floats is the existing,
    # already-used-elsewhere fix for exactly this.
    cleaned_rows = sanitize_floats(df.where(pd.notnull(df), None).values.tolist())

    dataset_name = request.dataset_name or f"{source['name']} - {request.table or 'query'}"
    dataset_doc = {
        "_id": str(ObjectId()),
        "user_id": current_user["_id"],
        "name": dataset_name,
        "columns": column_names,
        "rows": cleaned_rows,
        "data_types": data_types,
        "row_count": row_count,
        "col_count": col_count,
        "missing_values": missing_values,
        "source_type": "database",
        "source_data_source_id": data_source_id,
        "source_engine": source["engine"],
        "source_table": request.table,
        "import_row_limit": limit,
        "import_truncated": result.truncated,
    }

    dataset_id = await db_client.insert_one("datasets", dataset_doc)

    return ok(
        {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "rows": row_count,
            "columns": col_count,
            "column_names": column_names,
            "data_types": data_types,
            "missing_values": missing_values,
            "truncated": result.truncated,
            "row_limit_applied": limit,
        },
        message="Dataset imported successfully."
        + (" Row limit applied — not all matching rows were imported." if result.truncated else ""),
    )
