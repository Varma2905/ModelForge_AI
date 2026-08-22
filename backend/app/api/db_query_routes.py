import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.connectors import ConnectorError, TableSchema, get_connector
from app.database.mongodb import db_client
from app.services.nl_query_service import NLQueryError, explain_result, generate_query
from app.services.query_service import validate_mongo_query_spec, validate_select_sql
from app.utils.crypto import decrypt_secret
from app.utils.db_errors import raise_http_from_connector_error
from app.utils.response import ok, sanitize_floats

logger = logging.getLogger("regression_studio.db_query_routes")

router = APIRouter(prefix="/db-query", tags=["Database AI Query"])

RESULT_ROW_LIMIT = 200

_NL_ERROR_STATUS = {
    "not_configured": status.HTTP_503_SERVICE_UNAVAILABLE,
    "generation_failed": status.HTTP_502_BAD_GATEWAY,
}


def _raise_http_from_nl_error(e: NLQueryError) -> None:
    raise HTTPException(status_code=_NL_ERROR_STATUS.get(e.category, status.HTTP_502_BAD_GATEWAY), detail=str(e))


async def _get_owned_data_source(data_source_id: str, user_id: str) -> dict:
    source = await db_client.find_one("data_sources", {"_id": data_source_id})
    if not source or source.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source with ID {data_source_id} not found.",
        )
    return source


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


def _format_schema_context(tables: List[TableSchema]) -> str:
    lines = []
    for t in tables:
        col_desc = ", ".join(f"{c.name} ({c.type})" for c in t.columns)
        lines.append(f"- {t.name}: {col_desc}" if col_desc else f"- {t.name}: (no columns found)")
    return "\n".join(lines) if lines else "(no tables/collections found)"


class AskQuestionRequest(BaseModel):
    data_source_id: str
    database: str
    question: str


@router.post("/ask")
async def ask_database_question(request: AskQuestionRequest, current_user: dict = Depends(get_current_user)):
    source = await _get_owned_data_source(request.data_source_id, current_user["_id"])
    connector = get_connector(source["engine"], _connector_config_from_saved(source))

    try:
        try:
            tables = await connector.list_tables(request.database)
        except ConnectorError as e:
            raise_http_from_connector_error(e)
            return

        schema_context = _format_schema_context(tables)

        try:
            generated = await generate_query(request.question, source["engine"], schema_context)
        except NLQueryError as e:
            _raise_http_from_nl_error(e)
            return

        if source["engine"] == "mongodb":
            collection = generated.get("collection")
            real_collections = {t.name for t in tables}
            if collection not in real_collections:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="The AI referenced a collection that doesn't exist in this database. Try rephrasing your question.",
                )
            try:
                filter_, sort, fields = validate_mongo_query_spec(
                    generated.get("filter"), generated.get("sort"), generated.get("fields")
                )
            except ConnectorError as e:
                raise_http_from_connector_error(e)
                return
            query_spec: Dict[str, Any] = {"collection": collection, "filter": filter_, "sort": sort, "fields": fields}
            query_display: Any = {"collection": collection, "filter": filter_, "sort": sort, "fields": fields}
        else:
            try:
                validated_sql = validate_select_sql(generated.get("sql", ""))
            except ConnectorError as e:
                raise_http_from_connector_error(e)
                return
            query_spec = {"sql": validated_sql}
            query_display = validated_sql

        try:
            result = await connector.import_query(request.database, query_spec, RESULT_ROW_LIMIT)
        except ConnectorError as e:
            raise_http_from_connector_error(e)
            return
    finally:
        await connector.close()

    rows = sanitize_floats(result.rows)
    explanation = await explain_result(request.question, result.columns, rows, result.row_count_returned)

    return ok(
        {
            "engine": source["engine"],
            "query": query_display,
            "columns": result.columns,
            "rows": rows,
            "row_count_returned": result.row_count_returned,
            "truncated": result.truncated,
            "explanation": explanation,
        }
    )
