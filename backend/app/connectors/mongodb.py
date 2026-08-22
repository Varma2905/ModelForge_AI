import logging
from typing import Any, Dict, List, Optional, Tuple

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import (
    ConfigurationError,
    ConnectionFailure,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from app.connectors.base import ColumnSchema, ColumnStat, ConnectorError, DatabaseConnector, QueryResult, TableSchema
from app.services.query_service import flatten_mongo_document, union_mongo_fields

logger = logging.getLogger("regression_studio.connectors.mongodb")

SAMPLE_SIZE_FOR_SCHEMA = 50


def _classify(exc: Exception) -> ConnectorError:
    if isinstance(exc, ServerSelectionTimeoutError):
        return ConnectorError("timeout", "The database request timed out.")
    if isinstance(exc, OperationFailure):
        if exc.code in (18, 13):  # AuthenticationFailed, Unauthorized
            return ConnectorError("auth", "Database authentication failed.")
        return ConnectorError("query", "The database rejected the operation.")
    if isinstance(exc, (ConnectionFailure, ConfigurationError)):
        return ConnectorError("connection", "Unable to connect. Please verify your connection details.")
    if isinstance(exc, PyMongoError):
        return ConnectorError("connection", "Unable to connect. Please verify your connection details.")
    return ConnectorError("connection", "The database operation failed.")


def _python_type_name(value: Any) -> str:
    if value is None:
        return "null"
    tname = type(value).__name__
    return {
        "ObjectId": "ObjectId",
        "datetime": "datetime",
        "str": "string",
        "int": "int",
        "float": "float",
        "bool": "bool",
        "dict": "object",
        "list": "array",
    }.get(tname, tname)


class MongoConnector(DatabaseConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        timeout_ms = int(config.get("connection_timeout_sec", 10)) * 1000
        try:
            self._client = AsyncIOMotorClient(
                config["uri"],
                serverSelectionTimeoutMS=timeout_ms,
                connectTimeoutMS=timeout_ms,
            )
        except Exception as e:
            raise _classify(e)

    async def test_connection(self) -> Optional[str]:
        try:
            info = await self._client.admin.command("buildInfo")
            await self._client.admin.command("ping")
            return info.get("version")
        except Exception as e:
            raise _classify(e)

    async def list_databases(self) -> List[str]:
        try:
            names = await self._client.list_database_names()
            return [n for n in names if n not in ("admin", "local", "config")]
        except Exception:
            return [self.config["database"]]

    async def list_tables(self, database: str) -> List[TableSchema]:
        try:
            db = self._client[database]
            collection_names = await db.list_collection_names()
        except Exception as e:
            raise _classify(e)

        tables: List[TableSchema] = []
        for name in collection_names:
            try:
                coll = db[name]
                sample = await coll.find().limit(SAMPLE_SIZE_FOR_SCHEMA).to_list(length=SAMPLE_SIZE_FOR_SCHEMA)
                fields = union_mongo_fields(sample)
                columns = [
                    ColumnSchema(
                        name=f,
                        type=_python_type_name(next((d.get(f) for d in sample if d.get(f) is not None), None)),
                        nullable=any(f not in d or d.get(f) is None for d in sample),
                        is_pk=(f == "_id"),
                    )
                    for f in fields
                ]
                row_count = await coll.estimated_document_count()
                tables.append(TableSchema(name=name, columns=columns, row_count_estimate=row_count))
            except Exception:
                # A single problematic collection shouldn't fail the whole listing.
                tables.append(TableSchema(name=name, columns=[], row_count_estimate=None))
        return tables

    async def preview_table(
        self, database: str, table: str, limit: int
    ) -> Tuple[QueryResult, List[ColumnStat]]:
        result = await self.import_table(database, table, limit)
        stats = self._compute_stats(result)
        return result, stats

    async def import_table(self, database: str, table: str, limit: int) -> QueryResult:
        try:
            coll = self._client[database][table]
            docs = await coll.find().limit(limit + 1).to_list(length=limit + 1)
        except Exception as e:
            raise _classify(e)
        return self._to_query_result(docs, limit)

    async def import_query(self, database: str, query_spec: Dict[str, Any], limit: int) -> QueryResult:
        collection = query_spec["collection"]
        filter_ = query_spec.get("filter") or {}
        sort = query_spec.get("sort") or []
        fields = query_spec.get("fields")

        projection = None
        if fields:
            projection = {f: 1 for f in fields}
            if "_id" not in fields:
                projection["_id"] = 0

        try:
            coll = self._client[database][collection]
            cursor = coll.find(filter_, projection)
            if sort:
                cursor = cursor.sort(sort)
            docs = await cursor.limit(limit + 1).to_list(length=limit + 1)
        except Exception as e:
            raise _classify(e)
        return self._to_query_result(docs, limit)

    def _to_query_result(self, docs: List[Dict[str, Any]], limit: int) -> QueryResult:
        truncated = len(docs) > limit
        docs = docs[:limit]
        columns = union_mongo_fields(docs)
        rows = [flatten_mongo_document(d, columns) for d in docs]
        return QueryResult(
            columns=columns,
            rows=rows,
            row_count_returned=len(rows),
            truncated=truncated,
            limit_applied=limit,
        )

    def _compute_stats(self, result: QueryResult) -> List[ColumnStat]:
        stats: List[ColumnStat] = []
        for idx, col in enumerate(result.columns):
            values = [row[idx] for row in result.rows]
            null_count = sum(1 for v in values if v is None)
            numeric = [v for v in values if isinstance(v, (int, float)) and v is not None and not isinstance(v, bool)]
            dtype = "numeric" if numeric and len(numeric) == (len(values) - null_count) else "mixed/string"
            stat = ColumnStat(column=col, dtype=dtype, null_count=null_count)
            if numeric:
                stat.min = min(numeric)
                stat.max = max(numeric)
                stat.mean = sum(numeric) / len(numeric)
            stats.append(stat)
        return stats

    async def close(self) -> None:
        self._client.close()
