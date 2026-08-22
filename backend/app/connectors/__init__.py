from typing import Any, Dict

from app.connectors.base import ColumnSchema, ColumnStat, ConnectorError, DatabaseConnector, QueryResult, TableSchema
from app.connectors.mongodb import MongoConnector
from app.connectors.mysql import MySQLConnector
from app.connectors.postgres import PostgresConnector

_ENGINES = {
    "mysql": MySQLConnector,
    "postgresql": PostgresConnector,
    "mongodb": MongoConnector,
}


def get_connector(engine: str, config: Dict[str, Any]) -> DatabaseConnector:
    cls = _ENGINES.get(engine)
    if cls is None:
        raise ConnectorError("unsupported", f"Unsupported database engine: {engine}")
    return cls(config)


__all__ = [
    "get_connector",
    "DatabaseConnector",
    "ConnectorError",
    "ColumnSchema",
    "ColumnStat",
    "TableSchema",
    "QueryResult",
]
