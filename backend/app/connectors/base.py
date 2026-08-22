from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ColumnSchema:
    name: str
    type: str  # native/driver-reported type string, e.g. "varchar(255)", "int", "ObjectId"
    nullable: bool = True
    is_pk: bool = False


@dataclass
class TableSchema:
    name: str  # table (SQL) or collection (Mongo) name
    columns: List[ColumnSchema] = field(default_factory=list)
    row_count_estimate: Optional[int] = None


@dataclass
class ColumnStat:
    column: str
    dtype: str
    null_count: int
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[List[Any]]  # already normalized to JSON-safe primitives
    row_count_returned: int
    truncated: bool
    limit_applied: int


class ConnectorError(Exception):
    """Every connector method must raise this instead of letting a raw
    driver exception propagate — the app's global exception handler echoes
    str(exc) straight into the client response, and raw driver errors can
    embed hosts, DSNs, or even passwords."""

    def __init__(self, category: str, message: str):
        # category: "connection" | "auth" | "timeout" | "unsupported" | "query" | "not_found"
        self.category = category
        super().__init__(message)


class DatabaseConnector(ABC):
    """One instance per operation — cheap to construct, always closed via
    `close()` in a try/finally at the call site. Not a pooled long-lived
    singleton; Phase 1 deliberately keeps connection lifecycle simple."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def test_connection(self) -> Optional[str]:
        """Connect, run a trivial no-op query, return an optional server
        version string. Raises ConnectorError on failure. Must never
        persist anything."""

    @abstractmethod
    async def list_databases(self) -> List[str]:
        """Best-effort: on permission failure, return [config['database']]
        rather than raising — degrading gracefully is more useful here."""

    @abstractmethod
    async def list_tables(self, database: str) -> List[TableSchema]:
        """Full column detail up front (no separate per-table detail call)."""

    @abstractmethod
    async def preview_table(
        self, database: str, table: str, limit: int
    ) -> Tuple[QueryResult, List[ColumnStat]]:
        """Bounded preview plus per-column stats computed over the previewed
        rows only — never a full-table scan."""

    @abstractmethod
    async def import_table(self, database: str, table: str, limit: int) -> QueryResult:
        """Full table/collection import, capped at `limit` server-side."""

    @abstractmethod
    async def import_query(self, database: str, query_spec: Dict[str, Any], limit: int) -> QueryResult:
        """SQL connectors: query_spec = {"sql": "<already-validated SELECT>"}.
        Mongo connector: query_spec = {"collection": str, "filter": dict,
        "sort": List[Tuple[str,int]], "fields": Optional[List[str]]}.
        Validation happens in app/services/query_service.py before this is
        called — the connector still must not string-concatenate anything
        beyond what's already been validated."""

    @abstractmethod
    async def close(self) -> None:
        """Release any open connection/client."""
