import asyncio
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg2

from app.connectors.base import ColumnSchema, ColumnStat, ConnectorError, DatabaseConnector, QueryResult, TableSchema
from app.services.query_service import normalize_value, quote_identifier, validate_identifier, wrap_with_limit

# See mysql.py for why this is a module-level filter rather than a scoped
# catch_warnings() — the actual read happens inside asyncio.to_thread.
warnings.filterwarnings(
    "ignore", message="pandas only supports SQLAlchemy connectable", category=UserWarning
)


def _classify(exc: Exception) -> ConnectorError:
    # psycopg2 doesn't expose clean per-failure error codes for connection-
    # phase errors the way pymysql does — string matching on the driver's
    # own message is the accepted pattern for this library.
    message = str(exc).lower()
    if "password authentication failed" in message or "authentication failed" in message:
        return ConnectorError("auth", "Database authentication failed.")
    if "timeout" in message or "timed out" in message:
        return ConnectorError("timeout", "The database request timed out.")
    if isinstance(exc, psycopg2.OperationalError):
        return ConnectorError("connection", "Unable to connect. Please verify your connection details.")
    if isinstance(exc, psycopg2.Error):
        return ConnectorError("query", "The database rejected the operation.")
    return ConnectorError("connection", "Unable to connect. Please verify your connection details.")


class PostgresConnector(DatabaseConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _connect(self, database: Optional[str] = None):
        try:
            timeout = int(self.config.get("connection_timeout_sec", 10))
            conn = psycopg2.connect(
                host=self.config["host"],
                port=int(self.config["port"]),
                dbname=database or self.config.get("database") or "postgres",
                user=self.config["username"],
                password=self.config["password"],
                connect_timeout=timeout,
                sslmode="require" if self.config.get("ssl_enabled") else "prefer",
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {timeout * 1000}")
            return conn
        except Exception as e:
            raise _classify(e)

    async def test_connection(self) -> Optional[str]:
        def _run():
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    row = cur.fetchone()
                    return row[0] if row else None
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_run)
        except ConnectorError:
            raise
        except Exception as e:
            raise _classify(e)

    async def list_databases(self) -> List[str]:
        def _run():
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname")
                    return [r[0] for r in cur.fetchall()]
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return [self.config.get("database")] if self.config.get("database") else []

    async def list_tables(self, database: str) -> List[TableSchema]:
        validate_identifier(database, "database name")

        def _run():
            # Postgres requires a real reconnect to browse a different
            # database (unlike MySQL's single-connection cross-db queries).
            conn = self._connect(database=database)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT table_name, column_name, data_type, is_nullable,
                               (SELECT true FROM information_schema.table_constraints tc
                                JOIN information_schema.key_column_usage kcu
                                  ON tc.constraint_name = kcu.constraint_name
                                WHERE tc.constraint_type = 'PRIMARY KEY'
                                  AND tc.table_name = c.table_name
                                  AND kcu.column_name = c.column_name
                                LIMIT 1) AS is_pk
                        FROM information_schema.columns c
                        WHERE table_schema = 'public'
                        ORDER BY table_name, ordinal_position
                        """
                    )
                    col_rows = cur.fetchall()

                    cur.execute(
                        """
                        SELECT relname, reltuples::bigint
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = 'public' AND c.relkind = 'r'
                        """
                    )
                    row_estimates = {r[0]: r[1] for r in cur.fetchall()}

                tables: Dict[str, TableSchema] = {}
                for table_name, col_name, data_type, is_nullable, is_pk in col_rows:
                    # reltuples is -1 for a table that's never been ANALYZE'd
                    # (e.g. freshly created) — not a real estimate, so treat
                    # it as unknown rather than showing a misleading "-1 rows".
                    estimate = row_estimates.get(table_name)
                    if estimate is not None and estimate < 0:
                        estimate = None
                    tbl = tables.setdefault(
                        table_name,
                        TableSchema(name=table_name, columns=[], row_count_estimate=estimate),
                    )
                    tbl.columns.append(
                        ColumnSchema(
                            name=col_name,
                            type=data_type,
                            nullable=(is_nullable == "YES"),
                            is_pk=bool(is_pk),
                        )
                    )
                return list(tables.values())
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_run)
        except ConnectorError:
            raise
        except Exception as e:
            raise _classify(e)

    async def preview_table(
        self, database: str, table: str, limit: int
    ) -> Tuple[QueryResult, List[ColumnStat]]:
        result = await self.import_table(database, table, limit)
        stats = self._compute_stats(result)
        return result, stats

    async def import_table(self, database: str, table: str, limit: int) -> QueryResult:
        validate_identifier(database, "database name")
        validate_identifier(table, "table name")
        sql = f'SELECT * FROM "public".{quote_identifier(table, "postgresql")}'
        return await self._run_data_query(database, sql, limit)

    async def import_query(self, database: str, query_spec: Dict[str, Any], limit: int) -> QueryResult:
        return await self._run_data_query(database, query_spec["sql"], limit)

    async def _run_data_query(self, database: str, sql: str, limit: int) -> QueryResult:
        wrapped = wrap_with_limit(sql, limit + 1)

        def _run():
            conn = self._connect(database=database)
            try:
                # Re-applied here (not just at module level) because test
                # runners such as pytest can reset the warnings filter list
                # per test, which would otherwise undo the module-level filter.
                warnings.filterwarnings(
                    "ignore", message="pandas only supports SQLAlchemy connectable", category=UserWarning
                )
                df = pd.read_sql_query(wrapped, conn)
            finally:
                conn.close()
            return df

        try:
            df = await asyncio.to_thread(_run)
        except ConnectorError:
            raise
        except Exception as e:
            raise _classify(e)

        truncated = len(df) > limit
        df = df.iloc[:limit]
        columns = df.columns.tolist()
        rows = [[normalize_value(v) for v in row] for row in df.itertuples(index=False, name=None)]
        return QueryResult(
            columns=columns, rows=rows, row_count_returned=len(rows), truncated=truncated, limit_applied=limit
        )

    def _compute_stats(self, result: QueryResult) -> List[ColumnStat]:
        stats: List[ColumnStat] = []
        for idx, col in enumerate(result.columns):
            values = [row[idx] for row in result.rows]
            null_count = sum(1 for v in values if v is None)
            numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            dtype = "numeric" if numeric and len(numeric) == (len(values) - null_count) else "mixed/string"
            stat = ColumnStat(column=col, dtype=dtype, null_count=null_count)
            if numeric:
                stat.min = min(numeric)
                stat.max = max(numeric)
                stat.mean = sum(numeric) / len(numeric)
            stats.append(stat)
        return stats

    async def close(self) -> None:
        pass  # connections are opened/closed per-operation (see _connect/_run)
