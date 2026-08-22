import asyncio
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pymysql
import pymysql.cursors

from app.connectors.base import ColumnSchema, ColumnStat, ConnectorError, DatabaseConnector, QueryResult, TableSchema
from app.services.query_service import quote_identifier, validate_identifier, wrap_with_limit

SYSTEM_DATABASES = {"information_schema", "mysql", "performance_schema", "sys"}

# pandas warns that PyMySQL isn't a SQLAlchemy connectable; it's a fully
# supported DBAPI2 connection and works correctly here (verified) — purely
# cosmetic. A module-level filter (rather than a scoped catch_warnings())
# because the actual read happens inside asyncio.to_thread, and the
# warnings module's filter state isn't reliably scoped across threads.
warnings.filterwarnings(
    "ignore", message="pandas only supports SQLAlchemy connectable", category=UserWarning
)


def _classify(exc: Exception) -> ConnectorError:
    if isinstance(exc, pymysql.err.OperationalError):
        code = exc.args[0] if exc.args else None
        message = str(exc.args[1]) if len(exc.args) > 1 else str(exc)
        if code == 1045:  # Access denied
            return ConnectorError("auth", "Database authentication failed.")
        if code in (2003,) and "timed out" in message.lower():
            return ConnectorError("timeout", "The database request timed out.")
        if code in (2003, 2005, 2002):
            return ConnectorError("connection", "Unable to connect. Please verify your connection details.")
        return ConnectorError("connection", "Unable to connect. Please verify your connection details.")
    if isinstance(exc, pymysql.err.MySQLError):
        return ConnectorError("query", "The database rejected the operation.")
    if "timed out" in str(exc).lower():
        return ConnectorError("timeout", "The database request timed out.")
    return ConnectorError("connection", "Unable to connect. Please verify your connection details.")


class MySQLConnector(DatabaseConnector):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _connect(self):
        try:
            timeout = int(self.config.get("connection_timeout_sec", 10))
            conn = pymysql.connect(
                host=self.config["host"],
                port=int(self.config["port"]),
                user=self.config["username"],
                password=self.config["password"],
                database=self.config.get("database") or None,
                connect_timeout=timeout,
                cursorclass=pymysql.cursors.Cursor,
                ssl={"ssl": {}} if self.config.get("ssl_enabled") else None,
                autocommit=True,
            )
            with conn.cursor() as cur:
                cur.execute(f"SET SESSION MAX_EXECUTION_TIME={timeout * 1000}")
            return conn
        except Exception as e:
            raise _classify(e)

    async def test_connection(self) -> Optional[str]:
        def _run():
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT VERSION()")
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
                    cur.execute("SHOW DATABASES")
                    return [r[0] for r in cur.fetchall() if r[0] not in SYSTEM_DATABASES]
            finally:
                conn.close()

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return [self.config.get("database")] if self.config.get("database") else []

    async def list_tables(self, database: str) -> List[TableSchema]:
        validate_identifier(database, "database name")

        def _run():
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
                        FROM information_schema.columns
                        WHERE TABLE_SCHEMA = %s
                        ORDER BY TABLE_NAME, ORDINAL_POSITION
                        """,
                        (database,),
                    )
                    col_rows = cur.fetchall()

                    cur.execute(
                        "SELECT TABLE_NAME, TABLE_ROWS FROM information_schema.tables WHERE TABLE_SCHEMA = %s",
                        (database,),
                    )
                    row_estimates = {r[0]: r[1] for r in cur.fetchall()}

                tables: Dict[str, TableSchema] = {}
                for table_name, col_name, data_type, is_nullable, col_key in col_rows:
                    tbl = tables.setdefault(
                        table_name,
                        TableSchema(name=table_name, columns=[], row_count_estimate=row_estimates.get(table_name)),
                    )
                    tbl.columns.append(
                        ColumnSchema(
                            name=col_name,
                            type=data_type,
                            nullable=(is_nullable == "YES"),
                            is_pk=(col_key == "PRI"),
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
        sql = f"SELECT * FROM {quote_identifier(database, 'mysql')}.{quote_identifier(table, 'mysql')}"
        return await self._run_data_query(sql, limit)

    async def import_query(self, database: str, query_spec: Dict[str, Any], limit: int) -> QueryResult:
        return await self._run_data_query(query_spec["sql"], limit)

    async def _run_data_query(self, sql: str, limit: int) -> QueryResult:
        wrapped = wrap_with_limit(sql, limit + 1)

        def _run():
            conn = self._connect()
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

        return _dataframe_to_result(df, limit)

    def _compute_stats(self, result: QueryResult) -> List[ColumnStat]:
        return _compute_column_stats(result)

    async def close(self) -> None:
        pass  # connections are opened/closed per-operation (see _connect/_run)


def _dataframe_to_result(df: pd.DataFrame, limit: int) -> QueryResult:
    from app.services.query_service import normalize_value

    truncated = len(df) > limit
    df = df.iloc[:limit]
    columns = df.columns.tolist()
    rows = [[normalize_value(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return QueryResult(
        columns=columns, rows=rows, row_count_returned=len(rows), truncated=truncated, limit_applied=limit
    )


def _compute_column_stats(result: QueryResult) -> List[ColumnStat]:
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
