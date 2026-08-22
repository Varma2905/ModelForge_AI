import datetime
import decimal
import json
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import sqlparse
from sqlparse.tokens import DDL, DML, Comment, Keyword

from app.connectors.base import ConnectorError

IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")
MONGO_FIELD_RE = re.compile(r"^[A-Za-z0-9_.]+$")
MAX_SQL_LENGTH = 20_000
MAX_MONGO_FILTER_DEPTH = 6

_DISALLOWED_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "MERGE", "GRANT", "REVOKE", "CALL", "EXEC", "EXECUTE",
    "ATTACH", "DETACH", "VACUUM", "COPY", "SET", "USE", "PREPARE", "LOCK",
    "UNLOCK", "PRAGMA", "REINDEX", "ANALYZE",
}

_ALLOWED_MONGO_OPERATORS = {
    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin",
    "$and", "$or", "$not", "$nor", "$exists", "$regex", "$options",
    "$type", "$elemMatch", "$size", "$all",
}


# ── Value normalization ──────────────────────────────────────────────────
def normalize_value(value: Any) -> Any:
    """Converts a raw driver value into a JSON-safe primitive matching what
    the rest of the app already stores in dataset docs. Required because the
    JSON-fallback store (json.dump) crashes on datetime/Decimal/ObjectId/
    bytes, and a DATE or DECIMAL column is routine in real tables."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        # A SQL NULL in a numeric column becomes NaN once pandas reads it —
        # JSON has no NaN/Infinity; Starlette's JSONResponse rejects them
        # outright (allow_nan=False), so this must become None here, not
        # downstream, since preview/schema responses read connector output
        # directly without a later sanitize_floats pass.
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    # bson.ObjectId duck-typed to avoid a hard bson import dependency here
    type_name = type(value).__name__
    if type_name == "ObjectId":
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        # Binary blobs aren't regression-usable; keeping them would corrupt
        # dtype inference and bloat the JSON-fallback store. Treated as missing.
        return None
    if isinstance(value, dict):
        flat: Dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, dict):
                flat[f"{k}"] = json.dumps({kk: normalize_value(vv) for kk, vv in v.items()})
            else:
                flat[k] = normalize_value(v)
        return json.dumps(flat)
    if isinstance(value, (list, tuple)):
        return json.dumps([normalize_value(v) for v in value])
    return str(value)


def normalize_rows(columns: List[str], raw_rows: List[Any]) -> List[List[Any]]:
    return [[normalize_value(v) for v in row] for row in raw_rows]


def flatten_mongo_document(doc: Dict[str, Any], all_fields: List[str]) -> List[Any]:
    """Aligns one Mongo document to a fixed column list (union of fields
    seen across the sampled batch), one level of dot-notation flattening."""
    flat: Dict[str, Any] = {}
    for k, v in doc.items():
        if k == "_id":
            flat["_id"] = normalize_value(v)
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = normalize_value(vv)
        else:
            flat[k] = normalize_value(v)
    return [flat.get(col) for col in all_fields]


def union_mongo_fields(docs: List[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    seen_set = set()

    def add(name: str):
        if name not in seen_set:
            seen_set.add(name)
            seen.append(name)

    for doc in docs:
        for k, v in doc.items():
            if k == "_id":
                add("_id")
                continue
            if isinstance(v, dict):
                for kk in v.keys():
                    add(f"{k}.{kk}")
            else:
                add(k)
    return seen


# ── Identifier safety (plain table/database browsing, not custom SQL) ───
def validate_identifier(name: str, label: str = "identifier") -> str:
    if not name or not IDENTIFIER_RE.match(name) or len(name) > 128:
        raise ConnectorError("query", f"Invalid {label}.")
    return name


def quote_identifier(name: str, engine: str) -> str:
    validate_identifier(name)
    if engine == "mysql":
        return f"`{name}`"
    return f'"{name}"'  # postgresql


# ── Read-only SQL validation ─────────────────────────────────────────────
def validate_select_sql(raw_sql: str) -> str:
    """Returns the validated, stripped SQL (unwrapped) on success. Raises
    ConnectorError("query", ...) on any rejection. Uses sqlparse rather than
    regex: Postgres's simple query protocol genuinely executes multiple
    ';'-separated statements in one execute() call, so single-statement
    enforcement is a hard requirement, and a token-typed parser avoids both
    false negatives (keywords hidden in string literals/comments) and false
    positives (a column named delete_flag isn't a DELETE keyword token)."""
    if not raw_sql or not raw_sql.strip():
        raise ConnectorError("query", "Query must not be empty.")
    if len(raw_sql) > MAX_SQL_LENGTH:
        raise ConnectorError("query", "Query is too long.")

    try:
        statements = [s for s in sqlparse.split(raw_sql) if s.strip()]
    except Exception:
        raise ConnectorError("query", "Unable to parse the SQL query.")

    if len(statements) != 1:
        raise ConnectorError("query", "Only a single SELECT statement is allowed.")

    stripped = statements[0].strip().rstrip(";").strip()
    if ";" in stripped:
        raise ConnectorError("query", "Multiple statements are not allowed.")

    try:
        parsed = sqlparse.parse(stripped)
    except Exception:
        raise ConnectorError("query", "Unable to parse the SQL query.")

    if not parsed:
        raise ConnectorError("query", "Query must not be empty.")
    stmt = parsed[0]

    first_token = stmt.token_first(skip_cm=True)
    if first_token is None or first_token.normalized.upper() not in ("SELECT", "WITH"):
        raise ConnectorError("query", "Only read-only SELECT queries are allowed.")

    for token in stmt.flatten():
        if token.ttype in Comment:
            raise ConnectorError("query", "Comments are not allowed in the query.")
        if token.ttype is DDL:
            # DROP/ALTER/CREATE/TRUNCATE — never valid in a read-only query.
            raise ConnectorError("query", "Only read-only SELECT queries are allowed.")
        if token.ttype is DML:
            # sqlparse tags SELECT itself as Keyword.DML alongside
            # INSERT/UPDATE/DELETE — only the SELECT value is allowed here.
            if token.normalized.upper() != "SELECT":
                raise ConnectorError("query", "Only read-only SELECT queries are allowed.")
        elif token.ttype in Keyword and token.normalized.upper() in _DISALLOWED_KEYWORDS:
            raise ConnectorError("query", "Only read-only SELECT queries are allowed.")

    return stripped


def wrap_with_limit(validated_sql: str, limit: int) -> str:
    """Wraps rather than injects a LIMIT — DDL/DML isn't valid as a subquery
    expression inside FROM (...), so this is itself a structural backstop
    on top of validate_select_sql's checks."""
    return f"SELECT * FROM ({validated_sql}) AS _q LIMIT {int(limit)}"


# ── Mongo structured query validation ────────────────────────────────────
def _validate_mongo_filter(node: Any, depth: int = 0) -> None:
    if depth > MAX_MONGO_FILTER_DEPTH:
        raise ConnectorError("query", "Filter is nested too deeply.")
    if isinstance(node, dict):
        for k, v in node.items():
            if k.startswith("$"):
                if k not in _ALLOWED_MONGO_OPERATORS:
                    raise ConnectorError("query", f"Operator '{k}' is not allowed.")
            elif not MONGO_FIELD_RE.match(k):
                raise ConnectorError("query", f"Invalid field name '{k}'.")
            _validate_mongo_filter(v, depth + 1)
    elif isinstance(node, list):
        for item in node:
            _validate_mongo_filter(item, depth + 1)


def validate_mongo_query_spec(
    filter_: Optional[Dict[str, Any]],
    sort: Optional[List[Tuple[str, int]]],
    fields: Optional[List[str]],
) -> Tuple[Dict[str, Any], List[Tuple[str, int]], Optional[List[str]]]:
    filter_ = filter_ or {}
    _validate_mongo_filter(filter_)

    validated_sort: List[Tuple[str, int]] = []
    for field_name, direction in sort or []:
        if not MONGO_FIELD_RE.match(field_name):
            raise ConnectorError("query", f"Invalid sort field '{field_name}'.")
        if direction not in (1, -1):
            raise ConnectorError("query", "Sort direction must be 1 or -1.")
        validated_sort.append((field_name, direction))

    validated_fields: Optional[List[str]] = None
    if fields:
        for f in fields:
            if not MONGO_FIELD_RE.match(f):
                raise ConnectorError("query", f"Invalid field name '{f}'.")
        validated_fields = fields

    return filter_, validated_sort, validated_fields
