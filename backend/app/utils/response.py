import math
from typing import Any, Optional


def ok(data: Any = None, message: Optional[str] = None) -> dict:
    """Standard success envelope: {success, data, message}."""
    return {"success": True, "data": data, "message": message}


def err(code: int, message: str) -> dict:
    """Standard error envelope: {success, error: {code, message}}."""
    return {"success": False, "error": {"code": code, "message": message}}


def sanitize_floats(value: Any) -> Any:
    """Recursively replaces NaN/Infinity floats with None.

    sklearn/statsmodels can produce NaN or inf in degenerate cases — e.g. R2
    is undefined for a 1-sample test split, or an OLS fit on a near-singular
    matrix. Python's json module allows NaN by default, but Starlette's
    JSONResponse sets allow_nan=False, so an unsanitized NaN reaching a
    response crashes JSON encoding entirely (a 500 with no usable body).
    Apply this to any metrics/statistics dict before returning or persisting it.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {k: sanitize_floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_floats(v) for v in value]
    return value
