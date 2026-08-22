from fastapi import HTTPException, status

from app.connectors.base import ConnectorError

_CATEGORY_MAP = {
    "connection": (status.HTTP_502_BAD_GATEWAY, "Unable to connect. Please verify your connection details."),
    "auth": (status.HTTP_401_UNAUTHORIZED, "Database authentication failed."),
    "timeout": (status.HTTP_504_GATEWAY_TIMEOUT, "The database request timed out."),
    "unsupported": (status.HTTP_400_BAD_REQUEST, "This dataset contains unsupported data types."),
    "query": (status.HTTP_400_BAD_REQUEST, None),  # message passed through — already safe by construction
    "not_found": (status.HTTP_404_NOT_FOUND, None),
}


def raise_http_from_connector_error(exc: ConnectorError) -> None:
    """Maps a ConnectorError to a safe HTTPException — never re-raises or
    lets the raw driver exception text reach the client."""
    status_code, generic_message = _CATEGORY_MAP.get(
        exc.category, (status.HTTP_502_BAD_GATEWAY, "The database operation failed.")
    )
    detail = generic_message or str(exc) or "The database operation failed."
    raise HTTPException(status_code=status_code, detail=detail)
