import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

logger = logging.getLogger("regression_studio.hf.client")

# Works fully unauthenticated against the public Hub. HF_TOKEN is optional —
# only needed for gated/private repos or higher rate limits.
_api = HfApi(token=os.getenv("HF_TOKEN") or None)


class HFClientError(Exception):
    """Every function in this module raises this instead of letting a raw
    huggingface_hub/requests exception propagate — mirrors ConnectorError's
    category pattern (app/connectors/base.py) so route handlers can map it to
    a safe HTTPException detail without leaking request/response internals."""

    def __init__(self, category: str, message: str):
        # category: "not_found" | "rate_limit" | "network" | "unknown"
        self.category = category
        super().__init__(message)


def _model_summary(info: Any) -> Dict[str, Any]:
    return {
        "model_id": info.id,
        "author": info.author,
        "pipeline_tag": info.pipeline_tag,
        "library_name": info.library_name,
        "tags": info.tags or [],
        "downloads": info.downloads,
        "likes": info.likes,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
    }


def _model_details(info: Any) -> Dict[str, Any]:
    files = [s.rfilename for s in (info.siblings or [])]
    license_str = getattr(info.card_data, "license", None) if info.card_data else None
    return {
        **_model_summary(info),
        "license": license_str,
        "files": files,
        "gated": bool(info.gated) if info.gated is not None else False,
    }


def _search_models_sync(query: Optional[str], task: Optional[str], limit: int) -> List[Dict[str, Any]]:
    try:
        results = _api.list_models(
            search=query or None,
            pipeline_tag=task or None,
            limit=limit,
            sort="downloads",
        )
        return [_model_summary(m) for m in results]
    except HfHubHTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code == 429:
            raise HFClientError("rate_limit", "Hugging Face Hub is rate-limiting requests. Please try again shortly.")
        raise HFClientError("network", "Failed to reach the Hugging Face Hub.")
    except Exception as e:
        logger.warning(f"HF search failed (query={query!r}, task={task!r}): {e}")
        raise HFClientError("unknown", "Failed to search the Hugging Face Hub.")


def _get_model_info_sync(model_id: str) -> Dict[str, Any]:
    try:
        info = _api.model_info(model_id, files_metadata=False)
        return _model_details(info)
    except RepositoryNotFoundError:
        raise HFClientError("not_found", f"Model '{model_id}' was not found on the Hugging Face Hub.")
    except HfHubHTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code == 429:
            raise HFClientError("rate_limit", "Hugging Face Hub is rate-limiting requests. Please try again shortly.")
        raise HFClientError("network", "Failed to reach the Hugging Face Hub.")
    except Exception as e:
        logger.warning(f"HF get_model_info failed for {model_id!r}: {e}")
        raise HFClientError("unknown", f"Failed to fetch details for model '{model_id}'.")


async def search_models(query: Optional[str], task: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_search_models_sync, query, task, limit)


async def get_model_info(model_id: str) -> Dict[str, Any]:
    return await asyncio.to_thread(_get_model_info_sync, model_id)
