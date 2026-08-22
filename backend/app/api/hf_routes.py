import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.hf import provider as hf_provider
from app.hf.client import HFClientError
from app.utils.response import ok

logger = logging.getLogger("regression_studio.hf_routes")

router = APIRouter(prefix="/hf", tags=["Hugging Face Models"])

_ERROR_STATUS = {
    "not_found": status.HTTP_404_NOT_FOUND,
    "rate_limit": status.HTTP_429_TOO_MANY_REQUESTS,
    "network": status.HTTP_502_BAD_GATEWAY,
    "unknown": status.HTTP_502_BAD_GATEWAY,
}


def _raise_http_from_hf_error(e: HFClientError) -> None:
    raise HTTPException(status_code=_ERROR_STATUS.get(e.category, status.HTTP_502_BAD_GATEWAY), detail=str(e))


async def _get_owned_dataset(dataset_id: str, user_id: str) -> dict:
    dataset = await db_client.find_one("datasets", {"_id": dataset_id})
    if not dataset or dataset.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found.",
        )
    return dataset


class CheckCompatibilityRequest(BaseModel):
    dataset_id: str


@router.get("/search")
async def search_models(
    query: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(get_current_user),
):
    try:
        results = await hf_provider.search(query, task=task, limit=limit)
    except HFClientError as e:
        _raise_http_from_hf_error(e)
    return ok(results)


@router.get("/models/{model_id:path}")
async def get_model(model_id: str, current_user: dict = Depends(get_current_user)):
    try:
        details = await hf_provider.get_details(model_id)
    except HFClientError as e:
        _raise_http_from_hf_error(e)
    return ok(details)


@router.post("/models/{model_id:path}/check-compatibility")
async def check_compatibility(
    model_id: str, request: CheckCompatibilityRequest, current_user: dict = Depends(get_current_user)
):
    dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])
    dataset_schema = {"data_types": dataset.get("data_types", {})}

    try:
        result = await hf_provider.check_compatibility(model_id, dataset_schema)
    except HFClientError as e:
        _raise_http_from_hf_error(e)

    return ok(
        {
            "model_id": model_id,
            "dataset_id": request.dataset_id,
            "compatible": result.compatible,
            "execution_mode": result.execution_mode,
            "confidence": result.confidence,
            "reasons": result.reasons,
        }
    )
