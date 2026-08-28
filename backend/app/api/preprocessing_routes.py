import asyncio
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from app.auth.dependencies import get_current_user
from app.datasets import store
from app.ml.preprocessing import preprocess_dataframe
from app.utils.response import ok

router = APIRouter(prefix="", tags=["Data Preprocessing"])

class PreprocessConfig(BaseModel):
    missing: str = Field("mean", description="Imputation strategy: 'remove', 'mean', 'median', 'mode', or 'none'")
    dedupe: bool = Field(True, description="Remove duplicate rows if True")
    outlier: str = Field("iqr", description="Outlier detection strategy: 'iqr', 'zscore', or 'none'")
    scaling: str = Field("standard", description="Scaling strategy applied later at train time: 'standard', 'minmax', or 'none'")

class PreprocessRequest(BaseModel):
    dataset_id: str
    config: PreprocessConfig

class PreprocessSummaryResponse(BaseModel):
    preprocessed_dataset_id: str
    summary: Dict[str, Any]

@router.post("/preprocess")
async def preprocess_dataset(
    request: PreprocessRequest, current_user: dict = Depends(get_current_user)
):
    # Fetch the original dataset, scoped to the current user
    dataset = store.load_meta(request.dataset_id)
    if not dataset or dataset.get("user_id") != current_user["_id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {request.dataset_id} not found."
        )

    try:
        df = await asyncio.to_thread(store.load_dataframe, request.dataset_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to load dataset rows: {e}"
        )

    # Execute ML Preprocessing (dedupe/impute/outliers — scaling is deferred to train time)
    config_dict = request.config.model_dump()
    try:
        processed_df, summary = preprocess_dataframe(df, config_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing execution failed: {str(e)}"
        )

    # Parquet stores NaN natively (unlike JSON), so processed_df is saved
    # as-is — no None-conversion needed here, only at API-response boundaries.
    preprocessed_meta = {
        "user_id": current_user["_id"],
        "name": f"preprocessed_{dataset['name']}",
        "is_preprocessed": True,
        "original_dataset_id": request.dataset_id,
        "preprocessing_config": config_dict,
        # Lineage carried forward from the original dataset rather than
        # re-derived, so a preprocessed Kaggle/Drive dataset still reports
        # its true origin.
        "source": dataset.get("source", "local"),
    }

    preprocessed_dataset_id = await asyncio.to_thread(store.save, processed_df, preprocessed_meta)

    return ok({
        "preprocessed_dataset_id": preprocessed_dataset_id,
        "summary": summary
    }, message="Preprocessing completed successfully.")
