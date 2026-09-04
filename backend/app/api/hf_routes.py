import asyncio
import logging
import uuid
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.database.database import db_client
from app.datasets import store as dataset_store
from app.hf import provider as hf_provider
from app.hf.client import HFClientError
from app.hf.inference import HFExecutionError
from app.ml.evaluation import calculate_evaluation_metrics
from app.ml.prediction import save_model_package
from app.utils.response import ok, sanitize_floats

logger = logging.getLogger("regression_studio.hf_routes")

router = APIRouter(prefix="/hf", tags=["Hugging Face Models"])

_ERROR_STATUS = {
    "not_found": status.HTTP_404_NOT_FOUND,
    "rate_limit": status.HTTP_429_TOO_MANY_REQUESTS,
    "network": status.HTTP_502_BAD_GATEWAY,
    "unknown": status.HTTP_502_BAD_GATEWAY,
}

_EXECUTION_ERROR_STATUS = {
    "size_limit": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    "download_failed": status.HTTP_502_BAD_GATEWAY,
    "unsafe_content": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "load_failed": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "not_predictive": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "feature_mismatch": status.HTTP_400_BAD_REQUEST,
    "inference_failed": status.HTTP_422_UNPROCESSABLE_ENTITY,
}


def _raise_http_from_hf_error(e: HFClientError) -> None:
    raise HTTPException(status_code=_ERROR_STATUS.get(e.category, status.HTTP_502_BAD_GATEWAY), detail=str(e))


def _raise_http_from_execution_error(e: HFExecutionError) -> None:
    raise HTTPException(
        status_code=_EXECUTION_ERROR_STATUS.get(e.category, status.HTTP_502_BAD_GATEWAY), detail=str(e)
    )


async def _get_owned_dataset(dataset_id: str, user_id: str) -> dict:
    dataset = dataset_store.load_meta(dataset_id)
    if not dataset or dataset.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found.",
        )
    return dataset


class CheckCompatibilityRequest(BaseModel):
    dataset_id: str


class RunModelRequest(BaseModel):
    dataset_id: str
    features: List[str]
    target: str


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
            "artifact_file": result.artifact_file,
        }
    )


@router.post("/models/{model_id:path}/run")
async def run_model(model_id: str, request: RunModelRequest, current_user: dict = Depends(get_current_user)):
    dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])
    df = await asyncio.to_thread(dataset_store.load_dataframe, request.dataset_id)

    all_selected = request.features + [request.target]
    missing_cols = [c for c in all_selected if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Column(s) {missing_cols} not found in the dataset."
        )

    non_numeric_cols = [c for c in all_selected if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns {non_numeric_cols} are not numerical. Only numerical columns can be used.",
        )

    missing_val_counts = df[all_selected].isna().sum()
    cols_with_missing = missing_val_counts[missing_val_counts > 0]
    if not cols_with_missing.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns {list(cols_with_missing.index)} contain missing values. Please preprocess first.",
        )

    dataset_schema = {"data_types": dataset.get("data_types", {})}
    try:
        compat = await hf_provider.check_compatibility(model_id, dataset_schema)
    except HFClientError as e:
        _raise_http_from_hf_error(e)

    # Re-validated server-side, never trusted from a prior client-side check.
    if compat.execution_mode != "inference_only" or not compat.artifact_file:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Model is not ready to run (status: {compat.execution_mode}). {' '.join(compat.reasons)}",
        )

    X = df[request.features]
    y = df[request.target]

    try:
        estimator, predictions, version_warning = await hf_provider.run(model_id, compat.artifact_file, X)
    except HFExecutionError as e:
        _raise_http_from_execution_error(e)

    metrics = calculate_evaluation_metrics(
        y_true=y.values, y_pred=predictions, n_samples=len(y), n_features=X.shape[1]
    )

    y_actual_list = [float(v) for v in np.asarray(y.values).tolist()]
    y_pred_list = [float(v) for v in np.asarray(predictions).tolist()]
    residuals_list = [a - p for a, p in zip(y_actual_list, y_pred_list)]

    corr_cols = [c for c in all_selected if pd.api.types.is_numeric_dtype(df[c])]
    corr_df = df[corr_cols].corr().round(4).fillna(0)
    correlation_matrix = {"columns": corr_df.columns.tolist(), "matrix": corr_df.values.tolist()}

    chart_data = sanitize_floats(
        {
            "actual": y_actual_list,
            "predicted": y_pred_list,
            "residuals": residuals_list,
            "feature_importance": [],
            "correlation_matrix": correlation_matrix,
        }
    )
    metrics = sanitize_floats(metrics)

    # Externally-sourced model, never fit via OLS locally — every existing
    # consumer (ensure_graphs, run_ai_explanation_pipeline, pdf_generator)
    # reads these keys unconditionally, so they must exist even though empty.
    statistical_analysis = {
        "coefficients": {},
        "standard_errors": {},
        "t_statistics": {},
        "p_values": {},
        "f_statistic": 0.0,
        "f_pvalue": 0.0,
        "summary_text": "Not applicable — externally-sourced Hugging Face model, not fit via OLS.",
    }

    new_model_id = str(uuid.uuid4())[:8]
    try:
        save_model_package(
            model_id=new_model_id, model=estimator, features=request.features, target=request.target, preprocessor=None
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save model: {e}"
        )

    model_doc = {
        "_id": new_model_id,
        "user_id": current_user["_id"],
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "model": model_id,
        "features": request.features,
        "target": request.target,
        "split": None,
        "metrics": metrics,
        "statistical_analysis": statistical_analysis,
        "chart_data": chart_data,
        "graph_paths": {},
        "status": "completed",
        "source": "huggingface",
        "hf_model_id": model_id,
        "execution_mode": "inference_only",
        "version_warning": version_warning,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    await db_client.insert_one("models", model_doc)

    return ok(
        {
            "status": "completed",
            "model_id": new_model_id,
            "hf_model_id": model_id,
            "source": "huggingface",
            "metrics": metrics,
            "version_warning": version_warning,
        },
        message="Hugging Face model executed successfully.",
    )
