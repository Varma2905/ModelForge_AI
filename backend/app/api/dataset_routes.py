import asyncio
import logging
import os

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app.auth.dependencies import get_current_user
from app.database.database import db_client
from app.datasets import store
from app.datasets.ingest import SUPPORTED_EXTENSIONS, ingest_source
from app.datasets.providers.local import LocalFileProvider
from app.ml.feature_types import classify_columns
from app.utils.response import ok, sanitize_floats

logger = logging.getLogger("regression_studio.dataset_routes")

router = APIRouter(prefix="", tags=["Dataset Management"])

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024


class ManualDatasetInput(BaseModel):
    name: Optional[str] = "manual_dataset"
    columns: List[str]
    rows: List[List[Any]]


class DatasetResponse(BaseModel):
    dataset_id: str
    dataset_name: str
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    missing_values: int


async def _get_owned_dataset(dataset_id: str, user_id: str) -> dict:
    meta = store.load_meta(dataset_id)
    if not meta or meta.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found.",
        )
    return meta


@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    source: str = Form("local"),
    current_user: dict = Depends(get_current_user),
):
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    dataset_source = LocalFileProvider().stage(file, contents)
    try:
        result = await ingest_source(dataset_source, current_user["_id"])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return ok(result, message="Dataset uploaded successfully.")


@router.post("/create-dataset")
async def create_dataset(
    input_data: ManualDatasetInput, current_user: dict = Depends(get_current_user)
):
    # Perform validation on row widths
    for idx, row in enumerate(input_data.rows):
        if len(row) != len(input_data.columns):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Row index {idx} has length {len(row)} which does not match columns length {len(input_data.columns)}."
            )

    # Read to Pandas to infer datatypes
    try:
        df = pd.DataFrame(input_data.rows, columns=input_data.columns)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to align dataset columns: {e}"
        )

    meta = {
        "user_id": current_user["_id"],
        "name": input_data.name or "manual_dataset",
        "source": "manual",
    }
    dataset_id = await asyncio.to_thread(store.save, df, meta)
    stored_meta = store.load_meta(dataset_id)

    return ok({
        "dataset_id": dataset_id,
        "dataset_name": stored_meta["name"],
        "rows": stored_meta["row_count"],
        "columns": stored_meta["col_count"],
        "column_names": stored_meta["columns"],
        "data_types": stored_meta["data_types"],
        "missing_values": stored_meta["missing_values"],
        "source": stored_meta["source"],
    }, message="Dataset created successfully.")


@router.get("/preview-dataset/{dataset_id}")
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(200, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
):
    meta = await _get_owned_dataset(dataset_id, current_user["_id"])

    # Preview-only sampling — the full dataset is still what's used for
    # preprocessing/training (see DatasetStore.load_dataframe callers).
    df = await asyncio.to_thread(store.load_dataframe, dataset_id)
    preview_df = df.head(limit)
    rows_preview = sanitize_floats(preview_df.where(pd.notnull(preview_df), None).values.tolist())

    return ok({
        "name": meta["name"],
        "columns": meta["columns"],
        "rows": rows_preview,
        "total_rows": meta["row_count"],
        "total_columns": meta["col_count"],
        "data_types": meta["data_types"],
        "missing_values": meta.get("missing_values", 0),
        "source": meta.get("source"),
    })


@router.get("/datasets/{dataset_id}/profile")
async def profile_dataset(dataset_id: str, current_user: dict = Depends(get_current_user)):
    """Per-column feature classification (numerical/categorical/datetime,
    missing/unique counts, high-cardinality flag) for the Variables step —
    drives which columns are offered as target vs. features and which
    warnings to surface, without the frontend guessing from raw dtypes.
    """
    meta = await _get_owned_dataset(dataset_id, current_user["_id"])
    df = await asyncio.to_thread(store.load_dataframe, dataset_id)
    classification = classify_columns(df)

    return ok({
        "dataset_id": dataset_id,
        "name": meta["name"],
        "rows": meta["row_count"],
        "columns": meta["col_count"],
        "features": [
            {"name": col, **info} for col, info in classification.items()
        ],
    })


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_dataset(dataset_id, current_user["_id"])
    store.delete(dataset_id)
    return ok({"deleted": True, "dataset_id": dataset_id}, message="Dataset deleted.")


# Also support a stateless preview endpoint or general GET to retrieve all datasets
@router.get("/datasets")
async def list_datasets(current_user: dict = Depends(get_current_user)):
    datasets = store.list_for_user(current_user["_id"])
    return ok([
        {
            "dataset_id": d["dataset_id"],
            "name": d["name"],
            "rows": d["row_count"],
            "columns": d["col_count"],
            "source": d.get("source"),
            "created_at": d.get("created_at"),
            "size_bytes": store.data_file_size(d["dataset_id"]),
        } for d in datasets
    ])


@router.get("/datasets/{dataset_id}/models")
async def list_dataset_models(
    dataset_id: str, current_user: dict = Depends(get_current_user)
):
    await _get_owned_dataset(dataset_id, current_user["_id"])
    models = await db_client.find_many("models", {"dataset_id": dataset_id, "user_id": current_user["_id"]})
    sorted_models = sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)
    return ok(sanitize_floats([
        {
            "model_id": m["_id"],
            "model": m["model"],
            "model_type": m.get("model_type", "regression"),
            "metrics": m["metrics"],
            "created_at": m.get("created_at"),
            "source": m.get("source"),
            "hf_model_id": m.get("hf_model_id"),
        } for m in sorted_models
    ]))
