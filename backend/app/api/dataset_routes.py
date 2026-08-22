import io
import logging
import os

import pandas as pd
from bson import ObjectId
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
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
    dataset = await db_client.find_one("datasets", {"_id": dataset_id})
    if not dataset or dataset.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found.",
        )
    return dataset


@router.post("/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    filename = file.filename
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only CSV and Excel (.xlsx, .xls) files are supported."
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum upload size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            # Requires openpyxl installed
            df = pd.read_excel(io.BytesIO(contents))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse the file: {str(e)}"
        )

    # Get metadata
    column_names = df.columns.tolist()
    data_types = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    row_count = len(df)
    col_count = len(column_names)
    missing_values = int(df.isna().sum().sum())

    # Clean rows for JSON serialization (convert NaN to None). Note:
    # df.where(pd.notnull(df), None) alone does NOT reliably turn NaN into
    # None on float-dtype columns (a documented pandas quirk — assigning
    # None into a float column gets coerced back to NaN to preserve dtype),
    # and Starlette's JSONResponse rejects raw NaN outright — sanitize_floats
    # is the backstop already used elsewhere in this app for exactly this.
    cleaned_rows = sanitize_floats(df.where(pd.notnull(df), None).values.tolist())

    dataset_doc = {
        "_id": str(ObjectId()),
        "user_id": current_user["_id"],
        "name": filename,
        "columns": column_names,
        "rows": cleaned_rows,
        "data_types": data_types,
        "row_count": row_count,
        "col_count": col_count,
        "missing_values": missing_values,
    }

    dataset_id = await db_client.insert_one("datasets", dataset_doc)

    return ok({
        "dataset_id": dataset_id,
        "dataset_name": filename,
        "rows": row_count,
        "columns": col_count,
        "column_names": column_names,
        "data_types": data_types,
        "missing_values": missing_values,
    }, message="Dataset uploaded successfully.")


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

    column_names = df.columns.tolist()
    data_types = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    row_count = len(df)
    col_count = len(column_names)
    missing_values = int(df.isna().sum().sum())

    dataset_doc = {
        "_id": str(ObjectId()),
        "user_id": current_user["_id"],
        "name": input_data.name or "manual_dataset",
        "columns": column_names,
        "rows": input_data.rows,
        "data_types": data_types,
        "row_count": row_count,
        "col_count": col_count,
        "missing_values": missing_values,
    }

    dataset_id = await db_client.insert_one("datasets", dataset_doc)

    return ok({
        "dataset_id": dataset_id,
        "dataset_name": dataset_doc["name"],
        "rows": row_count,
        "columns": col_count,
        "column_names": column_names,
        "data_types": data_types,
        "missing_values": missing_values,
    }, message="Dataset created successfully.")


@router.get("/preview-dataset/{dataset_id}")
async def preview_dataset(
    dataset_id: str,
    limit: int = Query(200, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
):
    dataset = await _get_owned_dataset(dataset_id, current_user["_id"])

    # Return limited preview
    rows_preview = dataset["rows"][:limit]

    return ok({
        "name": dataset["name"],
        "columns": dataset["columns"],
        "rows": rows_preview,
        "total_rows": dataset["row_count"],
        "total_columns": dataset["col_count"],
        "data_types": dataset["data_types"],
        "missing_values": dataset.get("missing_values", 0),
    })


# Also support a stateless preview endpoint or general GET to retrieve all datasets
@router.get("/datasets")
async def list_datasets(current_user: dict = Depends(get_current_user)):
    datasets = await db_client.find_many("datasets", {"user_id": current_user["_id"]})
    return ok([
        {
            "dataset_id": d["_id"],
            "name": d["name"],
            "rows": d["row_count"],
            "columns": d["col_count"],
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
            "metrics": m["metrics"],
            "created_at": m.get("created_at"),
        } for m in sorted_models
    ]))
