from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import pandas as pd
from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.ml.preprocessing import preprocess_dataframe
from app.utils.response import ok
from bson import ObjectId

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
    dataset = await db_client.find_one("datasets", {"_id": request.dataset_id})
    if not dataset or dataset.get("user_id") != current_user["_id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {request.dataset_id} not found."
        )

    try:
        # Load dataset into pandas DataFrame
        df = pd.DataFrame(dataset["rows"], columns=dataset["columns"])
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

    # Prepare details of new dataset
    column_names = processed_df.columns.tolist()
    data_types = {col: str(dtype) for col, dtype in zip(processed_df.columns, processed_df.dtypes)}
    row_count = len(processed_df)
    col_count = len(column_names)

    # Convert NaN to None for JSON compliance
    cleaned_rows = processed_df.where(pd.notnull(processed_df), None).values.tolist()

    preprocessed_doc = {
        "_id": str(ObjectId()),
        "user_id": current_user["_id"],
        "name": f"preprocessed_{dataset['name']}",
        "columns": column_names,
        "rows": cleaned_rows,
        "data_types": data_types,
        "row_count": row_count,
        "col_count": col_count,
        "missing_values": int(processed_df.isna().sum().sum()),
        "is_preprocessed": True,
        "original_dataset_id": request.dataset_id,
        "preprocessing_config": config_dict
    }

    preprocessed_dataset_id = await db_client.insert_one("datasets", preprocessed_doc)

    return ok({
        "preprocessed_dataset_id": preprocessed_dataset_id,
        "summary": summary
    }, message="Preprocessing completed successfully.")
