import logging
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.ml.regression_models import get_regression_model
from app.ml.evaluation import calculate_evaluation_metrics, calculate_statistical_properties
from app.ml.prediction import save_model_package, MODELS_DIR
from app.visualization.graph_generator import generate_dataset_graphs, generate_regression_graphs
from app.utils.response import ok, sanitize_floats
from app.utils.perf import PerfTimer

logger = logging.getLogger("regression_studio.training_routes")

router = APIRouter(prefix="", tags=["Model Training & Evaluation"])

class FeatureSelectionRequest(BaseModel):
    dataset_id: str
    features: List[str]
    target: str

class DataSplitConfig(BaseModel):
    test_size: float = Field(0.2, ge=0.05, le=0.95)
    val_size: float = Field(0.0, ge=0.0, le=0.45)
    random_state: int = Field(42, ge=0)

class DataSplitRequest(BaseModel):
    dataset_id: str
    config: DataSplitConfig

class TrainModelRequest(BaseModel):
    dataset_id: str
    model: str = Field("LinearRegression", description="Name of regression model")
    features: List[str]
    target: str
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


async def _get_owned_dataset(dataset_id: str, user_id: str) -> dict:
    dataset = await db_client.find_one("datasets", {"_id": dataset_id})
    if not dataset or dataset.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {dataset_id} not found."
        )
    return dataset


async def _get_owned_model(model_id: str, user_id: str) -> dict:
    model_doc = await db_client.find_one("models", {"_id": model_id})
    if not model_doc or model_doc.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found."
        )
    return model_doc


async def ensure_graphs(model_doc: dict) -> Dict[str, str]:
    """Generates and persists the PNG diagnostic charts for a trained model
    if they haven't been generated yet (see the note in train_model() for
    why this is deferred out of the training request). Reloads the saved
    model package and the source dataset to reproduce exactly what training
    would have plotted. Called from report_routes.py — the only two callers
    that actually need these PNGs (the PDF report and the /graphs endpoint)."""
    if model_doc.get("graph_paths"):
        return model_doc["graph_paths"]

    model_id = model_doc["_id"]
    dataset = await db_client.find_one("datasets", {"_id": model_doc.get("dataset_id")})
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if not dataset or not os.path.exists(model_path):
        return {}

    try:
        package = joblib.load(model_path)
        model_instance = package["model"]
        scaler = package["scaler"]
        features = model_doc["features"]
        target = model_doc["target"]

        df = pd.DataFrame(dataset["rows"], columns=dataset["columns"])
        X = df[features]
        X_for_predict = X
        if scaler is not None:
            X_for_predict = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        y_pred = model_instance.predict(X_for_predict)

        ds_paths = generate_dataset_graphs(df[features], target, model_id)
        reg_paths = generate_regression_graphs(
            y_actual=df[target].values,
            y_predicted=y_pred,
            features=features,
            coefficients=model_doc["statistical_analysis"]["coefficients"],
            model_id=model_id
        )
        graph_paths = {**ds_paths, **reg_paths}
    except Exception as e:
        logger.warning(f"Deferred graph generation failed for model {model_id}: {e}")
        return {}

    await db_client.update_one("models", {"_id": model_id}, {"$set": {"graph_paths": graph_paths}})
    return graph_paths


# --- ROUTE 1: FEATURE SELECTION ---
@router.post("/select-features")
async def select_features(
    request: FeatureSelectionRequest, current_user: dict = Depends(get_current_user)
):
    dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])
    df = pd.DataFrame(dataset["rows"], columns=dataset["columns"])

    # 1. Validate target column exists
    if request.target not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{request.target}' does not exist in the dataset."
        )

    # 2. Validate feature columns exist
    missing_feats = [f for f in request.features if f not in df.columns]
    if missing_feats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature columns {missing_feats} do not exist in the dataset."
        )

    # 3. Validate numerical columns
    non_numeric_cols = []
    all_cols = request.features + [request.target]
    for col in all_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)

    if non_numeric_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns {non_numeric_cols} are not numerical. Only numerical columns can be selected for regression."
        )

    # 4. Check for missing values
    missing_val_counts = df[all_cols].isna().sum().to_dict()
    cols_with_missing = {k: v for k, v in missing_val_counts.items() if v > 0}
    if cols_with_missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns {list(cols_with_missing.keys())} contain missing values ({cols_with_missing}). Please apply missing values preprocessing first."
        )

    return ok({
        "status": "validated",
        "features": request.features,
        "target": request.target
    })

# --- ROUTE 2: DATASET SPLITTING (preview only) ---
@router.post("/split-data")
async def split_data(
    request: DataSplitRequest, current_user: dict = Depends(get_current_user)
):
    dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])

    total_rows = dataset["row_count"]

    # Calculate row counts based on proportions
    test_size = request.config.test_size
    val_size = request.config.val_size

    test_rows = int(total_rows * test_size)
    val_rows = int(total_rows * val_size) if val_size > 0 else 0
    train_rows = total_rows - test_rows - val_rows

    if train_rows <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Split ratios result in 0 training rows. Please adjust test/validation sizes."
        )

    return ok({
        "train_rows": train_rows,
        "test_rows": test_rows,
        "val_rows": val_rows,
        "random_state": request.config.random_state
    })

# --- ROUTE 3: TRAIN MODEL ---
@router.post("/train-model")
async def train_model(
    request: TrainModelRequest, current_user: dict = Depends(get_current_user)
):
    perf = PerfTimer(f"train-model ({request.model})")

    # 1. Fetch dataset
    with perf.stage("fetch_dataset (db)"):
        dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])

    with perf.stage("build_dataframe"):
        df = pd.DataFrame(dataset["rows"], columns=dataset["columns"])

    # 2. Extract selected features X and target y
    all_selected = request.features + [request.target]
    for col in all_selected:
        if col not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{col}' not found in the dataset."
            )

    X = df[request.features]
    y = df[request.target]

    # 3. Perform splits
    test_size = request.split.test_size
    val_size = request.split.val_size
    random_state = request.split.random_state

    with perf.stage("train_test_split"):
        try:
            # If val_size is requested, do a 3-way split
            if val_size > 0:
                # First split: train + val vs test
                test_val_ratio = test_size + val_size
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=test_size/test_val_ratio, random_state=random_state
                )
                # Second split: train vs val
                val_ratio_in_train_val = val_size / test_val_ratio
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_ratio_in_train_val, random_state=random_state
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                X_val, y_val = None, None
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Data split failed. Typically caused by too few rows: {e}"
            )

    # 3.5 Feature scaling — fit ONLY on the training split to avoid data leakage.
    # The scaling method is whatever was chosen during /preprocess (stored on the
    # dataset doc); if the caller trained straight off a raw (non-preprocessed)
    # dataset, no scaling method is present and none is applied.
    with perf.stage("scaling"):
        scaling_method = dataset.get("preprocessing_config", {}).get("scaling", "none")
        scaler = None
        if scaling_method in ("standard", "minmax"):
            scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            if X_val is not None:
                X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

    # 4. Instantiate and fit model
    with perf.stage("model_fit"):
        try:
            model_instance = get_regression_model(request.model, request.hyperparameters)
            model_instance.fit(X_train, y_train)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Model fitting failed: {e}"
            )

    # 5. Predictions & Evaluation Metrics (on the held-out test split)
    with perf.stage("predict_test + metrics"):
        y_test_pred = model_instance.predict(X_test)
        metrics = calculate_evaluation_metrics(
            y_true=y_test.values,
            y_pred=y_test_pred,
            n_samples=len(X_test),
            n_features=X_test.shape[1]
        )

    # 6. Statistical OLS properties (using Statsmodels, fit on the same scaled train split)
    with perf.stage("statsmodels_ols"):
        stats_properties = calculate_statistical_properties(X_train, y_train)

    # 7. Generate a unique model_id and save model package (.pkl), including the
    # fitted scaler so /predict applies the same transform used at train time.
    model_id = str(uuid.uuid4())[:8]  # Keep ID readable
    with perf.stage("save_model_pkl"):
        try:
            save_model_package(
                model_id=model_id,
                model=model_instance,
                features=request.features,
                target=request.target,
                scaler=scaler
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to serialize/save trained model: {e}"
            )

    # 8. PNG diagnostic charts (for the PDF report) are intentionally NOT
    # generated here. Profiling showed matplotlib/seaborn rendering of the
    # 7 dataset+regression plots costs ~5-7s per training call — real work,
    # but wasted here: nothing in the training response or the frontend's
    # interactive charts (which use chart_data, built below) touches these
    # PNGs. Only the PDF report and the /graphs endpoint need them, and
    # neither is requested during training. They're generated lazily on
    # first actual need by _ensure_graphs() (ownership: report_routes.py),
    # the same deferred pattern already used for the AI explanation.
    graph_paths: Dict[str, str] = {}

    # Real chart_data for the frontend's interactive Recharts visualizations —
    # reuses the same OLS coefficients as the feature-importance PNG so both
    # views agree, rather than sourcing "importance" from two different places.
    with perf.stage("chart_data + correlation"):
        y_test_actual_list = [float(v) for v in np.asarray(y_test.values).tolist()]
        y_test_pred_list = [float(v) for v in np.asarray(y_test_pred).tolist()]
        residuals_list = [a - p for a, p in zip(y_test_actual_list, y_test_pred_list)]
        feature_importance = [
            {"feature": k, "value": float(v)}
            for k, v in stats_properties["coefficients"].items()
            if k != "const" and k in request.features
        ]

        corr_cols = [c for c in all_selected if pd.api.types.is_numeric_dtype(df[c])]
        corr_df = df[corr_cols].corr().round(4).fillna(0)
        correlation_matrix = {
            "columns": corr_df.columns.tolist(),
            "matrix": corr_df.values.tolist()
        }

        chart_data = {
            "actual": y_test_actual_list,
            "predicted": y_test_pred_list,
            "residuals": residuals_list,
            "feature_importance": feature_importance,
            "correlation_matrix": correlation_matrix
        }

    # R2/Adjusted R2 (and occasionally OLS coefficients on a near-singular fit)
    # can come back as NaN/inf in degenerate cases — most commonly a test split
    # so small it leaves 0-1 test samples, where R2 is mathematically undefined.
    # Sanitize before this reaches persistence or a JSON response: Starlette's
    # JSONResponse rejects NaN/inf outright (allow_nan=False), which would
    # otherwise crash the request with an unrecoverable 500.
    with perf.stage("sanitize_floats"):
        metrics = sanitize_floats(metrics)
        stats_properties = sanitize_floats(stats_properties)
        chart_data = sanitize_floats(chart_data)

    # Store metadata in DB
    model_doc = {
        "_id": model_id,
        "user_id": current_user["_id"],
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "model": request.model,
        "features": request.features,
        "target": request.target,
        "split": request.split.model_dump(),
        "metrics": metrics,
        "statistical_analysis": stats_properties,
        "chart_data": chart_data,
        "graph_paths": graph_paths,
        "status": "completed",
        "created_at": pd.Timestamp.now().isoformat()
    }

    with perf.stage("db_insert (json write)"):
        await db_client.insert_one("models", model_doc)

    perf.report()

    return ok({
        "status": "completed",
        "model": request.model,
        "model_id": model_id,
        "metrics": metrics
    }, message="Model trained successfully.")

# --- ROUTE 4: GET MODEL METRICS ---
@router.get("/model-metrics/{model_id}")
async def get_model_metrics(model_id: str, current_user: dict = Depends(get_current_user)):
    model_doc = await _get_owned_model(model_id, current_user["_id"])

    # Defensive backstop against any record persisted before responses were
    # sanitized at write time (see /train-model) — a NaN/inf here would
    # otherwise crash JSON encoding on the way out.
    return ok(sanitize_floats({
        "model_id": model_doc["_id"],
        "model": model_doc["model"],
        "features": model_doc["features"],
        "target": model_doc["target"],
        "metrics": model_doc["metrics"],
        "chart_data": model_doc.get("chart_data", {}),
        "statistical_analysis": {
            "coefficients": model_doc["statistical_analysis"]["coefficients"],
            "p_values": model_doc["statistical_analysis"]["p_values"],
            "standard_errors": model_doc["statistical_analysis"]["standard_errors"],
            "t_statistics": model_doc["statistical_analysis"]["t_statistics"],
            "f_statistic": model_doc["statistical_analysis"]["f_statistic"],
            "f_pvalue": model_doc["statistical_analysis"]["f_pvalue"]
        }
    }))

# Also support static generic "/metrics" to fetch the current user's latest model
@router.get("/metrics")
async def get_latest_metrics(current_user: dict = Depends(get_current_user)):
    models = await db_client.find_many("models", {"user_id": current_user["_id"]})
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No trained models found. Train a model first."
        )
    # Sort models by creation date (descending)
    sorted_models = sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
    latest = sorted_models[0]
    return await get_model_metrics(latest["_id"], current_user)

# List all of the current user's trained models (used by History/Compare views)
@router.get("/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    models = await db_client.find_many("models", {"user_id": current_user["_id"]})
    sorted_models = sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
    return ok(sanitize_floats([
        {
            "model_id": m["_id"],
            "dataset_id": m["dataset_id"],
            "dataset_name": m.get("dataset_name"),
            "model": m["model"],
            "metrics": m["metrics"],
            "created_at": m.get("created_at"),
            "source": m.get("source"),
            "hf_model_id": m.get("hf_model_id"),
        } for m in sorted_models
    ]))
