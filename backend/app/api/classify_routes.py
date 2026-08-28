import asyncio
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.datasets import store as dataset_store
from app.ml.feature_types import classify_columns, map_expanded_coefficients
from app.ml.classification_models import build_classification_pipeline
from app.ml.classification_evaluation import (
    calculate_classification_metrics,
    calculate_classification_statistical_properties,
    extract_class_distribution,
    extract_feature_importance,
    extract_roc_curve,
    extract_precision_recall_curve,
)
from app.ml.prediction import save_model_package
from app.ml.classification_registry import list_classification_models
from app.utils.response import ok, sanitize_floats
from app.utils.perf import PerfTimer
from app.api.training_routes import _get_owned_dataset, DataSplitConfig

logger = logging.getLogger("regression_studio.classify_routes")

router = APIRouter(prefix="/classify", tags=["Classification Training & Evaluation"])

# Realistic classification problems (binary, Iris/Wine-style multiclass,
# loan grades, star ratings) virtually always have <=20 distinct classes;
# genuine numeric regression targets have hundreds+ distinct values in
# practice. This is a heuristic, not a hard guarantee — a tiny dataset with
# a rounded continuous target could in principle slip under this threshold,
# same inherent limitation as feature_types.py's _HIGH_CARDINALITY_MIN_UNIQUE.
_MAX_INTEGER_TARGET_CLASSES = 20


class ClassificationFeatureSelectionRequest(BaseModel):
    dataset_id: str
    features: List[str]
    target: str


class ClassificationTrainModelRequest(BaseModel):
    dataset_id: str
    model: str = Field("LogisticRegression", description="Name of classification model")
    features: List[str]
    target: str
    split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


def _is_valid_classification_target(col_info: Dict[str, Any]) -> bool:
    """A column is a valid classification target if it's categorical, or a
    numerical column that's actually integer-encoded class labels (0/1/2)
    with few enough distinct values to plausibly be classes rather than a
    continuous regression target. See feature_types.py's is_integer_like
    for why "numerical" dtype alone can't distinguish the two."""
    if col_info["kind"] == "categorical":
        return col_info["unique_count"] >= 2
    if col_info["kind"] == "numerical" and col_info["is_integer_like"]:
        return 2 <= col_info["unique_count"] <= _MAX_INTEGER_TARGET_CLASSES
    return False


# --- ROUTE: LIST CLASSIFICATION MODELS (registry-backed catalog + live availability) ---
@router.get("/classification-models")
async def get_classification_models(current_user: dict = Depends(get_current_user)):
    return ok(list_classification_models())


# --- ROUTE 1: FEATURE SELECTION ---
@router.post("/select-features")
async def select_features(
    request: ClassificationFeatureSelectionRequest, current_user: dict = Depends(get_current_user)
):
    await _get_owned_dataset(request.dataset_id, current_user["_id"])
    df = await asyncio.to_thread(dataset_store.load_dataframe, request.dataset_id)

    if request.target not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{request.target}' does not exist in the dataset."
        )

    missing_feats = [f for f in request.features if f not in df.columns]
    if missing_feats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature columns {missing_feats} do not exist in the dataset."
        )

    classification = classify_columns(df)

    # Target must be categorical, OR a low-cardinality integer-encoded label
    # column — never accepted purely because it's "numerical" (that would
    # wrongly admit a continuous regression target), and never silently
    # coerced/rejected without explanation.
    if not _is_valid_classification_target(classification[request.target]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Classification requires a categorical target, or a numerical target with a "
                f"small number of distinct integer values (2-{_MAX_INTEGER_TARGET_CLASSES} classes). "
                f"'{request.target}' has {classification[request.target]['unique_count']} distinct values."
            ),
        )

    datetime_feats = [f for f in request.features if classification[f]["kind"] == "datetime"]
    if datetime_feats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns {datetime_feats} look like date/time values and can't be used as classification features directly.",
        )

    numeric_features = [f for f in request.features if classification[f]["kind"] == "numerical"]
    categorical_features = [f for f in request.features if classification[f]["kind"] == "categorical"]

    target_missing = int(df[request.target].isna().sum())
    if target_missing > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{request.target}' has {target_missing} missing value(s). "
            "Please apply missing-value preprocessing first.",
        )

    high_cardinality_warnings = [
        f"'{f}' appears to be an identifier and contains many unique values. Consider excluding it from the model."
        for f in categorical_features
        if classification[f]["high_cardinality"]
    ]

    return ok({
        "status": "validated",
        "features": request.features,
        "target": request.target,
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "warnings": high_cardinality_warnings,
    })


# --- ROUTE 2: TRAIN CLASSIFICATION MODEL ---
@router.post("/train-model")
async def train_model(
    request: ClassificationTrainModelRequest, current_user: dict = Depends(get_current_user)
):
    perf = PerfTimer(f"classify-train-model ({request.model})")

    # 0. Reject a model the registry marks "unavailable" (its optional
    # package isn't installed) before touching the dataset at all — the
    # frontend already disables these as unselectable, but this is the
    # actual enforcement for a direct API call.
    for entry in list_classification_models():
        if entry["id"].lower() == request.model.strip().lower() and entry["status"] != "available":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=entry["reason"])

    with perf.stage("fetch_dataset (meta)"):
        dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])

    with perf.stage("build_dataframe"):
        df = await asyncio.to_thread(dataset_store.load_dataframe, request.dataset_id)

    all_selected = request.features + [request.target]
    for col in all_selected:
        if col not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{col}' not found in the dataset."
            )

    X = df[request.features]
    y = df[request.target]

    # Defensive re-check (select-features already validates this at
    # feature-selection time, but a client can call /train-model directly) —
    # a target with a single distinct value can't be classified at all.
    target_unique_count = int(y.nunique(dropna=True))
    if target_unique_count < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Classification requires at least two classes in the target column, but '{request.target}' has {target_unique_count}.",
        )

    logger.info(
        f"[classify-train-model] dataset_id={request.dataset_id} total_rows={len(df)} "
        f"features={request.features} target={request.target}"
    )

    test_size = request.split.test_size
    val_size = request.split.val_size
    random_state = request.split.random_state

    with perf.stage("train_test_split"):
        try:
            if val_size > 0:
                test_val_ratio = test_size + val_size
                X_train_val, X_test, y_train_val, y_test = train_test_split(
                    X, y, test_size=test_size / test_val_ratio, random_state=random_state, stratify=y,
                )
                val_ratio_in_train_val = val_size / test_val_ratio
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val, y_train_val, test_size=val_ratio_in_train_val, random_state=random_state,
                    stratify=y_train_val,
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y,
                )
                X_val, y_val = None, None
        except Exception as e:
            # stratify=y raises if any class has fewer members than the
            # number of splits it needs to appear in (e.g. a class with a
            # single row) — fall back to an unstratified split rather than
            # rejecting a legitimately tiny/imbalanced dataset outright.
            try:
                if val_size > 0:
                    test_val_ratio = test_size + val_size
                    X_train_val, X_test, y_train_val, y_test = train_test_split(
                        X, y, test_size=test_size / test_val_ratio, random_state=random_state,
                    )
                    val_ratio_in_train_val = val_size / test_val_ratio
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val, y_train_val, test_size=val_ratio_in_train_val, random_state=random_state,
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state,
                    )
                    X_val, y_val = None, None
            except Exception as e2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Data split failed. Typically caused by too few rows: {e2}"
                )

    logger.info(
        f"[classify-train-model] dataset_id={request.dataset_id} total_rows={len(df)} "
        f"train_rows={len(X_train)} test_rows={len(X_test)} "
        f"val_rows={len(X_val) if X_val is not None else 0}"
    )

    with perf.stage("classify_features"):
        classification = classify_columns(df)
        numeric_features = [f for f in request.features if classification[f]["kind"] == "numerical"]
        categorical_features = [f for f in request.features if classification[f]["kind"] == "categorical"]
        scaling_method = dataset.get("preprocessing_config", {}).get("scaling", "none")

    with perf.stage("model_fit"):
        try:
            pipeline = build_classification_pipeline(
                request.model, request.hyperparameters,
                numeric_features, categorical_features, scaling_method,
            )
            pipeline.fit(X_train, y_train)
        except (ImportError, ModuleNotFoundError):
            # XGBoost/LightGBM/CatBoost are imported lazily inside
            # get_classification_model() specifically so a deployment
            # without those packages installed doesn't crash on startup —
            # this is where that trade-off surfaces, as a curated message
            # instead of a raw "No module named 'xgboost'". The registry
            # guard above should already have caught this earlier, but this
            # stays as a defensive backstop.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{request.model} is unavailable because a required package is not installed. "
                "Please choose another algorithm.",
            )
        except ValueError as e:
            msg = str(e)
            msg_lower = msg.lower()
            # get_classification_model() itself raises this for a genuinely
            # unsupported model name, with an already curated, specific
            # message — surface it directly rather than reinterpreting it.
            if "unsupported model type" in msg_lower:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=msg)
            # sklearn's actual phrasing when a random (non-stratified) split
            # happened to put too few examples of some class into the
            # training split for the model to fit — matched narrowly on its
            # real wording so this never shadows an unrelated ValueError
            # (e.g. "unsupported model type ... select one of ...", which
            # also contains generic words like "class"/"one").
            if "least populated class" in msg_lower or "n_splits" in msg_lower:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Training could not start because one or more classes don't have enough samples "
                    "in the training split. Please provide more samples for that class, or choose a different "
                    "test/train split ratio.",
                )
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Model fitting failed: {e}"
            )
        preprocessor = pipeline.named_steps["preprocessor"]
        model_instance = pipeline.named_steps["model"]
        # Sourced from the FITTED model, not an independently pre-split
        # sorted list — this is the exact ordering predict_proba()'s columns
        # follow, so confusion-matrix/ROC-AUC label alignment is guaranteed
        # correct. A class entirely absent from the training split (and so
        # missing from classes_) is an inherent limitation of any classifier
        # that never saw it during fit, not something to work around here.
        classes = list(model_instance.classes_)

    with perf.stage("predict_test + metrics"):
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test) if hasattr(model_instance, "predict_proba") else None
        metrics = calculate_classification_metrics(
            y_true=y_test.values,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
            classes=classes,
        )

    with perf.stage("statsmodels_logit"):
        try:
            feature_names_out = list(preprocessor.get_feature_names_out())
            X_train_transformed = preprocessor.transform(X_train)
            X_train_named = pd.DataFrame(X_train_transformed, columns=feature_names_out, index=X_train.index)
            stats_properties = calculate_classification_statistical_properties(X_train_named, y_train, classes)
        except Exception as e:
            # Same rationale as training_routes.py's regression counterpart:
            # an unhandled exception here bypasses CORSMiddleware, and the
            # browser reports it as a misleading CORS failure instead of the
            # real error. Logistic MLE (Logit/MNLogit) fails to converge far
            # more often than OLS's closed-form solve, so this path is more
            # likely to be hit here than in the regression pipeline.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Statistical analysis failed: {e}"
            )

    model_id = str(uuid.uuid4())[:8]
    with perf.stage("save_model_pkl"):
        try:
            save_model_package(
                model_id=model_id,
                model=model_instance,
                features=request.features,
                target=request.target,
                preprocessor=preprocessor,
                numeric_features=numeric_features,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to serialize/save trained model: {e}"
            )

    # PNG diagnostic charts (confusion matrix / ROC / PR) are deferred to
    # ensure_graphs(), same lazy-generation pattern as regression — see
    # training_routes.py's train_model() for the full profiling rationale.
    graph_paths: Dict[str, str] = {}

    with perf.stage("chart_data + correlation"):
        y_test_actual_list = [str(v) for v in np.asarray(y_test.values).tolist()]
        y_test_pred_list = [str(v) for v in np.asarray(y_test_pred).tolist()]

        encoded_categorical_names: List[str] = []
        if categorical_features:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            encoded_categorical_names = list(cat_encoder.get_feature_names_out(categorical_features))
        feature_importance = map_expanded_coefficients(
            stats_properties["coefficients"], numeric_features, categorical_features, encoded_categorical_names,
        )

        corr_cols = [c for c in request.features if pd.api.types.is_numeric_dtype(df[c])]
        correlation_matrix = {"columns": [], "matrix": []}
        if len(corr_cols) > 1:
            corr_df = df[corr_cols].corr().round(4).fillna(0)
            correlation_matrix = {"columns": corr_df.columns.tolist(), "matrix": corr_df.values.tolist()}

        chart_data = {
            "actual": y_test_actual_list,
            "predicted": y_test_pred_list,
            "feature_importance": feature_importance,
            "correlation_matrix": correlation_matrix,
            "confusion_matrix": {
                "classes": [str(c) for c in classes],
                "matrix": metrics["ConfusionMatrix"],
            },
        }

    # Dedicated data for the Classification Visualization dashboard — see
    # classification_evaluation.py's helpers for why this reads the fitted
    # sklearn model's own signals instead of reusing the statsmodels-derived
    # `chart_data["feature_importance"]` above. Every field is independently
    # optional (None when unsupported by this model/dataset combination);
    # the frontend renders a "not available" message per-field rather than
    # requiring the whole object to be complete.
    with perf.stage("classification_visualizations"):
        visualizations = {
            "class_distribution": extract_class_distribution(y, classes),
            "confusion_matrix": {
                "labels": [str(c) for c in classes],
                "matrix": metrics["ConfusionMatrix"],
            },
            "feature_importance": extract_feature_importance(
                model_instance, feature_names_out, numeric_features, categorical_features, encoded_categorical_names,
            ),
            "roc_curve": extract_roc_curve(y_test.values, y_test_proba, classes),
            "precision_recall_curve": extract_precision_recall_curve(y_test.values, y_test_proba, classes),
        }

    with perf.stage("sanitize_floats"):
        metrics = sanitize_floats(metrics)
        stats_properties = sanitize_floats(stats_properties)
        chart_data = sanitize_floats(chart_data)
        visualizations = sanitize_floats(visualizations)

    categorical_options = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in categorical_features
    }

    model_doc = {
        "_id": model_id,
        "user_id": current_user["_id"],
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "model": request.model,
        "model_type": "classification",
        "classes": [str(c) for c in classes],
        "features": request.features,
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_options": categorical_options,
        "target": request.target,
        "split": request.split.model_dump(),
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "val_rows": len(X_val) if X_val is not None else 0,
        "metrics": metrics,
        "statistical_analysis": stats_properties,
        "chart_data": chart_data,
        "visualizations": visualizations,
        "graph_paths": graph_paths,
        "status": "completed",
        "created_at": pd.Timestamp.now().isoformat(),
    }

    with perf.stage("db_insert (json write)"):
        await db_client.insert_one("models", model_doc)

    perf.report()

    return ok({
        "status": "completed",
        "model": request.model,
        "model_id": model_id,
        "metrics": metrics,
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "val_rows": len(X_val) if X_val is not None else 0,
    }, message="Classification model trained successfully.")
