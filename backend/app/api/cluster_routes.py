import asyncio
import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from app.auth.dependencies import get_current_user
from app.database.database import db_client
from app.datasets import store as dataset_store
from app.ml.feature_types import classify_columns
from app.ml.clustering_models import build_clustering_pipeline
from app.ml.clustering_registry import list_clustering_models, get_clustering_model_entry
from app.ml.clustering_evaluation import (
    calculate_clustering_metrics,
    compute_cluster_sizes,
    compute_cluster_profiles,
    compute_pca_projection,
    compute_silhouette_plot,
    compute_elbow_data,
    compute_dendrogram_data,
)
from app.utils.response import ok, sanitize_floats
from app.utils.perf import PerfTimer
from app.api.training_routes import _get_owned_dataset

logger = logging.getLogger("regression_studio.cluster_routes")

router = APIRouter(prefix="/cluster", tags=["Clustering Training & Evaluation"])


class ClusteringFeatureSelectionRequest(BaseModel):
    dataset_id: str
    features: List[str]


class ClusteringTrainModelRequest(BaseModel):
    dataset_id: str
    model: str = Field("K-Means", description="Name of clustering algorithm")
    features: List[str]
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


async def _get_owned_clustering_model(model_id: str, user_id: str) -> dict:
    model_doc = await db_client.find_one("models", {"_id": model_id})
    if not model_doc or model_doc.get("user_id") != user_id or model_doc.get("model_type") != "clustering":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clustering model with ID {model_id} not found.",
        )
    return model_doc


# --- ROUTE: LIST CLUSTERING MODELS (registry-backed catalog + live availability) ---
@router.get("/clustering-models")
async def get_clustering_models_route(current_user: dict = Depends(get_current_user)):
    return ok(list_clustering_models())


# --- ROUTE 1: FEATURE SELECTION (no target — clustering is unsupervised) ---
@router.post("/select-features")
async def select_features(
    request: ClusteringFeatureSelectionRequest, current_user: dict = Depends(get_current_user)
):
    await _get_owned_dataset(request.dataset_id, current_user["_id"])
    df = await asyncio.to_thread(dataset_store.load_dataframe, request.dataset_id)

    if not request.features:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Select at least one input feature to run clustering.",
        )

    missing_feats = [f for f in request.features if f not in df.columns]
    if missing_feats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature columns {missing_feats} do not exist in the dataset.",
        )

    classification = classify_columns(df)

    datetime_feats = [f for f in request.features if classification[f]["kind"] == "datetime"]
    if datetime_feats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns {datetime_feats} look like date/time values and can't be used as clustering features directly.",
        )

    numeric_features = [f for f in request.features if classification[f]["kind"] == "numerical"]
    categorical_features = [f for f in request.features if classification[f]["kind"] == "categorical"]

    missing_counts = {f: int(df[f].isna().sum()) for f in request.features}
    warnings = []
    for f in request.features:
        if classification[f].get("is_identifier"):
            warnings.append(f"'{f}' appears to be an identifier. Consider excluding it from clustering.")
        elif classification[f].get("high_cardinality"):
            warnings.append(f"'{f}' contains many unique values. Consider excluding it from clustering.")

    return ok({
        "status": "validated",
        "features": request.features,
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "missing_counts": missing_counts,
        "warnings": warnings,
    })


# --- ROUTE 2: TRAIN CLUSTERING MODEL ---
@router.post("/train-model")
async def train_model(
    request: ClusteringTrainModelRequest, current_user: dict = Depends(get_current_user)
):
    perf = PerfTimer(f"cluster-train-model ({request.model})")

    # 0. Reject a model the registry marks "unavailable" (its optional
    # package isn't installed) before touching the dataset at all.
    for entry in list_clustering_models():
        if entry["id"].lower() == request.model.strip().lower() and entry["status"] != "available":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=entry["reason"])

    with perf.stage("fetch_dataset (meta)"):
        dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])

    with perf.stage("build_dataframe"):
        df = await asyncio.to_thread(dataset_store.load_dataframe, request.dataset_id)

    for col in request.features:
        if col not in df.columns:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Column '{col}' not found in the dataset.")

    if not request.features:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Select at least one input feature to run clustering.")

    # Clustering has no target/train-test split — the FULL selected feature
    # set (every row) is used to fit, matching "unsupervised learning uses
    # the complete dataset" and avoiding an arbitrary, meaningless split for
    # a task with no held-out label to validate against.
    X = df[request.features]
    total_rows = len(df)

    logger.info(f"[cluster-train-model] dataset_id={request.dataset_id} total_rows={total_rows} features={request.features}")

    with perf.stage("classify_features"):
        classification = classify_columns(df)
        numeric_features = [f for f in request.features if classification[f]["kind"] == "numerical"]
        categorical_features = [f for f in request.features if classification[f]["kind"] == "categorical"]
        scaling_method = dataset.get("preprocessing_config", {}).get("scaling", "standard")

    with perf.stage("model_fit"):
        try:
            pipeline = build_clustering_pipeline(
                request.model, request.hyperparameters, numeric_features, categorical_features, scaling_method,
            )
            labels = pipeline.fit_predict(X)
        except (ImportError, ModuleNotFoundError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{request.model} is unavailable because a required package is not installed. "
                "Please choose another algorithm.",
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Model fitting failed: {e}")

        preprocessor = pipeline.named_steps["preprocessor"]
        model_instance = pipeline.named_steps["model"]
        X_transformed = preprocessor.transform(X)
        labels = np.asarray(labels)
        feature_counts = {
            "raw_selected": len(request.features),
            "numeric": len(numeric_features),
            "categorical": len(categorical_features),
            "encoded_final": X_transformed.shape[1],
        }

    with perf.stage("metrics + analysis"):
        metrics = calculate_clustering_metrics(X_transformed, labels)
        cluster_sizes = compute_cluster_sizes(labels)
        cluster_profiles = compute_cluster_profiles(df[request.features], labels, numeric_features, categorical_features)

    registry_entry = get_clustering_model_entry(request.model) or {}
    with perf.stage("visualizations"):
        pca = compute_pca_projection(X_transformed, labels)
        silhouette_plot = compute_silhouette_plot(X_transformed, labels)
        elbow = compute_elbow_data(request.model, request.hyperparameters, X_transformed) if registry_entry.get("supports_elbow") else None
        dendrogram = compute_dendrogram_data(X_transformed) if registry_entry.get("supports_dendrogram") else None

        visualizations = {
            "pca": pca,
            "silhouette_plot": silhouette_plot,
            "elbow": elbow,
            "dendrogram": dendrogram,
            "cluster_sizes": {"labels": list(cluster_sizes.keys()), "values": list(cluster_sizes.values())},
        }

    with perf.stage("sanitize_floats"):
        metrics = sanitize_floats(metrics)
        cluster_profiles = sanitize_floats(cluster_profiles)
        visualizations = sanitize_floats(visualizations)

    model_id = str(uuid.uuid4())[:8]
    model_doc = {
        "_id": model_id,
        "user_id": current_user["_id"],
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "model": request.model,
        "model_type": "clustering",
        "algorithm_type": registry_entry.get("algorithm_type"),
        "features": request.features,
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "hyperparameters": request.hyperparameters or {},
        "total_rows": total_rows,
        "metrics": metrics,
        "cluster_sizes": cluster_sizes,
        "cluster_profiles": cluster_profiles,
        "visualizations": visualizations,
        "feature_counts": feature_counts,
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
        "cluster_sizes": cluster_sizes,
        "total_rows": total_rows,
    }, message="Clustering model trained successfully.")


# --- ROUTE 3: GET CLUSTERING MODEL DETAILS ---
@router.get("/model-metrics/{model_id}")
async def get_clustering_model_metrics(model_id: str, current_user: dict = Depends(get_current_user)):
    model_doc = await _get_owned_clustering_model(model_id, current_user["_id"])
    return ok(sanitize_floats({
        "model_id": model_doc["_id"],
        "model": model_doc["model"],
        "model_type": "clustering",
        "algorithm_type": model_doc.get("algorithm_type"),
        "dataset_name": model_doc.get("dataset_name"),
        "features": model_doc["features"],
        "numerical_features": model_doc.get("numerical_features", []),
        "categorical_features": model_doc.get("categorical_features", []),
        "hyperparameters": model_doc.get("hyperparameters", {}),
        "total_rows": model_doc.get("total_rows"),
        "metrics": model_doc["metrics"],
        "cluster_sizes": model_doc.get("cluster_sizes", {}),
        "cluster_profiles": model_doc.get("cluster_profiles", []),
        "visualizations": model_doc.get("visualizations", {}),
        "created_at": model_doc.get("created_at"),
    }))
