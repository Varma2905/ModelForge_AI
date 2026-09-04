import asyncio
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

from app.auth.dependencies import get_current_user
from app.database.database import db_client
from app.datasets import store as dataset_store
from app.ml.feature_types import (
    classify_columns,
    map_expanded_coefficients,
    detect_identifier_target,
    detect_target_leakage,
    extract_feature_importance,
)
from app.ml.regression_models import build_pipeline
from app.ml.regression_compatibility import check_regression_model_compatibility
from app.ml.regression_registry import list_regression_models
from app.ml.evaluation import calculate_evaluation_metrics, calculate_statistical_properties
from app.ml.prediction import save_model_package, MODELS_DIR
from app.visualization.graph_generator import generate_dataset_graphs, generate_regression_graphs
from app.utils.response import ok, sanitize_floats
from app.utils.sampling import sample_paired_series
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
    dataset = dataset_store.load_meta(dataset_id)
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


async def ensure_graphs(model_doc: dict, preloaded_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Generates and persists the PNG diagnostic charts for a trained model
    if they haven't been generated yet (see the note in train_model() for
    why this is deferred out of the training request). Reloads the saved
    model package and the source dataset to reproduce exactly what training
    would have plotted. Called from report_routes.py — the only two callers
    that actually need these PNGs (the PDF report and the /graphs endpoint).

    `preloaded_df` lets a caller that already loaded the dataset for another
    reason (report_routes.py's dataset profiling step) hand it over instead
    of this function loading it again from disk — purely an I/O dedup, the
    dataframe content and everything plotted from it is unchanged either way.
    """
    if model_doc.get("graph_paths"):
        return model_doc["graph_paths"]

    model_id = model_doc["_id"]
    dataset_id = model_doc.get("dataset_id")
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    
    # Clustering models don't write a .pkl package because they are fast to rebuild and fit
    is_clustering = model_doc.get("model_type") == "clustering"
    if not dataset_id or dataset_store.load_meta(dataset_id) is None or (not is_clustering and not os.path.exists(model_path)):
        return {}

    def _build_graphs() -> Dict[str, str]:
        df = preloaded_df if preloaded_df is not None else dataset_store.load_dataframe(dataset_id)
        features = model_doc["features"]
        numeric_features = model_doc.get("numerical_features", features)
        categorical_features = model_doc.get("categorical_features", [])

        if is_clustering:
            from app.ml.clustering_models import build_clustering_pipeline
            from app.visualization.graph_generator import generate_clustering_graphs
            
            dataset_meta = dataset_store.load_meta(dataset_id) or {}
            scaling_method = dataset_meta.get("preprocessing_config", {}).get("scaling", "standard")

            pipeline = build_clustering_pipeline(
                model_doc["model"],
                model_doc.get("hyperparameters"),
                numeric_features,
                categorical_features,
                scaling_method
            )
            labels = pipeline.fit_predict(df[features])
            X_transformed = pipeline.named_steps["preprocessor"].transform(df[features])

            ds_paths = generate_dataset_graphs(df[features], None, model_id, target_series=None)
            task_paths = generate_clustering_graphs(
                X_transformed=X_transformed,
                labels=labels,
                model_name=model_doc["model"],
                visualizations=model_doc,
                model_id=model_id
            )
            return {**ds_paths, **task_paths}

        package = joblib.load(model_path)
        model_instance = package["model"]
        preprocessor = package["preprocessor"]
        target = model_doc["target"]

        X = df[features]
        # NOT wrapped back into a DataFrame with X.columns — unlike a plain
        # scaler, a fitted ColumnTransformer with OneHotEncoder expands the
        # column count (one-hot columns), so the old
        # `pd.DataFrame(scaler.transform(X), columns=X.columns, ...)` pattern
        # would raise a column-count mismatch. The model was fit on this same
        # transformed array shape, so predicting straight off it is correct.
        X_for_predict = preprocessor.transform(X) if preprocessor is not None else X
        y_pred = model_instance.predict(X_for_predict)

        encoded_categorical_names: List[str] = []
        if categorical_features and preprocessor is not None:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            encoded_categorical_names = list(cat_encoder.get_feature_names_out(categorical_features))

        # The trained model's OWN feature_importances_/coef_ (None for KNN,
        # non-linear-kernel SVM/SVR, etc.) — preferred over the auxiliary
        # OLS/Logit fit's coefficients for the PDF's Feature Importance
        # chart, see generate_regression_graphs()/generate_classification_
        # graphs()'s real_importance parameter.
        real_importance = None
        if preprocessor is not None:
            try:
                real_importance = extract_feature_importance(
                    model_instance, list(preprocessor.get_feature_names_out()),
                    numeric_features, categorical_features, encoded_categorical_names,
                )
            except Exception as e:
                logger.warning(f"Real feature importance extraction failed for model {model_id}: {e}")

        is_classification = model_doc.get("model_type", "regression") == "classification"
        # Classification's EDA section draws from the FULL original dataset
        # (not just the narrow subset of columns selected as model inputs) —
        # a model trained on 1-2 features would otherwise leave the report's
        # Exploratory Data Analysis section with little or nothing to show.
        # Regression's behavior is untouched (still `df[features]` only).
        ds_paths = generate_dataset_graphs(
            df if is_classification else df[features],
            target, model_id, target_series=df[target],
            include_feature_vs_target=is_classification,
        )

        if is_classification:
            from app.visualization.graph_generator import generate_classification_graphs
            y_proba = (
                model_instance.predict_proba(X_for_predict)
                if hasattr(model_instance, "predict_proba") else None
            )
            classes = model_doc.get("classes") or sorted(set(str(c) for c in df[target].dropna().unique()))
            task_paths = generate_classification_graphs(
                y_actual=df[target].astype(str).values,
                y_predicted=y_pred,
                y_proba=y_proba,
                classes=classes,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                encoded_categorical_names=encoded_categorical_names,
                coefficients=model_doc["statistical_analysis"]["coefficients"],
                model_id=model_id,
                real_importance=real_importance,
            )
        else:
            task_paths = generate_regression_graphs(
                y_actual=df[target].values,
                y_predicted=y_pred,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                encoded_categorical_names=encoded_categorical_names,
                coefficients=model_doc["statistical_analysis"]["coefficients"],
                model_id=model_id,
                real_importance=real_importance,
            )
        return {**ds_paths, **task_paths}

    try:
        # CPU-bound (joblib load + matplotlib rendering) — run off the event
        # loop so it doesn't block other requests while it works.
        graph_paths = await asyncio.to_thread(_build_graphs)
    except Exception as e:
        logger.warning(f"Deferred graph generation failed for model {model_id}: {e}")
        return {}

    await db_client.update_one("models", {"_id": model_id}, {"$set": {"graph_paths": graph_paths}})
    return graph_paths


# --- ROUTE: LIST REGRESSION MODELS (registry-backed catalog + live availability) ---
@router.get("/regression-models")
async def get_regression_models(current_user: dict = Depends(get_current_user)):
    return ok(list_regression_models())


# --- ROUTE 1: FEATURE SELECTION ---
@router.post("/select-features")
async def select_features(
    request: FeatureSelectionRequest, current_user: dict = Depends(get_current_user)
):
    dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])
    df = await asyncio.to_thread(dataset_store.load_dataframe, request.dataset_id)

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

    # 3. Target must be numerical — this is a regression problem, and we
    # never silently coerce a categorical target into numbers just to force
    # regression to "work". Features, by contrast, may be numerical OR
    # categorical (handled below) — only the target is gated here.
    classification = classify_columns(df)
    if classification[request.target]["kind"] != "numerical":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Regression requires a numerical target variable. Please select a numerical column.",
        )

    # 4. Features: numerical and categorical are both allowed; datetime
    # columns are rejected outright (no date-part feature engineering here —
    # a raw datetime can't go through impute+one-hot meaningfully).
    datetime_feats = [f for f in request.features if classification[f]["kind"] == "datetime"]
    if datetime_feats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Columns {datetime_feats} look like date/time values and can't be used as regression features directly.",
        )

    numeric_features = [f for f in request.features if classification[f]["kind"] == "numerical"]
    categorical_features = [f for f in request.features if classification[f]["kind"] == "categorical"]

    # 5. Missing values: the training pipeline's SimpleImputer safely
    # handles NaNs in FEATURE columns (median for numeric, most-frequent for
    # categorical) — no need to force preprocessing first. The TARGET never
    # passes through that imputer, so it must still be complete.
    target_missing = int(df[request.target].isna().sum())
    if target_missing > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target column '{request.target}' has {target_missing} missing value(s). "
            "Please apply missing-value preprocessing first.",
        )

    warnings: List[str] = []
    for f in request.features:
        if classification[f].get("is_identifier"):
            warnings.append(f"'{f}' appears to be an identifier. Consider excluding it from the model.")
        elif classification[f].get("high_cardinality"):
            warnings.append(f"'{f}' appears to be an identifier and contains many unique values. Consider excluding it from the model.")

    identifier_target_warning = detect_identifier_target(request.target, classification)
    if identifier_target_warning:
        warnings.append(identifier_target_warning)

    leakage_warnings = detect_target_leakage(df, request.target, request.features, classification)
    warnings.extend(w["detail"] for w in leakage_warnings)

    return ok({
        "status": "validated",
        "features": request.features,
        "target": request.target,
        "numerical_features": numeric_features,
        "categorical_features": categorical_features,
        "warnings": warnings,
        "leakage_warnings": leakage_warnings,
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

    # 0. Reject a model the registry marks "coming_soon" (currently the
    # time-series models — ARIMA/SARIMA/Prophet/LSTM/Transformer) or
    # "unavailable" (an implemented model whose optional package isn't
    # installed) before touching the dataset at all. The frontend already
    # disables these as unselectable, but this is the actual enforcement —
    # never let a direct API call train something this app can't really run.
    for entry in list_regression_models():
        if entry["id"].lower() == request.model.strip().lower() and entry["status"] != "available":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=entry["reason"] or f"{request.model} is not currently available.",
            )

    # 1. Fetch dataset
    with perf.stage("fetch_dataset (meta)"):
        dataset = await _get_owned_dataset(request.dataset_id, current_user["_id"])

    with perf.stage("build_dataframe"):
        df = await asyncio.to_thread(dataset_store.load_dataframe, request.dataset_id)

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

    # The full dataset (no row sampling/truncation) always goes into training —
    # logged here so it's verifiable that "rows used for ML" == the dataset's
    # actual row count, not some hardcoded preview-sized subset.
    logger.info(
        f"[train-model] dataset_id={request.dataset_id} total_rows={len(df)} "
        f"features={request.features} target={request.target}"
    )

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

    logger.info(
        f"[train-model] dataset_id={request.dataset_id} total_rows={len(df)} "
        f"train_rows={len(X_train)} test_rows={len(X_test)} "
        f"val_rows={len(X_val) if X_val is not None else 0}"
    )

    # 3.5 Classify the selected features and build the leakage-safe
    # preprocessing + model Pipeline. Scaling method is whatever was chosen
    # during /preprocess (stored on the dataset meta); if the caller trained
    # straight off a raw (non-preprocessed) dataset, no scaling is applied.
    with perf.stage("classify_features"):
        classification = classify_columns(df)
        numeric_features = [f for f in request.features if classification[f]["kind"] == "numerical"]
        categorical_features = [f for f in request.features if classification[f]["kind"] == "categorical"]
        scaling_method = dataset.get("preprocessing_config", {}).get("scaling", "none")

        identifier_warnings: List[str] = []
        for f in request.features:
            if classification[f].get("is_identifier"):
                identifier_warnings.append(f"'{f}' appears to be an identifier. Consider excluding it from the model.")
        identifier_target_warning = detect_identifier_target(request.target, classification)
        if identifier_target_warning:
            identifier_warnings.append(identifier_target_warning)
        leakage_warnings = detect_target_leakage(df, request.target, request.features, classification)

    # 3.6 Reject a model/feature-selection combination that can't
    # meaningfully be trained (e.g. Simple Linear Regression with more than
    # one feature, or Polynomial Regression with zero numerical features)
    # BEFORE spending time fitting it — see regression_compatibility.py for
    # why every other model needs no extra check here.
    incompatibility_reason = check_regression_model_compatibility(
        request.model, request.features, numeric_features, categorical_features,
    )
    if incompatibility_reason:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=incompatibility_reason)

    # 4. Build + fit the full pipeline. Pipeline.fit() fits the
    # preprocessor (imputers/encoder/scaler) AND the model together on
    # X_train ONLY — .predict()/.transform() on test/val reuses that fitted
    # state without ever re-fitting on it, which is exactly the
    # leakage-safe shape the old standalone-scaler code already had.
    with perf.stage("model_fit"):
        try:
            pipeline = build_pipeline(
                request.model, request.hyperparameters,
                numeric_features, categorical_features, scaling_method,
            )
            pipeline.fit(X_train, y_train)
        except (ImportError, ModuleNotFoundError) as e:
            # XGBoost/LightGBM are imported lazily inside get_regression_model()
            # specifically so a deployment without those packages installed
            # doesn't crash on startup — this is where that trade-off surfaces,
            # as a curated message instead of a raw "No module named 'xgboost'".
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{request.model} is unavailable because a required package is not installed. "
                "Please choose another algorithm.",
            )
        except ValueError as e:
            # get_regression_model()/build_pipeline() raise ValueError for a
            # genuinely unsupported model name or an invalid hyperparameter
            # (e.g. polynomial degree out of range) with an already
            # curated, specific message — surface it directly rather than
            # burying it in the generic "Model fitting failed" prefix below.
            msg = str(e)
            if "unsupported model type" in msg.lower():
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=msg)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Model fitting failed: {e}"
            )
        preprocessor = pipeline.named_steps["preprocessor"]
        model_instance = pipeline.named_steps["model"]

    # 5. Predictions & Evaluation Metrics (on the held-out test split)
    with perf.stage("predict_test + metrics"):
        y_test_pred = pipeline.predict(X_test)
        n_features_expanded = preprocessor.transform(X_test).shape[1]
        metrics = calculate_evaluation_metrics(
            y_true=y_test.values,
            y_pred=y_test_pred,
            n_samples=len(X_test),
            n_features=n_features_expanded,
        )
        # Raw (user-selected) vs. numeric/categorical split vs. final
        # one-hot-expanded model input width — persisted so reports can show
        # all three instead of conflating "features used" with "model
        # dimensions" (a categorical column with 10 categories expands to 10
        # model input columns from 1 selected feature).
        feature_counts = {
            "raw_selected": len(request.features),
            "numeric": len(numeric_features),
            "categorical": len(categorical_features),
            "encoded_final": n_features_expanded,
        }

    # 6. Statistical OLS properties (using Statsmodels) — fit on the SAME
    # transformed train matrix the model itself was fit on, reconstructed as
    # a DataFrame with the expanded (one-hot/scaled) feature names so
    # coefficients/p-values are keyed by real, attributable names instead of
    # meaningless "Feature_0, Feature_1, ..." placeholders.
    with perf.stage("statsmodels_ols"):
        try:
            feature_names_out = list(preprocessor.get_feature_names_out())
            X_train_transformed = preprocessor.transform(X_train)
            X_train_named = pd.DataFrame(X_train_transformed, columns=feature_names_out, index=X_train.index)
            stats_properties = calculate_statistical_properties(X_train_named, y_train)
        except Exception as e:
            # An unhandled exception here would propagate past FastAPI's
            # HTTPException handling entirely, which also makes
            # CORSMiddleware unable to attach headers to the resulting 500 —
            # the browser then misreports it as a CORS failure instead of
            # surfacing the real error. Always fail with a proper HTTP error.
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Statistical analysis failed: {e}"
            )

    # 7. Generate a unique model_id and save model package (.pkl), including
    # the fitted preprocessor so /predict applies the exact same
    # impute+encode+scale transform used at train time.
    model_id = str(uuid.uuid4())[:8]  # Keep ID readable
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

        # Maps expanded one-hot/scaled coefficient keys (e.g. "cat__city_Chennai")
        # back to a display label attributed to their source column ("city =
        # Chennai") — the old `k in request.features` filter silently dropped
        # every categorical-derived coefficient once keys stopped matching
        # original feature names exactly.
        # NOT prefixed with "cat__" here — map_expanded_coefficients strips
        # the ColumnTransformer's "num__"/"cat__" prefix off each
        # COEFFICIENT key internally before matching, so the lookup table
        # must be built from the same bare (unprefixed) names or every
        # lookup silently misses.
        encoded_categorical_names: List[str] = []
        if categorical_features:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            encoded_categorical_names = list(cat_encoder.get_feature_names_out(categorical_features))
        feature_importance = map_expanded_coefficients(
            stats_properties["coefficients"], numeric_features, categorical_features, encoded_categorical_names,
        )

        corr_cols = [c for c in all_selected if pd.api.types.is_numeric_dtype(df[c])]
        corr_df = df[corr_cols].corr().round(4).fillna(0)
        correlation_matrix = {
            "columns": corr_df.columns.tolist(),
            "matrix": corr_df.values.tolist()
        }

        # Interactive Recharts points are capped for RENDERING only — metrics
        # above were already computed on the full y_test/y_test_pred. Without
        # this, a large dataset's test split (e.g. 20k+ rows) would ship
        # every single point to the browser and to Recharts' SVG renderer,
        # which is the actual "Visualization is slow" cost for big datasets
        # (this page never re-trains or re-predicts — see ensure_graphs()'s
        # docstring for where PNG generation, the other slow path, lives).
        # Same convention as graph_generator.py's _MAX_ROWS_FOR_PLOTS.
        sampled_points = sample_paired_series(
            actual=y_test_actual_list, predicted=y_test_pred_list, residuals=residuals_list,
        )
        chart_data = {
            "actual": sampled_points["actual"],
            "predicted": sampled_points["predicted"],
            "residuals": sampled_points["residuals"],
            "chart_points_sampled": sampled_points["sampled"],
            "chart_points_total": sampled_points["total_size"],
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

    # Categorical dropdown options for the Predict page — the actual values
    # observed in the TRAINING split (not the full dataset, to stay
    # consistent with "the pipeline only ever learns from train"). Cast to
    # str for sorting safety: a manually-created dataset's categorical
    # column can legitimately mix types per cell.
    categorical_options = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in categorical_features
    }

    # Store metadata in DB
    model_doc = {
        "_id": model_id,
        "user_id": current_user["_id"],
        "dataset_id": request.dataset_id,
        "dataset_name": dataset["name"],
        "model": request.model,
        "model_type": "regression",
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
        "graph_paths": graph_paths,
        "feature_counts": feature_counts,
        "leakage_warnings": leakage_warnings,
        "identifier_warnings": identifier_warnings,
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
        "metrics": metrics,
        "total_rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "val_rows": len(X_val) if X_val is not None else 0,
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
        "model_type": model_doc.get("model_type", "regression"),
        "classes": model_doc.get("classes"),
        "features": model_doc["features"],
        "numerical_features": model_doc.get("numerical_features", model_doc["features"]),
        "categorical_features": model_doc.get("categorical_features", []),
        "categorical_options": model_doc.get("categorical_options", {}),
        "target": model_doc["target"],
        "metrics": model_doc["metrics"],
        "chart_data": model_doc.get("chart_data", {}),
        # Only present for classification models (see classify_routes.py
        # train_model()) — absent/None for regression, which never reads it.
        "visualizations": model_doc.get("visualizations"),
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

# --- ROUTE: DELETE MODEL (used by the Reports page) ---
@router.delete("/models/{model_id}")
async def delete_model(model_id: str, current_user: dict = Depends(get_current_user)):
    model_doc = await _get_owned_model(model_id, current_user["_id"])

    # Best-effort cleanup of on-disk artifacts — the DB record is the source
    # of truth for "does this model exist", so an already-missing file here
    # (e.g. graphs that were never lazily generated) is not an error.
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)

    for graph_path in (model_doc.get("graph_paths") or {}).values():
        if graph_path and os.path.exists(graph_path):
            os.remove(graph_path)

    static_reports_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "static", "reports")
    )
    report_pdf_path = os.path.join(static_reports_dir, f"report_{model_id}.pdf")
    if os.path.exists(report_pdf_path):
        os.remove(report_pdf_path)

    await db_client.delete_one("reports", {"_id": f"rep_{model_id}"})
    await db_client.delete_one("models", {"_id": model_id})

    return ok({"deleted": True, "model_id": model_id}, message="Model deleted.")


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
            "model_type": m.get("model_type", "regression"),
            "metrics": m["metrics"],
            "created_at": m.get("created_at"),
            "source": m.get("source"),
            "hf_model_id": m.get("hf_model_id"),
        } for m in sorted_models
    ]))
