import importlib.util
from typing import Any, Dict, List, Optional, TypedDict

# Single source of truth for "what regression models does this app offer,
# and is each one actually usable right now" — the frontend's model-selection
# page renders directly off this list instead of hard-coding its own catalog,
# so a model can never be shown as selectable without a real, working
# implementation behind it (see get_regression_model() in
# regression_models.py for the actual training dispatch, which this registry
# describes but doesn't replace).


class RegressionModelEntry(TypedDict):
    id: str  # exact string sent as TrainModelRequest.model / get_regression_model()'s model_name
    display_name: str
    category: str
    description: str
    pros: List[str]
    requires_package: Optional[str]  # importable module name, if any
    implemented: bool  # False only for the time-series models (no pipeline exists yet)


_REGISTRY: List[RegressionModelEntry] = [
    # ── Basic Regression ─────────────────────────────────────────────────
    {
        "id": "Linear Regression",
        "display_name": "Linear Regression",
        "category": "Basic Regression",
        "description": "Fits a straight line to model the relationship between one input feature and the target.",
        "pros": ["Simple", "Fast", "Interpretable"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Multiple Linear Regression",
        "display_name": "Multiple Linear Regression",
        "category": "Basic Regression",
        "description": "Linear model with multiple input features.",
        "pros": ["Multi-feature", "Interpretable"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Polynomial Regression",
        "display_name": "Polynomial Regression",
        "category": "Basic Regression",
        "description": "Fits nonlinear polynomial curves over the numerical input features.",
        "pros": ["Handles curves", "Flexible"],
        "requires_package": None,
        "implemented": True,
    },
    # ── Regularization ───────────────────────────────────────────────────
    {
        "id": "Ridge Regression",
        "display_name": "Ridge Regression",
        "category": "Regularization",
        "description": "Linear regression with L2 regularization.",
        "pros": ["Reduces overfit", "Stable"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Lasso Regression",
        "display_name": "Lasso Regression",
        "category": "Regularization",
        "description": "Linear regression with L1 regularization.",
        "pros": ["Feature selection", "Sparse"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Elastic Net Regression",
        "display_name": "Elastic Net Regression",
        "category": "Regularization",
        "description": "Combines L1 and L2 regularization.",
        "pros": ["Balanced", "Robust"],
        "requires_package": None,
        "implemented": True,
    },
    # ── Tree Based ────────────────────────────────────────────────────────
    {
        "id": "Decision Tree Regression",
        "display_name": "Decision Tree Regression",
        "category": "Tree Based",
        "description": "Tree-based nonlinear regressor.",
        "pros": ["Nonlinear", "No scaling"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Random Forest Regression",
        "display_name": "Random Forest Regression",
        "category": "Tree Based",
        "description": "Ensemble of decision trees.",
        "pros": ["High accuracy", "Robust"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Gradient Boosting",
        "display_name": "Gradient Boosting",
        "category": "Tree Based",
        "description": "Sequentially boosted ensemble of shallow trees.",
        "pros": ["High accuracy", "Handles mixed data"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "XGBoost",
        "display_name": "XGBoost",
        "category": "Tree Based",
        "description": "Optimized, regularized gradient boosting.",
        "pros": ["State-of-the-art", "Fast"],
        "requires_package": "xgboost",
        "implemented": True,
    },
    {
        "id": "LightGBM",
        "display_name": "LightGBM",
        "category": "Tree Based",
        "description": "Histogram-based gradient boosting, built for scale.",
        "pros": ["Very fast", "Large datasets"],
        "requires_package": "lightgbm",
        "implemented": True,
    },
    {
        "id": "CatBoost",
        "display_name": "CatBoost",
        "category": "Tree Based",
        "description": "Gradient boosting with strong out-of-the-box handling of categorical features.",
        "pros": ["Low tuning effort", "Strong on categoricals"],
        "requires_package": "catboost",
        "implemented": True,
    },
    # ── Other ────────────────────────────────────────────────────────────
    {
        "id": "Support Vector Regression",
        "display_name": "Support Vector Regression",
        "category": "Other",
        "description": "SVM adapted for regression tasks.",
        "pros": ["Kernel tricks", "Robust"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "KNN Regression",
        "display_name": "KNN Regression",
        "category": "Other",
        "description": "Predicts from the average of nearest neighbors.",
        "pros": ["Simple", "No training phase"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Bayesian Regression",
        "display_name": "Bayesian Regression",
        "category": "Other",
        "description": "Linear regression with a Bayesian prior over weights.",
        "pros": ["Uncertainty-aware", "Regularized"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Huber Regression",
        "display_name": "Huber Regression",
        "category": "Other",
        "description": "Linear regression robust to outliers.",
        "pros": ["Outlier-robust", "Interpretable"],
        "requires_package": None,
        "implemented": True,
    },
    {
        "id": "Quantile Regression",
        "display_name": "Quantile Regression",
        "category": "Other",
        "description": "Models a specific conditional quantile of the target.",
        "pros": ["Robust to outliers", "Distributional insight"],
        "requires_package": None,
        "implemented": True,
    },
    # ── Time Series — no dedicated pipeline exists yet (see
    # regression_compatibility.py / classify_routes.py for why these can't
    # just run through the tabular Pipeline: they need a datetime-indexed
    # target, chronological splitting, and forecast-horizon handling that
    # nothing in this app currently builds). Listed so they're visible and
    # correctly categorized rather than silently absent, but never
    # selectable — "coming_soon" is enforced both here and by the
    # /train-model endpoint refusing to train them. ─────────────────────
    {
        "id": "ARIMA",
        "display_name": "ARIMA",
        "category": "Time Series",
        "description": "Autoregressive Integrated Moving Average — classical univariate time-series forecasting.",
        "pros": ["No external deps", "Interpretable"],
        "requires_package": None,
        "implemented": False,
    },
    {
        "id": "SARIMA",
        "display_name": "SARIMA",
        "category": "Time Series",
        "description": "ARIMA with an explicit seasonal component.",
        "pros": ["Handles seasonality"],
        "requires_package": None,
        "implemented": False,
    },
    {
        "id": "Prophet",
        "display_name": "Prophet",
        "category": "Time Series",
        "description": "Additive forecasting model designed for business time series with strong seasonality/holiday effects.",
        "pros": ["Handles holidays", "Robust to missing data"],
        "requires_package": "prophet",
        "implemented": False,
    },
    {
        "id": "LSTM Regression",
        "display_name": "LSTM Regression",
        "category": "Time Series",
        "description": "Recurrent neural network forecaster for sequential data.",
        "pros": ["Captures long-range patterns"],
        "requires_package": "tensorflow",
        "implemented": False,
    },
    {
        "id": "Transformer Forecasting",
        "display_name": "Transformer Forecasting",
        "category": "Time Series",
        "description": "Attention-based sequence model applied to forecasting.",
        "pros": ["State-of-the-art on long sequences"],
        "requires_package": "torch",
        "implemented": False,
    },
]


def _package_installed(package_name: str) -> bool:
    """Cheap availability check via the import machinery's finder, without
    actually importing the module — xgboost/lightgbm/catboost are
    intentionally imported lazily inside get_regression_model() only when a
    model of that type is actually trained (see that function's docstring),
    so this must not force an eager import just to answer "is it there"."""
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def list_regression_models() -> List[Dict[str, Any]]:
    """Returns the full catalog with a live-computed status for each entry —
    'available' (ready to train), 'unavailable' (implemented, but its
    optional package isn't installed), or 'coming_soon' (no pipeline exists
    yet). Never reports a model as available without actually being able to
    train it.
    """
    results: List[Dict[str, Any]] = []
    for entry in _REGISTRY:
        if not entry["implemented"]:
            status = "coming_soon"
            reason = "Time-series regression is not yet available in this app."
        elif entry["requires_package"] and not _package_installed(entry["requires_package"]):
            status = "unavailable"
            reason = f"Model unavailable: the '{entry['requires_package']}' package is not installed."
        else:
            status = "available"
            reason = None

        results.append({
            "id": entry["id"],
            "display_name": entry["display_name"],
            "category": entry["category"],
            "description": entry["description"],
            "pros": entry["pros"],
            "status": status,
            "reason": reason,
        })
    return results
