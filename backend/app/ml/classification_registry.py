from typing import Any, Dict, List, Optional, TypedDict

from app.ml.regression_registry import _package_installed

# Single source of truth for "what classification models does this app
# offer, and is each one actually usable right now" — mirrors
# regression_registry.py exactly (see that file for the rationale). Every
# model listed here is a REAL, implemented pipeline (see
# get_classification_model() in classification_models.py) — there is no
# "coming_soon" tier for classification, since all 12 models are backed by
# already-installed or optionally-installed libraries, unlike regression's
# time-series models which have no pipeline at all.


class ClassificationModelEntry(TypedDict):
    id: str  # exact string sent as ClassificationTrainModelRequest.model
    display_name: str
    description: str
    pros: List[str]
    requires_package: Optional[str]


_REGISTRY: List[ClassificationModelEntry] = [
    {
        "id": "Logistic Regression",
        "display_name": "Logistic Regression",
        "description": "Models the log-odds of a class as a linear combination of features.",
        "pros": ["Fast", "Interpretable", "Probabilistic"],
        "requires_package": None,
    },
    {
        "id": "Decision Tree Classification",
        "display_name": "Decision Tree Classification",
        "description": "Tree-based classifier with fully interpretable decision paths.",
        "pros": ["Nonlinear", "No scaling", "Visualizable"],
        "requires_package": None,
    },
    {
        "id": "Random Forest Classification",
        "display_name": "Random Forest Classification",
        "description": "Ensemble of decision trees, resistant to overfitting.",
        "pros": ["High accuracy", "Robust"],
        "requires_package": None,
    },
    {
        "id": "Extra Trees Classification",
        "display_name": "Extra Trees Classification",
        "description": "Ensemble of randomized decision trees — like Random Forest, but with extra randomization in split thresholds for lower variance.",
        "pros": ["Fast to train", "Reduces overfitting"],
        "requires_package": None,
    },
    {
        "id": "AdaBoost Classification",
        "display_name": "AdaBoost Classification",
        "description": "Sequentially boosts weak learners, focusing each new one on the previous ensemble's mistakes.",
        "pros": ["Simple", "Few hyperparameters"],
        "requires_package": None,
    },
    {
        "id": "Gradient Boosting",
        "display_name": "Gradient Boosting",
        "description": "Sequentially boosted ensemble of shallow trees.",
        "pros": ["High accuracy", "Handles mixed data"],
        "requires_package": None,
    },
    {
        "id": "XGBoost",
        "display_name": "XGBoost",
        "description": "Optimized, regularized gradient boosting.",
        "pros": ["State-of-the-art", "Fast"],
        "requires_package": "xgboost",
    },
    {
        "id": "LightGBM",
        "display_name": "LightGBM",
        "description": "Histogram-based gradient boosting, built for scale.",
        "pros": ["Very fast", "Large datasets"],
        "requires_package": "lightgbm",
    },
    {
        "id": "CatBoost",
        "display_name": "CatBoost",
        "description": "Gradient boosting with strong out-of-the-box handling of categorical features.",
        "pros": ["Low tuning effort", "Strong on categoricals"],
        "requires_package": "catboost",
    },
    {
        "id": "SVM",
        "display_name": "SVM",
        "description": "Finds the maximum-margin boundary between classes.",
        "pros": ["Kernel tricks", "Robust"],
        "requires_package": None,
    },
    {
        "id": "KNN Classification",
        "display_name": "KNN Classification",
        "description": "Classifies by majority vote among nearest neighbors.",
        "pros": ["Simple", "No training phase"],
        "requires_package": None,
    },
    {
        "id": "Naive Bayes",
        "display_name": "Naive Bayes",
        "description": "Fast probabilistic classifier assuming feature independence.",
        "pros": ["Very fast", "Works with little data"],
        "requires_package": None,
    },
]


def list_classification_models() -> List[Dict[str, Any]]:
    """Returns the full catalog with a live-computed status for each entry —
    'available' or 'unavailable' (an optional package isn't installed).
    Never reports a model as available without actually being able to
    train it."""
    results: List[Dict[str, Any]] = []
    for entry in _REGISTRY:
        if entry["requires_package"] and not _package_installed(entry["requires_package"]):
            status = "unavailable"
            reason = f"{entry['display_name']} is unavailable because the required package ('{entry['requires_package']}') is not installed."
        else:
            status = "available"
            reason = None

        results.append({
            "id": entry["id"],
            "display_name": entry["display_name"],
            "description": entry["description"],
            "pros": entry["pros"],
            "status": status,
            "reason": reason,
        })
    return results
