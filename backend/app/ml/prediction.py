import os
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "saved_models"))
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model_package(
    model_id: str,
    model: Any,
    features: list,
    target: str,
    preprocessor: Any = None,
    numeric_features: list = None,
) -> str:
    """
    Saves the model, feature order, fitted preprocessing transformer (a full
    impute+encode+scale ColumnTransformer for locally-trained models, or
    None for externally-sourced models like Hugging Face imports that never
    got one), and which of `features` are numeric (the rest are treated as
    categorical) — needed at prediction time to coerce each input value to
    the right type before it reaches the preprocessor.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    package = {
        "model": model,
        "features": features,
        "target": target,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features if numeric_features is not None else list(features),
    }
    joblib.dump(package, model_path)
    return model_path

def _load_package_and_transform(model_id: str, input_values: Dict[str, Any]):
    """
    Shared by predict_with_model() and predict_class_with_model(): loads the
    model package, builds a single-row DataFrame preserving each feature's
    original dtype (numeric values stay numeric, categorical values stay as
    their raw string), and applies the SAME fitted preprocessor used at
    training time (never refit here). Returns (model, X_transformed).
    """
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model package {model_id}.pkl not found in saved_models directory.")

    package = joblib.load(model_path)
    model = package["model"]
    features = package["features"]
    preprocessor = package["preprocessor"]
    numeric_features = set(package.get("numeric_features", features))

    # Build a single-row DataFrame in the original training column order —
    # unlike a bare `np.array([...]).reshape(1,-1)` construction, this
    # preserves per-column dtype (mixing a string city name with numeric age
    # in one untyped numpy array would silently upcast everything to one
    # dtype, corrupting the numeric values before they ever reach the
    # transformer). Numeric features are explicitly coerced to float (the
    # request body may carry them as JSON strings or numbers either way);
    # categorical features are passed through as their raw value, since a
    # fitted OneHotEncoder needs the real category strings, not floats.
    row: Dict[str, list] = {}
    for f in features:
        if f not in input_values:
            raise ValueError(f"Missing required feature: '{f}' in input values.")
        value = input_values[f]
        if f in numeric_features:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Feature '{f}' expects a numeric value, got '{value}'.")
        row[f] = [value]
    X = pd.DataFrame(row)[features]

    if preprocessor is not None:
        X = preprocessor.transform(X)

    return model, X


def predict_with_model(model_id: str, input_values: Dict[str, Any]) -> float:
    """
    Predicts a numeric target for a regression model. See
    _load_package_and_transform() for the shared row-construction logic.
    """
    model, X = _load_package_and_transform(model_id, input_values)
    prediction = model.predict(X)

    # Extract scalar value from prediction. Locally-trained models always
    # return a 1D array (fit on a pandas Series target), but externally
    # sourced models (e.g. Hugging Face) can return a 2D column vector
    # instead — reshape(-1) normalizes either case to a flat array before
    # indexing, since float() on a >0-d array raises in current numpy.
    pred_array = np.asarray(prediction).reshape(-1)
    return float(pred_array[0])


def predict_class_with_model(model_id: str, input_values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts a class label for a classification model, plus per-class
    probabilities where the underlying estimator supports predict_proba().

    Deliberately does NOT reuse predict_with_model()'s `float(pred_array[0])`
    cast — a classifier's predicted label can be a string ("yes"/"no") or an
    int class code (0/1/2); forcing it to float would crash on string labels
    outright and silently discard discrete-class semantics on int ones.
    """
    model, X = _load_package_and_transform(model_id, input_values)

    raw_pred = np.asarray(model.predict(X)).reshape(-1)
    prediction = str(raw_pred[0])

    probabilities = None
    if hasattr(model, "predict_proba"):
        proba_row = np.asarray(model.predict_proba(X)).reshape(-1)
        probabilities = {
            str(cls): round(float(p), 6) for cls, p in zip(model.classes_, proba_row)
        }

    return {"prediction": prediction, "probabilities": probabilities}
