import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "saved_models"))
os.makedirs(MODELS_DIR, exist_ok=True)

def save_model_package(model_id: str, model: Any, features: list, target: str, scaler: Any = None) -> str:
    """
    Saves the model, feature order, and scaling transformer in a single .pkl package.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    package = {
        "model": model,
        "features": features,
        "target": target,
        "scaler": scaler
    }
    joblib.dump(package, model_path)
    return model_path

def predict_with_model(model_id: str, input_values: Dict[str, float]) -> float:
    """
    Loads the model package, orders the input features correctly,
    applies scaling if applicable, and returns the prediction.
    """
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model package {model_id}.pkl not found in saved_models directory.")
        
    package = joblib.load(model_path)
    model = package["model"]
    features = package["features"]
    scaler = package["scaler"]
    
    # 1. Align input features in the correct order
    ordered_values = []
    for f in features:
        if f not in input_values:
            raise ValueError(f"Missing required feature: '{f}' in input values.")
        ordered_values.append(float(input_values[f]))
        
    # 2. Reshape into 2D array for prediction (1 sample, N features)
    X = np.array(ordered_values).reshape(1, -1)
    
    # 3. Apply scaling if it was saved
    if scaler is not None:
        # Scaler expects a DataFrame or array with the same feature count
        # In scikit-learn, standard scalers are fitted on the features X.
        X = scaler.transform(X)
        
    # 4. Predict
    prediction = model.predict(X)

    # Extract scalar value from prediction. Locally-trained models always
    # return a 1D array (fit on a pandas Series target), but externally
    # sourced models (e.g. Hugging Face) can return a 2D column vector
    # instead — reshape(-1) normalizes either case to a flat array before
    # indexing, since float() on a >0-d array raises in current numpy.
    pred_array = np.asarray(prediction).reshape(-1)
    return float(pred_array[0])
