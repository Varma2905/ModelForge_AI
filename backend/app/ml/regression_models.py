import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from typing import Any, Dict, Optional

def get_regression_model(model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Returns an un-fitted scikit-learn regression model based on the model name.
    
    Supported model names:
    - "LinearRegression" or "Linear" or "MultipleLinear"
    - "PolynomialRegression" or "Polynomial"
    - "Ridge" or "RidgeRegression"
    - "Lasso" or "LassoRegression"
    - "ElasticNet" or "ElasticNetRegression"
    - "DecisionTree" or "DecisionTreeRegressor"
    - "RandomForest" or "RandomForestRegressor"
    - "SVR" or "SupportVectorRegression"
    """
    if hyperparameters is None:
        hyperparameters = {}
        
    name_clean = model_name.replace(" ", "").lower()
    
    # 1 & 2. Linear and Multiple Linear Regression
    if name_clean in ["linearregression", "linear", "multiplelinear", "multiplelinearregression"]:
        return LinearRegression(**hyperparameters)
        
    # 3. Polynomial Regression
    elif name_clean in ["polynomialregression", "polynomial"]:
        degree = hyperparameters.get("degree", 2)
        # PolynomialFeatures output width is C(n_features + degree, degree) —
        # combinatorial in degree. An unvalidated high degree (e.g. a typo
        # like 50 instead of 5) can silently build a feature matrix with
        # millions of columns and hang or exhaust memory. 6 comfortably
        # covers legitimate curve-fitting use on the small/medium datasets
        # this app targets.
        if not isinstance(degree, int) or degree < 1 or degree > 6:
            raise ValueError(
                f"Polynomial degree must be an integer between 1 and 6 (got {degree}). "
                "Higher degrees create an unreasonably large feature matrix."
            )
        # Remove degree from hyperparameters so it doesn't get passed to LinearRegression
        lr_params = {k: v for k, v in hyperparameters.items() if k != "degree"}
        return Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear_regression", LinearRegression(**lr_params))
        ])
        
    # 4. Ridge Regression
    elif name_clean in ["ridge", "ridgeregression"]:
        alpha = hyperparameters.get("alpha", 1.0)
        return Ridge(alpha=alpha, **{k: v for k, v in hyperparameters.items() if k != "alpha"})
        
    # 5. Lasso Regression
    elif name_clean in ["lasso", "lassoregression"]:
        alpha = hyperparameters.get("alpha", 1.0)
        return Lasso(alpha=alpha, **{k: v for k, v in hyperparameters.items() if k != "alpha"})
        
    # 6. Elastic Net Regression
    elif name_clean in ["elasticnet", "elasticnetregression"]:
        alpha = hyperparameters.get("alpha", 1.0)
        l1_ratio = hyperparameters.get("l1_ratio", 0.5)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["alpha", "l1_ratio"]}
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **clean_params)
        
    # 7. Decision Tree Regression
    elif name_clean in ["decisiontree", "decisiontreeregressor", "decisiontreeregression"]:
        max_depth = hyperparameters.get("max_depth", None)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["max_depth", "random_state"]}
        return DecisionTreeRegressor(max_depth=max_depth, random_state=random_state, **clean_params)
        
    # 8. Random Forest Regression
    elif name_clean in ["randomforest", "randomforestregressor", "randomforestregression"]:
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", None)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_estimators", "max_depth", "random_state"]}
        return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **clean_params)
        
    # 9. Support Vector Regression (SVR)
    elif name_clean in ["svr", "supportvectorregression", "supportvectorregressor"]:
        kernel = hyperparameters.get("kernel", "rbf")
        C = hyperparameters.get("C", 1.0)
        epsilon = hyperparameters.get("epsilon", 0.1)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["kernel", "C", "epsilon"]}
        return SVR(kernel=kernel, C=C, epsilon=epsilon, **clean_params)
        
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Select one of the 9 standard regression algorithms.")
