import numpy as np
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    QuantileRegressor,
    Ridge,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from typing import Any, Dict, List, Optional

_POLYNOMIAL_NAMES = ("polynomialregression", "polynomial")


def _is_polynomial(model_name: str) -> bool:
    return (model_name or "").replace(" ", "").lower() in _POLYNOMIAL_NAMES


def get_regression_model(model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Returns an un-fitted scikit-learn (or XGBoost/LightGBM) regression model
    based on the model name.

    Supported model names:
    - "LinearRegression" or "Linear" or "MultipleLinear"
    - "PolynomialRegression" or "Polynomial" — returns bare LinearRegression;
      the polynomial feature expansion itself lives in build_pipeline()'s
      numeric branch, never applied to one-hot categorical columns.
    - "Ridge" or "RidgeRegression"
    - "Lasso" or "LassoRegression"
    - "ElasticNet" or "ElasticNetRegression"
    - "DecisionTree" or "DecisionTreeRegressor"
    - "RandomForest" or "RandomForestRegressor"
    - "GradientBoosting" or "GradientBoostingRegressor"
    - "XGBoost" or "XGBRegressor"
    - "LightGBM" or "LGBMRegressor"
    - "SVR" or "SupportVectorRegression"
    - "KNN" or "KNeighborsRegressor"
    - "Huber" or "HuberRegressor"
    - "BayesianRidge" or "BayesianRegression"
    - "QuantileRegression" or "QuantileRegressor"
    """
    if hyperparameters is None:
        hyperparameters = {}

    name_clean = model_name.replace(" ", "").lower()

    # 1 & 2. Linear and Multiple Linear Regression
    if name_clean in ["linearregression", "linear", "multiplelinear", "multiplelinearregression"]:
        return LinearRegression(**hyperparameters)

    # 3. Polynomial Regression — see _is_polynomial()/docstring above: the
    # PolynomialFeatures step is NOT here, it's in build_pipeline()'s numeric
    # sub-pipeline. This branch only validates degree and returns the final
    # LinearRegression estimator that sits after the expanded numeric +
    # one-hot categorical features are combined.
    elif _is_polynomial(model_name):
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
        return LinearRegression(**lr_params)

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

    # 10. Gradient Boosting Regression
    elif name_clean in ["gradientboosting", "gradientboostingregressor", "gradientboostingregression"]:
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", 3)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state"]
        }
        return GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, **clean_params,
        )

    # 11. XGBoost Regressor
    elif name_clean in ["xgboost", "xgbregressor", "xgboostregressor", "xgboostregression"]:
        from xgboost import XGBRegressor
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", 6)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state"]
        }
        return XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, **clean_params,
        )

    # 12. LightGBM Regressor
    elif name_clean in ["lightgbm", "lgbmregressor", "lightgbmregressor", "lightgbmregression"]:
        from lightgbm import LGBMRegressor
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", -1)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        # LightGBM logs verbose warnings (e.g. "No further splits") to
        # stdout/stderr by default, which is just noise for this app — muted
        # unless the caller explicitly overrides it via hyperparameters.
        verbose = hyperparameters.get("verbose", -1)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state", "verbose"]
        }
        return LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, verbose=verbose, **clean_params,
        )

    # 12.5 CatBoost Regressor
    elif name_clean in ["catboost", "catboostregressor", "catboostregression"]:
        from catboost import CatBoostRegressor
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", 6)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state"]
        }
        return CatBoostRegressor(
            iterations=n_estimators, depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, verbose=False, **clean_params,
        )

    # 13. K-Nearest Neighbors Regressor
    elif name_clean in ["knn", "kneighborsregressor", "knnregression", "kneighbors"]:
        n_neighbors = hyperparameters.get("n_neighbors", 5)
        clean_params = {k: v for k, v in hyperparameters.items() if k != "n_neighbors"}
        return KNeighborsRegressor(n_neighbors=n_neighbors, **clean_params)

    # 14. Huber Regressor (robust to outliers)
    elif name_clean in ["huber", "huberregressor", "huberregression"]:
        epsilon = hyperparameters.get("epsilon", 1.35)
        alpha = hyperparameters.get("alpha", 0.0001)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["epsilon", "alpha"]}
        return HuberRegressor(epsilon=epsilon, alpha=alpha, **clean_params)

    # 15. Bayesian Ridge Regression
    elif name_clean in ["bayesianridge", "bayesianregression", "bayesian"]:
        return BayesianRidge(**hyperparameters)

    # 16. Quantile Regression
    elif name_clean in ["quantileregression", "quantileregressor", "quantile"]:
        quantile = hyperparameters.get("quantile", 0.5)
        alpha = hyperparameters.get("alpha", 1.0)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["quantile", "alpha"]}
        return QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs", **clean_params)

    else:
        raise ValueError(f"Unsupported model type: {model_name}. Select one of the supported regression algorithms.")


def build_pipeline(
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]],
    numeric_features: List[str],
    categorical_features: List[str],
    scaling_method: str = "none",
) -> Pipeline:
    """
    Builds the full, leakage-safe preprocessing + model Pipeline:

        numeric features   -> SimpleImputer(median) -> [PolynomialFeatures if
                               model is Polynomial Regression] -> [scaler if
                               scaling_method is standard/minmax]
        categorical features -> SimpleImputer(most_frequent) ->
                               OneHotEncoder(handle_unknown="ignore", drop="first")
                               (drop="first" avoids the dummy-variable trap —
                               an undropped full one-hot block is exactly
                               collinear with the model's intercept, which
                               silently produces a singular/near-singular
                               design matrix for calculate_statistical_properties())
                                        |
                                  ColumnTransformer
                                        |
                                    regressor

    The caller (training_routes.train_model) is responsible for calling
    .fit() only on the training split and .transform()-only (never re-fit)
    on test/validation splits — this function only builds the unfitted
    Pipeline shape.
    """
    hyperparameters = hyperparameters or {}

    numeric_steps: List[Any] = [("imputer", SimpleImputer(strategy="median"))]
    if _is_polynomial(model_name):
        degree = hyperparameters.get("degree", 2)
        if not isinstance(degree, int) or degree < 1 or degree > 6:
            raise ValueError(
                f"Polynomial degree must be an integer between 1 and 6 (got {degree}). "
                "Higher degrees create an unreasonably large feature matrix."
            )
        numeric_steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    if scaling_method in ("standard", "minmax") and numeric_features:
        numeric_steps.append(("scaler", StandardScaler() if scaling_method == "standard" else MinMaxScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ]

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline(categorical_steps), categorical_features))

    # sparse_threshold=0 forces .transform() to always return a dense numpy
    # array. Left at its default, a high-cardinality categorical column
    # (e.g. a player-name column with hundreds of unique values) makes
    # OneHotEncoder's output sparse enough that ColumnTransformer returns a
    # scipy sparse matrix instead of an ndarray — which statsmodels' OLS
    # can't consume at all, and which crashes downstream
    # `pd.DataFrame(transformed, columns=..., index=...)` calls with a
    # "Shape of passed values is (n, 1), indices imply (n, p)" error (pandas
    # treats the whole sparse matrix as one opaque object per row). That
    # crash is an unhandled exception, so it never reaches an HTTPException
    # handler — FastAPI/Starlette's CORSMiddleware doesn't get a chance to
    # attach headers to the resulting 500, and the browser reports it as a
    # misleading "blocked by CORS policy" failure instead of the real cause.
    preprocessor = ColumnTransformer(transformers, sparse_threshold=0)
    estimator = get_regression_model(model_name, hyperparameters)

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
