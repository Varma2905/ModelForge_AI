from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from typing import Any, Dict, List, Optional


def get_classification_model(model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Returns an un-fitted scikit-learn (or XGBoost/LightGBM/CatBoost)
    classification model based on the model name. Mirrors
    get_regression_model()'s alias-branch pattern exactly (see
    app/ml/regression_models.py).

    Supported model names:
    - "LogisticRegression" or "Logistic"
    - "DecisionTree" or "DecisionTreeClassifier"
    - "RandomForest" or "RandomForestClassifier"
    - "GradientBoosting" or "GradientBoostingClassifier"
    - "XGBoost" or "XGBClassifier"
    - "LightGBM" or "LGBMClassifier"
    - "CatBoost" or "CatBoostClassifier"
    - "SVM" or "SupportVectorMachine" or "SVC"
    - "KNN" or "KNeighborsClassifier"
    - "NaiveBayes" or "GaussianNB"
    - "ExtraTrees" or "ExtraTreesClassifier"
    - "AdaBoost" or "AdaBoostClassifier"
    """
    if hyperparameters is None:
        hyperparameters = {}

    name_clean = model_name.replace(" ", "").lower()

    # 1. Logistic Regression
    if name_clean in ["logisticregression", "logistic"]:
        # max_iter bumped from sklearn's default of 100 — logistic
        # regression frequently fails to converge on one-hot-expanded
        # features within the default iteration budget.
        max_iter = hyperparameters.get("max_iter", 1000)
        clean_params = {k: v for k, v in hyperparameters.items() if k != "max_iter"}
        return LogisticRegression(max_iter=max_iter, **clean_params)

    # 2. Decision Tree Classifier
    elif name_clean in ["decisiontree", "decisiontreeclassifier", "decisiontreeclassification"]:
        max_depth = hyperparameters.get("max_depth", None)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["max_depth", "random_state"]}
        return DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, **clean_params)

    # 3. Random Forest Classifier
    elif name_clean in ["randomforest", "randomforestclassifier", "randomforestclassification"]:
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", None)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_estimators", "max_depth", "random_state"]}
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **clean_params)

    # 4. Gradient Boosting Classifier
    elif name_clean in ["gradientboosting", "gradientboostingclassifier", "gradientboostingclassification"]:
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", 3)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state"]
        }
        return GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, **clean_params,
        )

    # 5. XGBoost Classifier
    elif name_clean in ["xgboost", "xgbclassifier", "xgboostclassifier", "xgboostclassification"]:
        from xgboost import XGBClassifier
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", 6)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state"]
        }
        return XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, **clean_params,
        )

    # 6. LightGBM Classifier
    elif name_clean in ["lightgbm", "lgbmclassifier", "lightgbmclassifier", "lightgbmclassification"]:
        from lightgbm import LGBMClassifier
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", -1)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        verbose = hyperparameters.get("verbose", -1)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state", "verbose"]
        }
        return LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, verbose=verbose, **clean_params,
        )

    # 6.5 CatBoost Classifier
    elif name_clean in ["catboost", "catboostclassifier", "catboostclassification"]:
        from catboost import CatBoostClassifier
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", 6)
        learning_rate = hyperparameters.get("learning_rate", 0.1)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {
            k: v for k, v in hyperparameters.items()
            if k not in ["n_estimators", "max_depth", "learning_rate", "random_state"]
        }
        return CatBoostClassifier(
            iterations=n_estimators, depth=max_depth, learning_rate=learning_rate,
            random_state=random_state, verbose=False, **clean_params,
        )

    # 7. Support Vector Machine
    elif name_clean in ["svm", "supportvectormachine", "svc"]:
        kernel = hyperparameters.get("kernel", "rbf")
        C = hyperparameters.get("C", 1.0)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["kernel", "C", "probability"]}
        # probability=True is load-bearing, not user-overridable — ROC/PR
        # curves and the Predict page's per-class probability display both
        # require predict_proba(), which SVC only exposes when this is set.
        return SVC(kernel=kernel, C=C, probability=True, **clean_params)

    # 8. K-Nearest Neighbors Classifier
    elif name_clean in ["knn", "kneighborsclassifier", "knnclassification", "kneighbors"]:
        n_neighbors = hyperparameters.get("n_neighbors", 5)
        clean_params = {k: v for k, v in hyperparameters.items() if k != "n_neighbors"}
        return KNeighborsClassifier(n_neighbors=n_neighbors, **clean_params)

    # 9. Naive Bayes
    elif name_clean in ["naivebayes", "gaussiannb", "nb"]:
        return GaussianNB(**hyperparameters)

    # 10. Extra Trees Classifier
    elif name_clean in ["extratrees", "extratreesclassifier", "extratreesclassification"]:
        n_estimators = hyperparameters.get("n_estimators", 100)
        max_depth = hyperparameters.get("max_depth", None)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_estimators", "max_depth", "random_state"]}
        return ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **clean_params)

    # 11. AdaBoost Classifier
    elif name_clean in ["adaboost", "adaboostclassifier", "adaboostclassification"]:
        n_estimators = hyperparameters.get("n_estimators", 50)
        learning_rate = hyperparameters.get("learning_rate", 1.0)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_estimators", "learning_rate", "random_state"]}
        return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state, **clean_params)

    else:
        raise ValueError(f"Unsupported model type: {model_name}. Select one of the supported classification algorithms.")


def build_classification_pipeline(
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]],
    numeric_features: List[str],
    categorical_features: List[str],
    scaling_method: str = "none",
) -> Pipeline:
    """
    Builds the full, leakage-safe preprocessing + classifier Pipeline. Mirrors
    app/ml/regression_models.py::build_pipeline() exactly — same numeric/
    categorical sub-pipeline construction, since that part is entirely
    feature-side and target-agnostic. No PolynomialFeatures branch: no
    classification model here needs polynomial feature expansion.

        numeric features   -> SimpleImputer(median) -> [scaler if
                               scaling_method is standard/minmax]
        categorical features -> SimpleImputer(most_frequent) ->
                               OneHotEncoder(handle_unknown="ignore", drop="first")
                                        |
                                  ColumnTransformer(sparse_threshold=0)
                                        |
                                    classifier

    sparse_threshold=0 forces dense ndarray output — without it, a
    high-cardinality categorical column makes ColumnTransformer return a
    scipy sparse matrix, which crashes statsmodels' Logit/MNLogit fit and any
    `pd.DataFrame(transformed, columns=...)` call with a shape error. This is
    the exact bug class fixed for the regression pipeline this session; it
    applies identically here since it's about ColumnTransformer's general
    behavior with high-cardinality categoricals, not about the target type.

    The caller (classify_routes.train_model) is responsible for calling
    .fit() only on the training split and .transform()-only (never re-fit)
    on test/validation splits.
    """
    hyperparameters = hyperparameters or {}

    numeric_steps: List[Any] = [("imputer", SimpleImputer(strategy="median"))]
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

    preprocessor = ColumnTransformer(transformers, sparse_threshold=0)
    estimator = get_classification_model(model_name, hyperparameters)

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
