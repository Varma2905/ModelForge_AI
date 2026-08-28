"""
Pytest coverage for the classification ML core (feature classification,
pipeline building, metrics, statistical properties, prediction) — verified
in isolation before being wrapped in HTTP routes, mirroring how the
regression pipeline's sparse-matrix bug was caught this session.

API-level (TestClient) coverage for /classify/select-features and
/classify/train-model lives in test_classification_api.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.ml.feature_types import classify_columns, map_expanded_coefficients
from app.ml.classification_models import get_classification_model, build_classification_pipeline
from app.ml.classification_evaluation import (
    calculate_classification_metrics,
    calculate_classification_statistical_properties,
)
from app.ml.prediction import save_model_package, predict_class_with_model


CLASSIFICATION_MODEL_NAMES = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "GradientBoosting",
    "SVM",
    "KNN",
    "NaiveBayes",
]


def _mixed_dataset(n=200, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    cities = [f"city_{i}" for i in range(6)]
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "salary": rng.integers(20000, 90000, n),
        "city": rng.choice(cities, n),
    })
    if n_classes == 2:
        y = ((df["age"] > 40) & (df["salary"] > 50000)).map({True: "approved", False: "denied"})
    else:
        bins = pd.qcut(df["age"], n_classes, labels=[f"class_{i}" for i in range(n_classes)])
        y = bins.astype(str)
    return df, y


class TestIsIntegerLike:
    def test_int_column_is_integer_like(self):
        df = pd.DataFrame({"label": [0, 1, 2, 1, 0]})
        result = classify_columns(df)
        assert result["label"]["kind"] == "numerical"
        assert result["label"]["is_integer_like"] is True

    def test_float_with_nan_whole_numbers_is_integer_like(self):
        df = pd.DataFrame({"label": [0.0, 1.0, np.nan, 2.0, 1.0]})
        result = classify_columns(df)
        assert result["label"]["kind"] == "numerical"
        assert result["label"]["is_integer_like"] is True

    def test_float_with_decimals_is_not_integer_like(self):
        df = pd.DataFrame({"price": [10.5, 20.25, 30.75, 40.1, 50.9]})
        result = classify_columns(df)
        assert result["price"]["kind"] == "numerical"
        assert result["price"]["is_integer_like"] is False

    def test_high_cardinality_int_id_is_integer_like_but_flagged(self):
        df = pd.DataFrame({"customer_id": list(range(100))})
        result = classify_columns(df)
        assert result["customer_id"]["kind"] == "numerical"
        assert result["customer_id"]["is_integer_like"] is True
        # is_integer_like alone doesn't make it eligible as a classification
        # target — the caller must also apply the unique_count<=20 rule.
        assert result["customer_id"]["unique_count"] > 20

    def test_categorical_column_has_no_meaningful_integer_like(self):
        df = pd.DataFrame({"city": ["Chennai", "Coimbatore", "Madurai"]})
        result = classify_columns(df)
        assert result["city"]["kind"] == "categorical"
        assert result["city"]["is_integer_like"] is False


class TestBuildClassificationPipeline:
    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODEL_NAMES)
    def test_fit_predict_roundtrip_binary(self, model_name):
        df, y = _mixed_dataset(n=150, n_classes=2)
        pipeline = build_classification_pipeline(
            model_name, {}, numeric_features=["age", "salary"], categorical_features=["city"],
        )
        pipeline.fit(df, y)
        preds = pipeline.predict(df)
        assert len(preds) == len(df)
        assert set(preds).issubset(set(y.unique()))

    @pytest.mark.parametrize("model_name", CLASSIFICATION_MODEL_NAMES)
    def test_fit_predict_roundtrip_multiclass(self, model_name):
        df, y = _mixed_dataset(n=180, n_classes=3)
        pipeline = build_classification_pipeline(
            model_name, {}, numeric_features=["age", "salary"], categorical_features=["city"],
        )
        pipeline.fit(df, y)
        preds = pipeline.predict(df)
        assert len(preds) == len(df)

    def test_high_cardinality_categorical_does_not_crash_dense_transform(self):
        # Regression test for the exact sparse-matrix bug fixed this session:
        # ColumnTransformer must return a dense ndarray even when a
        # high-cardinality categorical column would otherwise push it into
        # sparse output, since statsmodels Logit/MNLogit and
        # pd.DataFrame(transformed, columns=...) can't consume a scipy
        # sparse matrix.
        rng = np.random.default_rng(1)
        n = 500
        players = [f"player_{i}" for i in range(180)]
        df = pd.DataFrame({
            "runs": rng.integers(0, 100, n),
            "batter": rng.choice(players, n),
        })
        y = (df["runs"] > 50).map({True: "high", False: "low"})

        pipeline = build_classification_pipeline(
            "LogisticRegression", {}, numeric_features=["runs"], categorical_features=["batter"],
        )
        pipeline.fit(df, y)
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(df)
        assert isinstance(transformed, np.ndarray), "ColumnTransformer must return dense output"

        feature_names = list(preprocessor.get_feature_names_out())
        named = pd.DataFrame(transformed, columns=feature_names, index=df.index)
        assert named.shape[1] == len(feature_names)

    def test_svc_exposes_predict_proba(self):
        # probability=True is load-bearing (see build_classification_pipeline's
        # docstring) — without it, SVC.predict_proba() raises at call time.
        df, y = _mixed_dataset(n=100, n_classes=2)
        pipeline = build_classification_pipeline(
            "SVM", {}, numeric_features=["age", "salary"], categorical_features=["city"],
        )
        pipeline.fit(df, y)
        proba = pipeline.predict_proba(df)
        assert proba.shape == (len(df), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_unsupported_model_name_raises(self):
        with pytest.raises(ValueError):
            get_classification_model("NotARealClassifier")


class TestCalculateClassificationMetrics:
    def test_binary_metrics_shape_and_ranges(self):
        y_true = np.array(["yes", "no", "yes", "yes", "no", "no"])
        y_pred = np.array(["yes", "no", "no", "yes", "no", "yes"])
        classes = ["no", "yes"]
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba=None, classes=classes)

        assert 0.0 <= metrics["Accuracy"] <= 1.0
        assert 0.0 <= metrics["Precision"] <= 1.0
        assert 0.0 <= metrics["Recall"] <= 1.0
        assert 0.0 <= metrics["F1"] <= 1.0
        assert metrics["ROC_AUC"] is None  # no y_proba given -> must not crash, must be None
        assert len(metrics["ConfusionMatrix"]) == 2
        assert metrics["Classes"] == ["no", "yes"]

    def test_multiclass_uses_weighted_average_and_roc_auc_with_proba(self):
        classes = ["a", "b", "c"]
        y_true = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
        y_pred = np.array(["a", "b", "c", "a", "c", "c", "a", "b", "b"])
        # A plausible-looking probability matrix aligned to `classes` order.
        rng = np.random.default_rng(2)
        y_proba = rng.dirichlet(alpha=[2, 2, 2], size=len(y_true))

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba=y_proba, classes=classes)
        assert metrics["ROC_AUC"] is not None
        assert 0.0 <= metrics["ROC_AUC"] <= 1.0
        assert len(metrics["ConfusionMatrix"]) == 3

    def test_roc_auc_failure_is_caught_not_raised(self):
        # A y_proba shape mismatched against `classes` is a hard
        # roc_auc_score ValueError — must degrade to None, not crash the
        # whole training request.
        classes = ["a", "b", "c"]
        y_true = np.array(["a", "a", "b", "b"])
        y_pred = np.array(["a", "b", "a", "b"])
        y_proba = np.tile([0.5, 0.5], (4, 1))  # only 2 columns for 3 classes
        metrics = calculate_classification_metrics(y_true, y_pred, y_proba=y_proba, classes=classes)
        assert metrics["ROC_AUC"] is None


class TestCalculateClassificationStatisticalProperties:
    def _transformed_frame(self, df, y, categorical_features):
        pipeline = build_classification_pipeline(
            "LogisticRegression", {}, numeric_features=["age", "salary"], categorical_features=categorical_features,
        )
        pipeline.fit(df, y)
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(df)
        feature_names = list(preprocessor.get_feature_names_out())
        return pd.DataFrame(transformed, columns=feature_names, index=df.index), pipeline.named_steps["model"]

    def test_binary_logit_produces_real_stats(self):
        df, y = _mixed_dataset(n=200, n_classes=2)
        X_named, model = self._transformed_frame(df, y, ["city"])
        classes = list(model.classes_)
        stats = calculate_classification_statistical_properties(X_named, y, classes)

        assert "const" in stats["coefficients"]
        assert stats["summary_text"]
        # A real (non-fallback) fit should not produce the fallback's
        # telltale all-0.5 p-values across every feature.
        non_const_p = [v for k, v in stats["p_values"].items() if k != "const"]
        assert not all(abs(p - 0.5) < 1e-9 for p in non_const_p)

    def test_multiclass_mnlogit_flattens_keys_and_maps_correctly(self):
        df, y = _mixed_dataset(n=240, n_classes=3)
        X_named, model = self._transformed_frame(df, y, ["city"])
        classes = list(model.classes_)
        stats = calculate_classification_statistical_properties(X_named, y, classes)

        # MNLogit keys are flattened as "{feature}__class_{k}" (including
        # "const__class_{k}" intercept terms) — confirm the shared
        # map_expanded_coefficients() helper still attributes real features
        # back to the correct source column via its longest-prefix match,
        # and internally excludes the const__class_* intercept terms the
        # same way it excludes a plain "const" for binary/regression fits.
        # Deliberately NOT pre-filtered here — real callers (classify_routes)
        # pass the raw coefficients dict straight through, same as regression.
        coefs = stats["coefficients"]
        assert any("__class_" in k for k in coefs if not k.startswith("const"))

        encoded_categorical_names = [k[len("cat__"):] for k in coefs if k.startswith("cat__")]
        mapped = map_expanded_coefficients(coefs, ["age", "salary"], ["city"], encoded_categorical_names)
        # Every mapped entry must resolve to a real source feature (age,
        # salary, or city) — not fall through to the "unrecognized key"
        # passthrough branch, which would indicate the __class_k suffix
        # broke the prefix matcher.
        for entry in mapped:
            assert entry["source_feature"] in ("age", "salary", "city")

    def test_perfect_separation_falls_back_gracefully(self):
        # A feature that perfectly predicts the class is a classic
        # non-convergence trigger for logistic MLE.
        df = pd.DataFrame({
            "age": [20, 21, 22, 60, 61, 62],
            "salary": [30000, 31000, 32000, 80000, 81000, 82000],
            "city": ["a", "a", "a", "b", "b", "b"],
        })
        y = pd.Series(["low", "low", "low", "high", "high", "high"])
        X_named, model = self._transformed_frame(df, y, ["city"])
        classes = list(model.classes_)
        stats = calculate_classification_statistical_properties(X_named, y, classes)
        # Must always return the full expected key set, fallback or not.
        for key in ("coefficients", "standard_errors", "t_statistics", "p_values", "f_statistic", "f_pvalue", "summary_text"):
            assert key in stats


class TestPredictClassWithModel:
    def test_string_label_roundtrip_does_not_crash(self, tmp_path, monkeypatch):
        # The exact case predict_with_model() (float-only) cannot handle.
        import app.ml.prediction as prediction_module
        monkeypatch.setattr(prediction_module, "MODELS_DIR", str(tmp_path))

        df, y = _mixed_dataset(n=120, n_classes=2)
        pipeline = build_classification_pipeline(
            "RandomForest", {}, numeric_features=["age", "salary"], categorical_features=["city"],
        )
        pipeline.fit(df, y)

        prediction_module.save_model_package(
            model_id="test_clf_model",
            model=pipeline.named_steps["model"],
            features=["age", "salary", "city"],
            target="outcome",
            preprocessor=pipeline.named_steps["preprocessor"],
            numeric_features=["age", "salary"],
        )

        result = prediction_module.predict_class_with_model(
            "test_clf_model", {"age": 45, "salary": 60000, "city": "city_0"},
        )
        assert result["prediction"] in ("approved", "denied")
        assert result["probabilities"] is not None
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6
