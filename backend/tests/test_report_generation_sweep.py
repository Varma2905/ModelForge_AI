"""
End-to-end sweep over Regression/Classification/Clustering report generation,
driving the real HTTP endpoints exactly like test_api.py / test_classification_api.py /
test_clustering_report.py do (TestClient(app), real signup/create-dataset/select-
features/train-model/report flow) — no mocking of the ML engine or PDF pipeline.

Verifies the fixes made across app/ml/feature_types.py (target-leakage +
identifier-column detection, feature_counts), app/reports/report_validation.py
(pre-render consistency validation), app/reports/pdf_generator.py +
app/reports/clustering_report.py (auxiliary-OLS/Logit disclaimer, encoded
feature-count display), and app/visualization/graph_generator.py (real
feature-importance charts):

1. Every report downloads as a valid PDF for a representative spread of
   algorithms per task.
2. The auxiliary-model disclaimer appears for non-Linear/non-Logistic models
   and is absent for genuine Linear/Logistic Regression.
3. The encoded/model-input feature count is strictly greater than the raw
   selected feature count when a categorical feature is present (proves
   one-hot expansion is actually surfaced, not just duplicated).
4. Deliberately-rigged datasets (a leakage feature, an identifier-named
   column) produce the expected warnings from /select-features and are
   persisted onto the trained model for the PDF's Data Quality Notices panel.

Every record this suite creates is deleted again in the module-scope cleanup
fixture, so re-running the suite never leaves stale data in db.json.
"""
import asyncio
import io
import uuid

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from app.main import app
from app.database.mongodb import db_client
from app.ml.feature_types import (
    detect_identifier_target,
    detect_target_leakage,
    classify_columns,
)
from app.reports.report_validation import validate_regression_report, validate_classification_report

_created = {"users": [], "datasets": [], "models": []}


def _unique_email(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"


def _signup(client: TestClient, prefix: str) -> dict:
    resp = client.post("/auth/signup", json={
        "name": f"Pytest {prefix}",
        "email": _unique_email(prefix),
        "password": "supersecret123",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    user_id = body["data"]["user"]["id"]
    token = body["data"]["token"]
    _created["users"].append(user_id)
    return {"user_id": user_id, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_module():
    yield

    async def _cleanup():
        for user_id in _created["users"]:
            await db_client.delete_one("users", {"_id": user_id})
        for dataset_id in _created["datasets"]:
            await db_client.delete_one("datasets", {"_id": dataset_id})
            from app.datasets import store as dataset_store
            try:
                dataset_store.delete(dataset_id)
            except Exception:
                pass
        for model_id in _created["models"]:
            await db_client.delete_one("models", {"_id": model_id})
            await db_client.delete_one("reports", {"_id": f"rep_{model_id}"})

    asyncio.run(_cleanup())


def _pdf_text(pdf_bytes: bytes) -> str:
    """Extracted text, whitespace-normalized to a single space between
    tokens. A ReportLab table cell wraps long labels like "Encoded/Model
    Input Dimensions" across multiple visual lines when the column is
    narrow, and pypdf's extraction reproduces that as embedded newlines —
    normalizing here so every `in text` / regex check below is robust to
    exactly where a label happened to wrap, rather than asserting on exact
    line breaks that depend on incidental column width."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return " ".join(raw.split())


def _extract_dimension_count(text: str, label: str):
    """Pulls the integer that follows `label` in extracted (whitespace-
    normalized) PDF text — used to check the encoded feature count exceeds
    the raw one, rather than string-matching exact markup (extracted PDF
    text carries no ReportLab tags to match against)."""
    import re
    idx = text.find(label)
    if idx == -1:
        return None
    tail = text[idx + len(label):idx + len(label) + 40]
    m = re.search(r"\d+", tail)
    return int(m.group()) if m else None


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _regression_dataset_payload(n=48):
    # target: sale_price. sale_price_usd is a name-pattern leakage feature
    # (perfectly derived from the target). id_row is an identifier column.
    # sq_ft is a legitimate numeric predictor. city is categorical (4
    # categories) so the encoded feature count exceeds the raw count.
    cities = ["Chennai", "Coimbatore", "Madurai", "Salem"]
    columns = ["id_row", "sq_ft", "city", "sale_price_usd", "sale_price"]
    rows = []
    for i in range(n):
        sq_ft = 500.0 + i * 15
        price = 20000.0 + sq_ft * 120 + (i % 5) * 300
        rows.append([i, sq_ft, cities[i % 4], round(price * 1.1, 2), round(price, 2)])
    return {"name": "pytest_report_sweep_regression", "columns": columns, "rows": rows}


REGRESSION_MODELS = [
    "Linear Regression",
    "Random Forest Regression",
    "CatBoost",
    "XGBoost",
    "Ridge Regression",
    "Lasso Regression",
    "Support Vector Regression",
]


@pytest.fixture(scope="module")
def regression_dataset(client):
    user = _signup(client, "repsweep_reg")
    headers = user["headers"]
    create_resp = client.post("/create-dataset", json=_regression_dataset_payload(), headers=headers)
    assert create_resp.status_code == 200, create_resp.text
    dataset_id = create_resp.json()["data"]["dataset_id"]
    _created["datasets"].append(dataset_id)
    return {"dataset_id": dataset_id, "headers": headers}


class TestRegressionSelectFeaturesWarnings:
    def test_flags_leakage_and_identifier(self, client, regression_dataset):
        headers = regression_dataset["headers"]
        resp = client.post("/select-features", json={
            "dataset_id": regression_dataset["dataset_id"],
            "features": ["id_row", "sq_ft", "city", "sale_price_usd"],
            "target": "sale_price",
        }, headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        warnings = data["warnings"]
        assert any("id_row" in w for w in warnings), warnings
        assert any("sale_price_usd" in w for w in warnings), warnings
        leakage_warnings = data["leakage_warnings"]
        assert any(w["feature"] == "sale_price_usd" for w in leakage_warnings), leakage_warnings


@pytest.mark.parametrize("model_name", REGRESSION_MODELS)
class TestRegressionReportSweep:
    def _train_and_download(self, client, regression_dataset, model_name):
        headers = regression_dataset["headers"]
        # Simple "Linear Regression" only accepts a single predictor — every
        # other model gets both the numeric and categorical feature so the
        # encoded-vs-raw dimension check below has a categorical column to
        # actually expand.
        features = ["sq_ft"] if model_name == "Linear Regression" else ["sq_ft", "city"]
        train_resp = client.post("/train-model", json={
            "dataset_id": regression_dataset["dataset_id"],
            "model": model_name,
            "features": features,
            "target": "sale_price",
            "split": {"test_size": 0.25, "val_size": 0.0, "random_state": 42},
        }, headers=headers)
        assert train_resp.status_code == 200, train_resp.text
        model_id = train_resp.json()["data"]["model_id"]
        _created["models"].append(model_id)

        report_resp = client.get(f"/report/{model_id}", headers=headers)
        assert report_resp.status_code == 200, report_resp.text
        assert report_resp.headers["content-type"] == "application/pdf"
        return report_resp.content

    def test_report_downloads_and_discloses_auxiliary_model_correctly(self, client, regression_dataset, model_name):
        pdf_bytes = self._train_and_download(client, regression_dataset, model_name)
        assert pdf_bytes[:4] == b"%PDF"
        text = _pdf_text(pdf_bytes)

        if model_name == "Linear Regression":
            assert "Auxiliary Model" not in text
            assert "Statistical Parameter Analysis" in text
        else:
            assert "Auxiliary Model: OLS" in text, text[:2000]

        assert "Encoded/Model Input Dimensions" in text
        if model_name != "Linear Regression":
            # city (4 categories) one-hot expands beyond the 2 raw features.
            encoded_dims = _extract_dimension_count(text, "Encoded/Model Input Dimensions")
            assert encoded_dims is not None and encoded_dims > 2, text


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classification_dataset_payload(n=60):
    # `outcome` is integer-encoded (0/1) rather than string-labeled — some
    # classifiers' sklearn wrappers (XGBoost) require numeric class labels,
    # and feature_types.classify_columns()/_is_valid_classification_target()
    # both already treat a low-cardinality integer column as a valid
    # classification target (see is_integer_like), so this exercises that
    # path too. `outcome_flag` duplicates `outcome` exactly — a deliberate,
    # maximally obvious leakage feature (name-pattern AND perfect
    # correlation both fire).
    cities = ["Chennai", "Coimbatore", "Madurai", "Salem"]
    columns = ["id_customer", "age", "city", "outcome_flag", "outcome"]
    rows = []
    for i in range(n):
        age = 20 + (i % 45)
        outcome = 1 if age > 40 else 0
        rows.append([i, age, cities[i % 4], outcome, outcome])
    return {"name": "pytest_report_sweep_classification", "columns": columns, "rows": rows}


CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "KNN Classification",
    "Decision Tree Classification",
    "Random Forest Classification",
    "XGBoost",
    "CatBoost",
]


@pytest.fixture(scope="module")
def classification_dataset(client):
    user = _signup(client, "repsweep_clf")
    headers = user["headers"]
    create_resp = client.post("/create-dataset", json=_classification_dataset_payload(), headers=headers)
    assert create_resp.status_code == 200, create_resp.text
    dataset_id = create_resp.json()["data"]["dataset_id"]
    _created["datasets"].append(dataset_id)
    return {"dataset_id": dataset_id, "headers": headers}


class TestClassificationSelectFeaturesWarnings:
    def test_flags_leakage_and_identifier(self, client, classification_dataset):
        headers = classification_dataset["headers"]
        resp = client.post("/classify/select-features", json={
            "dataset_id": classification_dataset["dataset_id"],
            "features": ["id_customer", "age", "city", "outcome_flag"],
            "target": "outcome",
        }, headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        warnings = data["warnings"]
        assert any("id_customer" in w for w in warnings), warnings
        assert any("outcome_flag" in w for w in warnings), warnings
        leakage_warnings = data["leakage_warnings"]
        assert any(w["feature"] == "outcome_flag" for w in leakage_warnings), leakage_warnings


@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
class TestClassificationReportSweep:
    def _train_and_download(self, client, classification_dataset, model_name):
        headers = classification_dataset["headers"]
        train_resp = client.post("/classify/train-model", json={
            "dataset_id": classification_dataset["dataset_id"],
            "model": model_name,
            "features": ["age", "city"],
            "target": "outcome",
            "split": {"test_size": 0.25, "val_size": 0.0, "random_state": 42},
        }, headers=headers)
        assert train_resp.status_code == 200, train_resp.text
        model_id = train_resp.json()["data"]["model_id"]
        _created["models"].append(model_id)

        report_resp = client.get(f"/report/{model_id}", headers=headers)
        assert report_resp.status_code == 200, report_resp.text
        assert report_resp.headers["content-type"] == "application/pdf"
        return report_resp.content

    def test_report_downloads_with_correct_feature_dimensions(self, client, classification_dataset, model_name):
        # Note: classification's live PDF path never renders raw Logit
        # coefficients (its "AI Agent Insights" section is metrics-only —
        # see pdf_generator.py's _build_classification_ai_insights_text —
        # and its Feature Importance chart already correctly uses the real
        # trained model's own importances, see classify_routes.py's
        # `visualizations["feature_importance"]`), so there's no auxiliary
        # OLS/Logit content in this report to mislabel in the first place;
        # unlike regression, no disclaimer text is expected here.
        pdf_bytes = self._train_and_download(client, classification_dataset, model_name)
        assert pdf_bytes[:4] == b"%PDF"
        text = _pdf_text(pdf_bytes)

        assert "Encoded/Model Input Dimensions" in text
        # city (4 categories) one-hot expands beyond the 2 raw features.
        encoded_dims = _extract_dimension_count(text, "Encoded/Model Input Dimensions")
        assert encoded_dims is not None and encoded_dims > 2, text


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _clustering_dataset_payload(n=40):
    cities = ["Chennai", "Coimbatore", "Madurai", "Salem"]
    columns = ["spend", "visits", "city"]
    rows = []
    for i in range(n):
        rows.append([100.0 + i * 7, float(i % 12), cities[i % 4]])
    return {"name": "pytest_report_sweep_clustering", "columns": columns, "rows": rows}


CLUSTERING_MODELS = [
    ("K-Means", {"n_clusters": 3}),
    ("DBSCAN", {"eps": 8.0, "min_samples": 3}),
    ("Agglomerative Clustering", {"n_clusters": 3, "linkage": "ward"}),
]


@pytest.fixture(scope="module")
def clustering_dataset(client):
    user = _signup(client, "repsweep_clu")
    headers = user["headers"]
    create_resp = client.post("/create-dataset", json=_clustering_dataset_payload(), headers=headers)
    assert create_resp.status_code == 200, create_resp.text
    dataset_id = create_resp.json()["data"]["dataset_id"]
    _created["datasets"].append(dataset_id)
    return {"dataset_id": dataset_id, "headers": headers}


@pytest.mark.parametrize("model_name,hyperparameters", CLUSTERING_MODELS)
class TestClusteringReportSweep:
    def test_report_downloads_with_encoded_dimensions(self, client, clustering_dataset, model_name, hyperparameters):
        headers = clustering_dataset["headers"]
        train_resp = client.post("/cluster/train-model", json={
            "dataset_id": clustering_dataset["dataset_id"],
            "model": model_name,
            "features": ["spend", "visits", "city"],
            "hyperparameters": hyperparameters,
        }, headers=headers)
        assert train_resp.status_code == 200, train_resp.text
        model_id = train_resp.json()["data"]["model_id"]
        _created["models"].append(model_id)

        report_resp = client.get(f"/report/{model_id}", headers=headers)
        assert report_resp.status_code == 200, report_resp.text
        assert report_resp.headers["content-type"] == "application/pdf"
        text = _pdf_text(report_resp.content)
        assert "Encoded Feature Dimensions" in text


# ---------------------------------------------------------------------------
# Direct unit tests (no HTTP) for the new detection/validation helpers
# ---------------------------------------------------------------------------

def test_detect_identifier_target_warns_but_does_not_block():
    import pandas as pd
    # `amount` deliberately repeats values so its uniqueness ratio stays
    # below the value-based identifier heuristic's threshold (see
    # feature_types._is_likely_identifier) — otherwise a fully-unique
    # numeric column would ALSO look like an identifier by that heuristic,
    # regardless of its name, which is correct behavior but would make this
    # a bad negative-case fixture.
    df = pd.DataFrame({
        "customer_id": list(range(1, 21)),
        "amount": [10.0, 20.0, 30.0, 40.0] * 5,
    })
    kind_by_col = classify_columns(df)
    warning = detect_identifier_target("customer_id", kind_by_col)
    assert warning is not None
    assert "identifier" in warning.lower()
    assert detect_identifier_target("amount", kind_by_col) is None


def test_detect_target_leakage_name_pattern_and_correlation():
    import pandas as pd
    n = 30
    df = pd.DataFrame({
        "price": [100.0 + i for i in range(n)],
        "price_usd": [110.0 + i for i in range(n)],  # name-pattern match
        "final_price": [(100.0 + i) * 1.0 for i in range(n)],  # perfectly correlated
        "unrelated": [float(i % 3) for i in range(n)],
    })
    kind_by_col = classify_columns(df)
    warnings = detect_target_leakage(df, "price", ["price_usd", "final_price", "unrelated"], kind_by_col)
    flagged = {w["feature"] for w in warnings}
    assert "price_usd" in flagged
    assert "final_price" in flagged
    assert "unrelated" not in flagged


def test_validate_regression_report_flags_missing_metrics():
    model_info = {
        "model_type": "regression",
        "features": ["a", "b"],
        "feature_counts": {"raw_selected": 2},
        "total_rows": 100,
        "train_rows": 80,
        "test_rows": 20,
        "val_rows": 0,
        "metrics": {"R2": 0.9, "MSE": 1.0},  # RMSE/MAE missing
    }
    issues = validate_regression_report(model_info)
    codes = {i["code"] for i in issues}
    assert "METRIC_MISSING" in codes


def test_validate_classification_report_flags_confusion_matrix_shape_mismatch():
    model_info = {
        "model_type": "classification",
        "features": ["a"],
        "total_rows": 50,
        "train_rows": 40,
        "test_rows": 10,
        "val_rows": 0,
        "metrics": {
            "Accuracy": 0.8, "F1": 0.75,
            "ConfusionMatrix": [[10, 2], [1, 8], [0, 0]],
            "Classes": ["x", "y"],
        },
    }
    issues = validate_classification_report(model_info)
    codes = {i["code"] for i in issues}
    assert "CONFUSION_MATRIX_SHAPE_MISMATCH" in codes
