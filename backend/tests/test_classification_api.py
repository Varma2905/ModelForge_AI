"""
Pytest + FastAPI TestClient coverage for the /classify/* API layer, mirroring
test_api.py's conventions (happy path, validation errors, predict, and
multi-tenant isolation). Also covers the shared endpoints (/model-metrics,
/models, /datasets/{id}/models, /predict) with a classification model doc,
since those are reused unmodified from the regression pipeline and must
correctly surface model_type + classification-shaped metrics.

Every record this suite creates is deleted again in the module-scope cleanup
fixture below, so re-running the suite never leaves stale data in db.json.
"""
import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database.database import db_client
from app.datasets import store as dataset_store

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_created = {"users": [], "datasets": [], "models": []}


def _classification_dataset_payload(name="pytest_classification_dataset", n=60):
    columns = ["age", "salary", "city", "outcome"]
    rows = []
    cities = ["Chennai", "Coimbatore", "Madurai"]
    for i in range(n):
        age = 20 + (i % 45)
        salary = 20000 + (i * 733) % 70000
        city = cities[i % 3]
        outcome = "approved" if (age > 40 and salary > 50000) else "denied"
        rows.append([age, salary, city, outcome])
    return {"name": name, "columns": columns, "rows": rows}


def _unique_email(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"


def _signup(client: TestClient, prefix: str) -> dict:
    email = _unique_email(prefix)
    resp = client.post("/auth/signup", json={
        "name": f"Pytest {prefix}",
        "email": email,
        "password": "supersecret123",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    user_id = body["data"]["user"]["id"]
    token = body["data"]["token"]
    _created["users"].append(user_id)
    return {"user_id": user_id, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def user_a(client):
    return _signup(client, "clf_usera")


@pytest.fixture(scope="module")
def user_b(client):
    return _signup(client, "clf_userb")


@pytest.fixture(scope="module")
def trained_classifier(client, user_a):
    """Runs the real upload -> classify/select-features -> classify/train-model
    pipeline once and shares the resulting model_id across the tests below."""
    headers = user_a["headers"]

    create_resp = client.post("/create-dataset", json=_classification_dataset_payload(), headers=headers)
    assert create_resp.status_code == 200, create_resp.text
    dataset_id = create_resp.json()["data"]["dataset_id"]
    _created["datasets"].append(dataset_id)

    select_resp = client.post("/classify/select-features", json={
        "dataset_id": dataset_id,
        "features": ["age", "salary", "city"],
        "target": "outcome",
    }, headers=headers)
    assert select_resp.status_code == 200, select_resp.text
    select_data = select_resp.json()["data"]
    assert select_data["numerical_features"] == ["age", "salary"]
    assert select_data["categorical_features"] == ["city"]

    train_resp = client.post("/classify/train-model", json={
        "dataset_id": dataset_id,
        "model": "LogisticRegression",
        "features": ["age", "salary", "city"],
        "target": "outcome",
        "split": {"test_size": 0.25, "val_size": 0.0, "random_state": 42},
    }, headers=headers)
    assert train_resp.status_code == 200, train_resp.text
    train_data = train_resp.json()["data"]
    model_id = train_data["model_id"]
    _created["models"].append(model_id)

    return {"dataset_id": dataset_id, "model_id": model_id, "metrics": train_data["metrics"]}


class TestClassificationHappyPath:
    def test_select_features_and_train_model(self, trained_classifier):
        metrics = trained_classifier["metrics"]
        assert 0.0 <= metrics["Accuracy"] <= 1.0
        assert 0.0 <= metrics["F1"] <= 1.0
        assert len(metrics["ConfusionMatrix"]) == 2
        assert metrics["Classes"] == ["approved", "denied"]

    def test_model_metrics_endpoint_reports_model_type(self, client, user_a, trained_classifier):
        resp = client.get(f"/model-metrics/{trained_classifier['model_id']}", headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["model_type"] == "classification"
        assert set(data["classes"]) == {"approved", "denied"}
        assert "confusion_matrix" in data["chart_data"]

    def test_models_list_reports_model_type(self, client, user_a, trained_classifier):
        resp = client.get("/models", headers=user_a["headers"])
        assert resp.status_code == 200
        model = next(m for m in resp.json()["data"] if m["model_id"] == trained_classifier["model_id"])
        assert model["model_type"] == "classification"

    def test_dataset_models_endpoint_reports_model_type(self, client, user_a, trained_classifier):
        resp = client.get(f"/datasets/{trained_classifier['dataset_id']}/models", headers=user_a["headers"])
        assert resp.status_code == 200
        model = next(m for m in resp.json()["data"] if m["model_id"] == trained_classifier["model_id"])
        assert model["model_type"] == "classification"

    def test_predict_returns_class_label_and_probabilities(self, client, user_a, trained_classifier):
        resp = client.post("/predict", json={
            "model_id": trained_classifier["model_id"],
            "values": {"age": 45, "salary": 60000, "city": "Chennai"},
        }, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["prediction"] in ("approved", "denied")
        assert data["probabilities"] is not None
        assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-6

    def test_ai_explain_uses_classification_agents(self, client, user_a, trained_classifier):
        resp = client.post("/ai/explain", json={"model_id": trained_classifier["model_id"]}, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        # The regression fallback text always mentions "R²" / "regression
        # line" — a classification explanation must not fall back to it.
        assert "R²" not in data["full_report"]

    def test_report_download_succeeds(self, client, user_a, trained_classifier):
        resp = client.get(f"/report/{trained_classifier['model_id']}", headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "application/pdf"

    def test_dashboard_reports_both_model_types_correctly(self, client, user_a, trained_classifier):
        # A regression model trained alongside the classification fixture is
        # the scenario most likely to surface a missed
        # .get("model_type", "regression") default somewhere in the
        # dashboard aggregation — verify both sides are counted correctly
        # and neither pollutes the other's average/best-model computation.
        headers = user_a["headers"]
        create_resp = client.post("/create-dataset", json={
            "name": "pytest_regression_for_dashboard",
            "columns": ["x1", "x2", "y"],
            "rows": [[float(i), float(i * 2), 3 * i + 5.0] for i in range(25)],
        }, headers=headers)
        dataset_id = create_resp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        train_resp = client.post("/train-model", json={
            "dataset_id": dataset_id,
            "model": "LinearRegression",
            "features": ["x1", "x2"],
            "target": "y",
            "split": {"test_size": 0.2, "val_size": 0.0, "random_state": 42},
        }, headers=headers)
        assert train_resp.status_code == 200, train_resp.text
        regression_model_id = train_resp.json()["data"]["model_id"]
        _created["models"].append(regression_model_id)

        resp = client.get("/dashboard/summary", headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]

        assert data["total_classification_analyses"] >= 1
        assert data["best_accuracy_model"] is not None
        assert 0.0 <= data["best_accuracy_model"]["accuracy"] <= 1.0
        assert data["avg_accuracy"] is not None
        # The regression model must still be counted/averaged on the R2 side,
        # not silently absorbed into the classification bucket.
        assert data["best_model"] is not None
        assert data["avg_r2"] is not None

        recent_ids = {a["model_id"] for a in data["recent_analyses"]}
        if trained_classifier["model_id"] in recent_ids:
            clf_entry = next(a for a in data["recent_analyses"] if a["model_id"] == trained_classifier["model_id"])
            assert clf_entry["model_type"] == "classification"
        if regression_model_id in recent_ids:
            reg_entry = next(a for a in data["recent_analyses"] if a["model_id"] == regression_model_id)
            assert reg_entry["model_type"] == "regression"


class TestClassificationValidationErrors:
    def test_high_cardinality_numeric_target_rejected(self, client, user_a):
        # "salary" has far more than 20 distinct values and isn't
        # integer-encoded class labels — must be rejected as a classification
        # target with the documented eligibility-rule error message.
        create_resp = client.post("/create-dataset", json=_classification_dataset_payload(
            name="pytest_bad_target_dataset",
        ), headers=user_a["headers"])
        dataset_id = create_resp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        resp = client.post("/classify/select-features", json={
            "dataset_id": dataset_id,
            "features": ["age", "city"],
            "target": "salary",
        }, headers=user_a["headers"])
        assert resp.status_code == 400
        assert "Classification requires a categorical target" in resp.json()["error"]["message"]

    def test_unsupported_classification_model_name(self, client, user_a, trained_classifier):
        resp = client.post("/classify/train-model", json={
            "dataset_id": trained_classifier["dataset_id"],
            "model": "NotARealClassifier",
            "features": ["age", "salary", "city"],
            "target": "outcome",
            "split": {"test_size": 0.25, "val_size": 0.0, "random_state": 42},
        }, headers=user_a["headers"])
        assert resp.status_code == 422


class TestClassificationMultiTenancy:
    def test_user_b_cannot_predict_with_user_a_classifier(self, client, user_b, trained_classifier):
        resp = client.post("/predict", json={
            "model_id": trained_classifier["model_id"],
            "values": {"age": 45, "salary": 60000, "city": "Chennai"},
        }, headers=user_b["headers"])
        assert resp.status_code == 404

    def test_user_b_cannot_see_user_a_classification_model_in_list(self, client, user_b, trained_classifier):
        resp = client.get("/models", headers=user_b["headers"])
        assert resp.status_code == 200
        model_ids = [m["model_id"] for m in resp.json()["data"]]
        assert trained_classifier["model_id"] not in model_ids


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_module():
    yield

    async def _cleanup():
        for user_id in _created["users"]:
            await db_client.delete_one("users", {"_id": user_id})
        for dataset_id in _created["datasets"]:
            dataset_store.delete(dataset_id)
        for model_id in _created["models"]:
            await db_client.delete_one("models", {"_id": model_id})
            await db_client.delete_one("reports", {"_id": f"rep_{model_id}"})
            pkl_path = os.path.join(BACKEND_DIR, "app", "models", "saved_models", f"{model_id}.pkl")
            if os.path.exists(pkl_path):
                os.remove(pkl_path)

    asyncio.run(_cleanup())
