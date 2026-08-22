"""
Pytest + FastAPI TestClient coverage for the API layer, added alongside (not
replacing) test_pipeline.py's standalone ML-pipeline script.

Covers: happy path (upload->preprocess->train->predict), bad target column,
unsupported model name, unknown model_id on predict, malformed predict
payload, and multi-tenant data isolation between two users.

Every record this suite creates is deleted again in the module-scope cleanup
fixture below, so re-running the suite never leaves stale data in db.json.
"""
import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database.mongodb import db_client

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_created = {"users": [], "datasets": [], "models": []}


def _linear_dataset_payload(name="pytest_dataset"):
    rows = []
    for i in range(25):
        x1 = float(i)
        x2 = float(i * 2 + 1)
        y = 3 * x1 + 2 * x2 + 5
        rows.append([x1, x2, y])
    return {"name": name, "columns": ["x1", "x2", "y"], "rows": rows}


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
    assert body["success"] is True
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
    return _signup(client, "usera")


@pytest.fixture(scope="module")
def user_b(client):
    return _signup(client, "userb")


@pytest.fixture(scope="module")
def trained_model(client, user_a):
    """Runs the real upload -> preprocess -> train pipeline once and shares the
    resulting model_id across the predict-focused tests."""
    headers = user_a["headers"]

    create_resp = client.post("/create-dataset", json=_linear_dataset_payload(), headers=headers)
    assert create_resp.status_code == 200, create_resp.text
    dataset_id = create_resp.json()["data"]["dataset_id"]
    _created["datasets"].append(dataset_id)

    preprocess_resp = client.post("/preprocess", json={
        "dataset_id": dataset_id,
        "config": {"missing": "mean", "dedupe": True, "outlier": "none", "scaling": "standard"},
    }, headers=headers)
    assert preprocess_resp.status_code == 200, preprocess_resp.text
    preprocessed_id = preprocess_resp.json()["data"]["preprocessed_dataset_id"]
    _created["datasets"].append(preprocessed_id)

    train_resp = client.post("/train-model", json={
        "dataset_id": preprocessed_id,
        "model": "LinearRegression",
        "features": ["x1", "x2"],
        "target": "y",
        "split": {"test_size": 0.2, "val_size": 0.0, "random_state": 42},
    }, headers=headers)
    assert train_resp.status_code == 200, train_resp.text
    train_data = train_resp.json()["data"]
    model_id = train_data["model_id"]
    _created["models"].append(model_id)

    return {
        "dataset_id": dataset_id,
        "preprocessed_id": preprocessed_id,
        "model_id": model_id,
        "metrics": train_data["metrics"],
    }


class TestHappyPath:
    def test_full_pipeline_produces_real_metrics(self, trained_model):
        metrics = trained_model["metrics"]
        # A perfectly linear synthetic target should fit essentially exactly —
        # this is also a regression check that scaling deferred to train time
        # doesn't break a simple LinearRegression fit.
        assert metrics["R2"] > 0.99
        assert metrics["RMSE"] < 1.0

    def test_predict_with_valid_model(self, client, user_a, trained_model):
        resp = client.post("/predict", json={
            "model_id": trained_model["model_id"],
            "values": {"x1": 10.0, "x2": 21.0},
        }, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        # y = 3*x1 + 2*x2 + 5 = 3*10 + 2*21 + 5 = 77
        assert abs(body["data"]["prediction"] - 77.0) < 1.0


class TestValidationErrors:
    def test_bad_target_column_returns_400(self, client, user_a):
        create_resp = client.post("/create-dataset", json=_linear_dataset_payload("pytest_bad_target"),
                                   headers=user_a["headers"])
        dataset_id = create_resp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        resp = client.post("/select-features", json={
            "dataset_id": dataset_id,
            "features": ["x1", "x2"],
            "target": "not_a_real_column",
        }, headers=user_a["headers"])
        assert resp.status_code == 400
        assert resp.json()["success"] is False

    def test_unsupported_model_name_returns_422(self, client, user_a):
        create_resp = client.post("/create-dataset", json=_linear_dataset_payload("pytest_bad_model"),
                                   headers=user_a["headers"])
        dataset_id = create_resp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        resp = client.post("/train-model", json={
            "dataset_id": dataset_id,
            "model": "TotallyFakeModelXYZ",
            "features": ["x1", "x2"],
            "target": "y",
        }, headers=user_a["headers"])
        assert resp.status_code == 422
        assert resp.json()["success"] is False

    def test_predict_unknown_model_id_returns_404(self, client, user_a):
        resp = client.post("/predict", json={
            "model_id": "does-not-exist",
            "values": {"x1": 1.0, "x2": 2.0},
        }, headers=user_a["headers"])
        assert resp.status_code == 404
        assert resp.json()["success"] is False

    def test_predict_missing_feature_returns_400(self, client, user_a, trained_model):
        resp = client.post("/predict", json={
            "model_id": trained_model["model_id"],
            "values": {"x1": 1.0},  # missing required "x2"
        }, headers=user_a["headers"])
        assert resp.status_code == 400
        assert resp.json()["success"] is False


class TestAuth:
    def test_signup_duplicate_email_returns_409(self, client):
        email = _unique_email("dupe")
        payload = {"name": "Dupe", "email": email, "password": "supersecret123"}
        first = client.post("/auth/signup", json=payload)
        assert first.status_code == 200
        _created["users"].append(first.json()["data"]["user"]["id"])

        second = client.post("/auth/signup", json=payload)
        assert second.status_code == 409

    def test_login_wrong_password_returns_401(self, client):
        email = _unique_email("wrongpw")
        signup_resp = client.post("/auth/signup", json={
            "name": "Wrong Password",
            "email": email,
            "password": "correct-password-123",
        })
        assert signup_resp.status_code == 200
        _created["users"].append(signup_resp.json()["data"]["user"]["id"])

        login_resp = client.post("/auth/login", json={
            "email": email,
            "password": "totally-wrong-password",
        })
        assert login_resp.status_code == 401
        assert login_resp.json()["success"] is False

    def test_protected_route_without_token_returns_401(self, client):
        resp = client.get("/datasets")
        assert resp.status_code == 401


class TestMultiTenancy:
    def test_user_b_cannot_see_user_a_datasets(self, client, user_a, user_b, trained_model):
        resp = client.get("/datasets", headers=user_b["headers"])
        assert resp.status_code == 200
        dataset_ids = [d["dataset_id"] for d in resp.json()["data"]]
        assert trained_model["dataset_id"] not in dataset_ids

    def test_user_b_cannot_fetch_user_a_dataset_directly(self, client, user_b, trained_model):
        resp = client.get(f"/preview-dataset/{trained_model['dataset_id']}", headers=user_b["headers"])
        assert resp.status_code == 404

    def test_user_b_cannot_predict_with_user_a_model(self, client, user_b, trained_model):
        resp = client.post("/predict", json={
            "model_id": trained_model["model_id"],
            "values": {"x1": 1.0, "x2": 2.0},
        }, headers=user_b["headers"])
        assert resp.status_code == 404


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_module():
    yield

    async def _cleanup():
        for user_id in _created["users"]:
            await db_client.delete_one("users", {"_id": user_id})
        for dataset_id in _created["datasets"]:
            await db_client.delete_one("datasets", {"_id": dataset_id})
        for model_id in _created["models"]:
            await db_client.delete_one("models", {"_id": model_id})
            await db_client.delete_one("reports", {"_id": f"rep_{model_id}"})
            pkl_path = os.path.join(BACKEND_DIR, "app", "models", "saved_models", f"{model_id}.pkl")
            if os.path.exists(pkl_path):
                os.remove(pkl_path)

    asyncio.run(_cleanup())
