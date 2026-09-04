"""
Coverage for the AI Insights / Report / Visualization performance work:
- run_ai_explanation_pipeline caches its result and skips the agent pipeline
  entirely on a repeat call for the same model_id.
- chart_data's interactive-chart arrays are capped for rendering while
  metrics stay computed on the full test split.
- /report/{model_id} reuses a previously generated PDF instead of
  re-rendering it on every download.

Follows the same TestClient + module-scoped user/cleanup conventions as
test_classification_api.py. Every record created here is removed in the
module-scope cleanup fixture, including the report_{model_id}.pdf files this
suite specifically generates (not covered by the shared /models/{id} DELETE
route cleanup, since these tests call db_client directly).
"""
import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database.database import db_client
from app.datasets import store as dataset_store
import app.agents.dataset_agent as dataset_agent_module
import app.api.training_routes as training_routes_module
import app.api.classify_routes as classify_routes_module
import app.api.report_routes as report_routes_module

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_REPORTS_DIR = os.path.join(BACKEND_DIR, "static", "reports")

_created = {"users": [], "datasets": [], "models": []}


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


def _regression_dataset_payload(name: str, n: int = 60):
    rows = [[float(i), float(i * 2 + 1), 3.0 * i + 5.0 + (i % 7)] for i in range(n)]
    return {"name": name, "columns": ["x1", "x2", "y"], "rows": rows}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def user_a(client):
    return _signup(client, "perf_usera")


def _train_regression_model(client, headers, n=60, test_size=0.25, name=None):
    create_resp = client.post(
        "/create-dataset",
        json=_regression_dataset_payload(name or f"pytest_perf_dataset_{uuid.uuid4().hex[:6]}", n=n),
        headers=headers,
    )
    assert create_resp.status_code == 200, create_resp.text
    dataset_id = create_resp.json()["data"]["dataset_id"]
    _created["datasets"].append(dataset_id)

    train_resp = client.post("/train-model", json={
        "dataset_id": dataset_id,
        "model": "LinearRegression",
        "features": ["x1", "x2"],
        "target": "y",
        "split": {"test_size": test_size, "val_size": 0.0, "random_state": 42},
    }, headers=headers)
    assert train_resp.status_code == 200, train_resp.text
    model_id = train_resp.json()["data"]["model_id"]
    _created["models"].append(model_id)
    return dataset_id, model_id


class TestAiInsightsCaching:
    def test_second_call_reuses_cached_result_and_skips_agents(self, client, user_a, monkeypatch):
        _, model_id = _train_regression_model(client, user_a["headers"])

        original_analyze = dataset_agent_module.DatasetAnalysisAgent.analyze
        call_count = {"n": 0}

        async def counting_analyze(self, *args, **kwargs):
            call_count["n"] += 1
            return await original_analyze(self, *args, **kwargs)

        monkeypatch.setattr(dataset_agent_module.DatasetAnalysisAgent, "analyze", counting_analyze)

        resp1 = client.post("/ai/explain", json={"model_id": model_id}, headers=user_a["headers"])
        assert resp1.status_code == 200, resp1.text
        resp2 = client.post("/ai/explain", json={"model_id": model_id}, headers=user_a["headers"])
        assert resp2.status_code == 200, resp2.text

        assert resp1.json()["data"] == resp2.json()["data"]
        assert call_count["n"] == 1, "agent pipeline must not re-run on a cached model_id"

    def test_force_true_bypasses_cache_and_regenerates(self, client, user_a, monkeypatch):
        _, model_id = _train_regression_model(client, user_a["headers"])

        original_analyze = dataset_agent_module.DatasetAnalysisAgent.analyze
        call_count = {"n": 0}

        async def counting_analyze(self, *args, **kwargs):
            call_count["n"] += 1
            return await original_analyze(self, *args, **kwargs)

        monkeypatch.setattr(dataset_agent_module.DatasetAnalysisAgent, "analyze", counting_analyze)

        resp1 = client.post("/ai/explain", json={"model_id": model_id}, headers=user_a["headers"])
        assert resp1.status_code == 200, resp1.text
        resp2 = client.post(
            "/ai/explain", json={"model_id": model_id, "force": True}, headers=user_a["headers"],
        )
        assert resp2.status_code == 200, resp2.text

        assert call_count["n"] == 2, "force=True must bypass the cache and re-run the pipeline"


class TestChartDataSampling:
    def test_regression_chart_points_capped_metrics_unaffected(self, client, user_a, monkeypatch):
        original = training_routes_module.sample_paired_series

        def small_cap(*args, **kwargs):
            kwargs["max_points"] = 10
            return original(*args, **kwargs)

        monkeypatch.setattr(training_routes_module, "sample_paired_series", small_cap)

        # test_size=0.3 on 60 rows -> 18 test rows, comfortably over the
        # forced 10-point cap above.
        _, model_id = _train_regression_model(client, user_a["headers"], n=60, test_size=0.3)

        resp = client.get(f"/model-metrics/{model_id}", headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]

        assert len(data["chart_data"]["actual"]) == 10
        assert len(data["chart_data"]["predicted"]) == 10
        assert len(data["chart_data"]["residuals"]) == 10
        assert data["chart_data"]["chart_points_sampled"] is True
        assert data["chart_data"]["chart_points_total"] == 18

        # Metrics must be computed on the FULL test split, not the capped
        # chart arrays — a real numeric R2, not None/NaN-sanitized-away.
        assert isinstance(data["metrics"]["R2"], float)

    def test_classification_chart_points_capped(self, client, user_a, monkeypatch):
        original = classify_routes_module.sample_paired_series

        def small_cap(*args, **kwargs):
            kwargs["max_points"] = 8
            return original(*args, **kwargs)

        monkeypatch.setattr(classify_routes_module, "sample_paired_series", small_cap)

        columns = ["age", "salary", "outcome"]
        rows = []
        for i in range(80):
            age = 20 + (i % 45)
            salary = 20000 + (i * 733) % 70000
            outcome = "approved" if (age > 40 and salary > 50000) else "denied"
            rows.append([age, salary, outcome])
        create_resp = client.post("/create-dataset", json={
            "name": f"pytest_perf_clf_{uuid.uuid4().hex[:6]}", "columns": columns, "rows": rows,
        }, headers=user_a["headers"])
        assert create_resp.status_code == 200, create_resp.text
        dataset_id = create_resp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        train_resp = client.post("/classify/train-model", json={
            "dataset_id": dataset_id,
            "model": "LogisticRegression",
            "features": ["age", "salary"],
            "target": "outcome",
            "split": {"test_size": 0.3, "val_size": 0.0, "random_state": 42},
        }, headers=user_a["headers"])
        assert train_resp.status_code == 200, train_resp.text
        model_id = train_resp.json()["data"]["model_id"]
        _created["models"].append(model_id)

        resp = client.get(f"/model-metrics/{model_id}", headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        chart_data = resp.json()["data"]["chart_data"]
        assert len(chart_data["actual"]) == 8
        assert chart_data["chart_points_sampled"] is True
        assert chart_data["chart_points_total"] == 24  # 80 * 0.3


class TestReportPdfCaching:
    def test_second_download_reuses_cached_pdf(self, client, user_a, monkeypatch):
        _, model_id = _train_regression_model(client, user_a["headers"])

        original_build = report_routes_module.build_pdf_report
        call_count = {"n": 0}

        def counting_build(*args, **kwargs):
            call_count["n"] += 1
            return original_build(*args, **kwargs)

        monkeypatch.setattr(report_routes_module, "build_pdf_report", counting_build)

        try:
            resp1 = client.get(f"/report/{model_id}", headers=user_a["headers"])
            assert resp1.status_code == 200, resp1.text
            resp2 = client.get(f"/report/{model_id}", headers=user_a["headers"])
            assert resp2.status_code == 200, resp2.text

            assert call_count["n"] == 1, "the cached PDF must be served, not re-rendered"
        finally:
            pdf_path = os.path.join(STATIC_REPORTS_DIR, f"report_{model_id}.pdf")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)


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
            pdf_path = os.path.join(STATIC_REPORTS_DIR, f"report_{model_id}.pdf")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    asyncio.run(_cleanup())
