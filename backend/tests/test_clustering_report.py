import os
import asyncio
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.database.database import db_client
from app.reports.clustering_report import validate_clustering_report, build_clustering_pdf_report
from app.ml.feature_types import classify_columns

_created = {"users": [], "datasets": [], "models": []}

def _clustering_dataset_payload():
    # 40 columns: 37 numerical, 3 categorical
    rows = []
    for i in range(30):
        row = [
            float(i), # Roll (identifier)
            float(100 - i), # Random Roll (identifier)
        ]
        # 35 more numerical columns
        for col_idx in range(35):
            row.append(float(i * col_idx + (col_idx % 3)))
        # 3 categorical columns
        row.append("Group A" if i % 2 == 0 else "Group B")
        row.append("Type X" if i % 3 == 0 else "Type Y")
        row.append("Cat Z" if i % 4 == 0 else "Cat W")
        rows.append(row)
        
    columns = ["Roll", "Random Roll"] + [f"Feature_{idx}" for idx in range(35)] + ["Cat_1", "Cat_2", "Cat_3"]
    return {"name": "clustering_test_dataset", "columns": columns, "rows": rows}

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def test_user(client):
    import uuid
    email = f"cluster_test_{uuid.uuid4().hex[:8]}@example.com"
    resp = client.post("/auth/signup", json={
        "name": "Cluster Test User",
        "email": email,
        "password": "supersecret123",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    user_id = body["data"]["user"]["id"]
    token = body["data"]["token"]
    _created["users"].append(user_id)
    return {"user_id": user_id, "headers": {"Authorization": f"Bearer {token}"}}

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

def test_identifier_detection():
    # Test identifier detection function
    df = pd.DataFrame({
        "Roll": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Random Roll": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "Value_Col": [1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 1.5],
        "Name": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    })
    classification = classify_columns(df)
    assert classification["Roll"]["is_identifier"] is True
    assert classification["Random Roll"]["is_identifier"] is True
    assert classification["Student_ID"]["is_identifier"] is True
    assert classification["Value_Col"]["is_identifier"] is False
    assert classification["Name"]["is_identifier"] is True # High uniqueness object col is an identifier

def test_select_features_warnings(client, test_user):
    headers = test_user["headers"]
    
    # 1. Create dataset
    create_resp = client.post("/create-dataset", json=_clustering_dataset_payload(), headers=headers)
    assert create_resp.status_code == 200
    dataset_id = create_resp.json()["data"]["dataset_id"]
    _created["datasets"].append(dataset_id)
    
    # 2. Call select-features with Roll and Random Roll explicitly selected
    select_resp = client.post("/cluster/select-features", json={
        "dataset_id": dataset_id,
        "features": ["Roll", "Random Roll", "Feature_0", "Feature_1"]
    }, headers=headers)
    assert select_resp.status_code == 200
    warnings = select_resp.json()["data"]["warnings"]
    assert len(warnings) >= 2
    assert any("Roll" in w for w in warnings)

def test_clustering_training_and_report(client, test_user):
    headers = test_user["headers"]
    dataset_id = _created["datasets"][0]
    
    # 1. Train Agglomerative Clustering
    train_resp = client.post("/cluster/train-model", json={
        "dataset_id": dataset_id,
        "model": "Agglomerative Clustering",
        "features": ["Feature_0", "Feature_1", "Feature_2"],
        "hyperparameters": {
            "n_clusters": 3,
            "linkage": "ward"
        }
    }, headers=headers)
    assert train_resp.status_code == 200, train_resp.text
    model_id = train_resp.json()["data"]["model_id"]
    _created["models"].append(model_id)
    
    # 2. Download report PDF
    report_resp = client.get(f"/report/{model_id}", headers=headers)
    assert report_resp.status_code == 200
    assert report_resp.headers["content-type"] == "application/pdf"
    
def test_consistency_validation_errors():
    # Test validate_clustering_report throws on mismatches
    valid_info = {
        "dataset_profile": {"total_rows": 100, "total_columns": 5},
        "total_rows": 100,
        "features": ["f1", "f2"],
        "metrics": {"ClusterCount": 2},
        "cluster_sizes": {"0": 40, "1": 60}
    }
    
    # Should pass without error
    validate_clustering_report(valid_info)
    
    # Row mismatch
    mismatch_rows = valid_info.copy()
    mismatch_rows["total_rows"] = 99
    with pytest.raises(ValueError, match="Dataset row count mismatch"):
        validate_clustering_report(mismatch_rows)
        
    # Cluster count mismatch
    mismatch_clusters = valid_info.copy()
    mismatch_clusters["metrics"] = {"ClusterCount": 3}
    with pytest.raises(ValueError, match="Cluster count mismatch"):
        validate_clustering_report(mismatch_clusters)

    # Size sum mismatch
    mismatch_sum = valid_info.copy()
    mismatch_sum["cluster_sizes"] = {"0": 40, "1": 59}
    with pytest.raises(ValueError, match="Sum of cluster sizes"):
        validate_clustering_report(mismatch_sum)
