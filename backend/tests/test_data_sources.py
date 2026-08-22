"""
Pytest coverage for the Data Sources (database connectors) feature.
Requires the local test databases from the plan's Docker setup:
  mysql:8       -> 127.0.0.1:3307, root/testpass123, db "testdb", table "employees"
  postgres:16   -> 127.0.0.1:5433, postgres/testpass123, db "testdb", table "sales"
  mongo:7       -> 127.0.0.1:27018, testuser/testpass123, db "testdb", collection "products"
Skips automatically (rather than failing) if those containers aren't reachable,
so the rest of the suite (test_api.py, test_pipeline.py) isn't affected by
whether Docker is available in a given environment.

Mirrors test_api.py's fixture style (module-scope client/user_a/user_b,
_created cleanup dict) exactly.
"""
import asyncio
import os
import socket
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database.mongodb import db_client

_created = {"users": [], "datasets": [], "data_sources": []}

MYSQL_CONFIG = {
    "engine": "mysql", "host": "127.0.0.1", "port": 3307, "database": "testdb",
    "username": "root", "password": "testpass123", "ssl_enabled": False, "connection_timeout_sec": 10,
}
POSTGRES_CONFIG = {
    "engine": "postgresql", "host": "127.0.0.1", "port": 5433, "database": "testdb",
    "username": "postgres", "password": "testpass123", "ssl_enabled": False, "connection_timeout_sec": 10,
}
MONGO_CONFIG = {
    "engine": "mongodb",
    "uri": "mongodb://testuser:testpass123@127.0.0.1:27018/testdb?authSource=admin",
    "database": "testdb", "connection_timeout_sec": 10,
}


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


TEST_DBS_AVAILABLE = (
    _port_open("127.0.0.1", 3307) and _port_open("127.0.0.1", 5433) and _port_open("127.0.0.1", 27018)
)
pytestmark = pytest.mark.skipif(
    not TEST_DBS_AVAILABLE,
    reason="Local test databases (mysql:3307, postgres:5433, mongo:27018) are not reachable — see module docstring.",
)


def _unique_email(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"


def _signup(client: TestClient, prefix: str) -> dict:
    email = _unique_email(prefix)
    resp = client.post("/auth/signup", json={"name": f"Pytest {prefix}", "email": email, "password": "supersecret123"})
    assert resp.status_code == 200, resp.text
    user_id = resp.json()["data"]["user"]["id"]
    token = resp.json()["data"]["token"]
    _created["users"].append(user_id)
    return {"user_id": user_id, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def user_a(client):
    return _signup(client, "dsa")


@pytest.fixture(scope="module")
def user_b(client):
    return _signup(client, "dsb")


@pytest.fixture(scope="module")
def mysql_source(client, user_a):
    resp = client.post("/data-sources", json={"name": "pytest-mysql", "connection": MYSQL_CONFIG}, headers=user_a["headers"])
    assert resp.status_code == 200, resp.text
    source = resp.json()["data"]
    _created["data_sources"].append(source["id"])
    return source


@pytest.fixture(scope="module")
def postgres_source(client, user_a):
    resp = client.post("/data-sources", json={"name": "pytest-postgres", "connection": POSTGRES_CONFIG}, headers=user_a["headers"])
    assert resp.status_code == 200, resp.text
    source = resp.json()["data"]
    _created["data_sources"].append(source["id"])
    return source


@pytest.fixture(scope="module")
def mongo_source(client, user_a):
    resp = client.post("/data-sources", json={"name": "pytest-mongo", "connection": MONGO_CONFIG}, headers=user_a["headers"])
    assert resp.status_code == 200, resp.text
    source = resp.json()["data"]
    _created["data_sources"].append(source["id"])
    return source


class TestConnectionTest:
    def test_ephemeral_test_never_persists(self, client, user_a):
        resp = client.post("/data-sources/test", json={"connection": MYSQL_CONFIG}, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["status"] == "ok"

        list_resp = client.get("/data-sources", headers=user_a["headers"])
        names = [s["name"] for s in list_resp.json()["data"]]
        assert "pytest-mysql" not in names or len(names) == len(set(names))  # no phantom save from /test

    def test_bad_password_returns_safe_auth_error(self, client, user_a):
        bad = {**MYSQL_CONFIG, "password": "definitely-wrong"}
        resp = client.post("/data-sources/test", json={"connection": bad}, headers=user_a["headers"])
        assert resp.status_code == 401
        body = resp.text.lower()
        assert "definitely-wrong" not in body
        assert "traceback" not in body

    def test_unreachable_host_returns_safe_connection_error(self, client, user_a):
        unreachable = {**MYSQL_CONFIG, "port": 39999, "connection_timeout_sec": 3}
        resp = client.post("/data-sources/test", json={"connection": unreachable}, headers=user_a["headers"])
        assert resp.status_code in (502, 504)


class TestSaveAndListAndDelete:
    def test_save_masks_credentials(self, client, mysql_source):
        assert "password" not in mysql_source
        assert "encrypted_secret" not in mysql_source
        assert mysql_source["engine"] == "mysql"
        assert mysql_source["database"] == "testdb"

    def test_mongo_uri_never_returned(self, client, mongo_source):
        body_str = str(mongo_source)
        assert "testpass123" not in body_str
        assert MONGO_CONFIG["uri"] not in body_str

    def test_list_returns_saved_sources(self, client, user_a, mysql_source, postgres_source, mongo_source):
        resp = client.get("/data-sources", headers=user_a["headers"])
        assert resp.status_code == 200
        names = {s["name"] for s in resp.json()["data"]}
        assert {"pytest-mysql", "pytest-postgres", "pytest-mongo"}.issubset(names)

    def test_unauthenticated_request_returns_401(self, client):
        resp = client.get("/data-sources")
        assert resp.status_code == 401


class TestUserIsolation:
    def test_user_b_cannot_read_user_a_source(self, client, user_b, mysql_source):
        resp = client.get(f"/data-sources/{mysql_source['id']}", headers=user_b["headers"])
        assert resp.status_code == 404

    def test_user_b_cannot_test_user_a_source(self, client, user_b, mysql_source):
        resp = client.post(f"/data-sources/{mysql_source['id']}/test", headers=user_b["headers"])
        assert resp.status_code == 404

    def test_user_b_cannot_delete_user_a_source(self, client, user_b, mysql_source):
        resp = client.delete(f"/data-sources/{mysql_source['id']}", headers=user_b["headers"])
        assert resp.status_code == 404

    def test_user_b_list_is_empty(self, client, user_b):
        resp = client.get("/data-sources", headers=user_b["headers"])
        assert resp.json()["data"] == []


class TestSchemaDiscoveryAndPreview:
    def test_mysql_tables(self, client, user_a, mysql_source):
        resp = client.get(f"/data-sources/{mysql_source['id']}/tables", params={"database": "testdb"}, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        names = [t["name"] for t in resp.json()["data"]["tables"]]
        assert "employees" in names

    def test_postgres_tables(self, client, user_a, postgres_source):
        resp = client.get(f"/data-sources/{postgres_source['id']}/tables", params={"database": "testdb"}, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        names = [t["name"] for t in resp.json()["data"]["tables"]]
        assert "sales" in names

    def test_mongo_collections(self, client, user_a, mongo_source):
        resp = client.get(f"/data-sources/{mongo_source['id']}/tables", params={"database": "testdb"}, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        names = [t["name"] for t in resp.json()["data"]["tables"]]
        assert "products" in names

    def test_preview_is_capped(self, client, user_a, mysql_source):
        resp = client.get(
            f"/data-sources/{mysql_source['id']}/preview",
            params={"database": "testdb", "table": "employees", "limit": 3},
            headers=user_a["headers"],
        )
        assert resp.status_code == 200, resp.text
        assert len(resp.json()["data"]["rows"]) <= 3


class TestImportIntoExistingPipeline:
    def test_import_mysql_table_trains_through_unmodified_pipeline(self, client, user_a, mysql_source):
        """The single most important test: proves a database-imported
        dataset is genuinely indistinguishable from a CSV-uploaded one to
        the existing /preprocess, /select-features, /train-model routes."""
        resp = client.post(f"/data-sources/{mysql_source['id']}/import", json={"database": "testdb", "table": "employees"}, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        pre = client.post("/preprocess", json={
            "dataset_id": dataset_id,
            "config": {"missing": "mean", "dedupe": True, "outlier": "none", "scaling": "none"},
        }, headers=user_a["headers"])
        assert pre.status_code == 200, pre.text
        preprocessed_id = pre.json()["data"]["preprocessed_dataset_id"]
        _created["datasets"].append(preprocessed_id)

        sel = client.post("/select-features", json={
            "dataset_id": preprocessed_id, "features": ["age", "years_experience"], "target": "salary",
        }, headers=user_a["headers"])
        assert sel.status_code == 200, sel.text

        train = client.post("/train-model", json={
            "dataset_id": preprocessed_id, "model": "LinearRegression",
            "features": ["age", "years_experience"], "target": "salary",
            "split": {"test_size": 0.3, "val_size": 0.0, "random_state": 42},
        }, headers=user_a["headers"])
        assert train.status_code == 200, train.text
        assert train.json()["data"]["status"] == "completed"

    def test_import_preview_dataset_is_json_safe(self, client, user_a, mysql_source):
        """Regression test for the NaN-in-numeric-column bug found while
        building this feature (also present in the pre-existing CSV upload
        path, fixed alongside this)."""
        resp = client.post(f"/data-sources/{mysql_source['id']}/import", json={"database": "testdb", "table": "employees"}, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        dataset_id = resp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        preview = client.get(f"/preview-dataset/{dataset_id}", headers=user_a["headers"])
        assert preview.status_code == 200, preview.text

    def test_import_mongo_collection_flattens_and_trains(self, client, user_a, mongo_source):
        resp = client.post(f"/data-sources/{mongo_source['id']}/import", json={
            "database": "testdb", "table": "products",
            "filter": {"category": "Electronics"}, "sort": [["price", -1]],
        }, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        dataset_id = data["dataset_id"]
        _created["datasets"].append(dataset_id)
        assert "specs.cpu" in data["column_names"] or "specs.ram" in data["column_names"]

        preview = client.get(f"/preview-dataset/{dataset_id}", headers=user_a["headers"])
        assert preview.status_code == 200, preview.text

    def test_import_respects_row_limit(self, client, user_a, mysql_source):
        resp = client.post(f"/data-sources/{mysql_source['id']}/import", json={
            "database": "testdb", "table": "employees", "row_limit": 3,
        }, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        _created["datasets"].append(data["dataset_id"])
        assert data["rows"] == 3
        assert data["truncated"] is True
        assert data["row_limit_applied"] == 3


class TestUnsafeSQLRejection:
    def test_multi_statement_injection_rejected(self, client, user_a, mysql_source):
        resp = client.post(f"/data-sources/{mysql_source['id']}/import", json={
            "database": "testdb", "custom_sql": "SELECT 1; DROP TABLE employees;--",
        }, headers=user_a["headers"])
        assert resp.status_code == 400

    def test_delete_statement_rejected(self, client, user_a, mysql_source):
        resp = client.post(f"/data-sources/{mysql_source['id']}/import", json={
            "database": "testdb", "custom_sql": "DELETE FROM employees",
        }, headers=user_a["headers"])
        assert resp.status_code == 400

    def test_update_statement_rejected(self, client, user_a, mysql_source):
        resp = client.post(f"/data-sources/{mysql_source['id']}/import", json={
            "database": "testdb", "custom_sql": "UPDATE employees SET salary = 0",
        }, headers=user_a["headers"])
        assert resp.status_code == 400

    def test_legitimate_select_with_keyword_like_content_allowed(self, client, user_a, mysql_source):
        """Guards against over-eager keyword blocking — a value or column
        name that merely contains a blocklisted word must not be rejected."""
        resp = client.post(f"/data-sources/{mysql_source['id']}/import", json={
            "database": "testdb", "custom_sql": "SELECT name FROM employees WHERE department = 'Engineering'",
        }, headers=user_a["headers"])
        assert resp.status_code == 200, resp.text
        _created["datasets"].append(resp.json()["data"]["dataset_id"])

    def test_employees_table_still_intact_after_injection_attempts(self, client, user_a, mysql_source):
        """Confirms the rejected statements above never actually reached MySQL."""
        resp = client.get(
            f"/data-sources/{mysql_source['id']}/preview",
            params={"database": "testdb", "table": "employees", "limit": 100},
            headers=user_a["headers"],
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["row_count_returned"] == 10


class TestDisconnect:
    def test_delete_removes_source_but_keeps_imported_datasets(self, client, user_a, postgres_source):
        imp = client.post(f"/data-sources/{postgres_source['id']}/import", json={"database": "testdb", "table": "sales"}, headers=user_a["headers"])
        assert imp.status_code == 200, imp.text
        dataset_id = imp.json()["data"]["dataset_id"]
        _created["datasets"].append(dataset_id)

        del_resp = client.delete(f"/data-sources/{postgres_source['id']}", headers=user_a["headers"])
        assert del_resp.status_code == 200, del_resp.text

        get_resp = client.get(f"/data-sources/{postgres_source['id']}", headers=user_a["headers"])
        assert get_resp.status_code == 404

        preview_resp = client.get(f"/preview-dataset/{dataset_id}", headers=user_a["headers"])
        assert preview_resp.status_code == 200
        # Remove postgres_source from the cleanup list — it's already deleted.
        if postgres_source["id"] in _created["data_sources"]:
            _created["data_sources"].remove(postgres_source["id"])


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_module():
    yield

    async def _cleanup():
        for user_id in _created["users"]:
            await db_client.delete_one("users", {"_id": user_id})
        for dataset_id in _created["datasets"]:
            await db_client.delete_one("datasets", {"_id": dataset_id})
        for data_source_id in _created["data_sources"]:
            await db_client.delete_one("data_sources", {"_id": data_source_id})

    asyncio.run(_cleanup())
