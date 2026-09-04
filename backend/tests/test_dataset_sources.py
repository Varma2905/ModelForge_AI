"""Tests for the dataset ingestion redesign (DatasetStore + providers).

Split by verification tier, per the implementation plan:
  - DatasetStore, ingest_source's extension dispatch, and
    KaggleProvider.parse_ref are pure/local logic — genuinely exercised
    here, no mocking needed.
  - Kaggle/Google Drive orchestration (resolve -> list -> pick -> download
    -> ingest) is exercised against MOCKED provider internals — there is no
    live Kaggle token or Google OAuth app available in this environment, so
    these tests prove the orchestration and gating logic, not that the real
    Kaggle/Google APIs behave as expected.
"""
import asyncio
import json
import os
import shutil
import tempfile
import zipfile
from unittest import mock

import pandas as pd
import pytest

from app.datasets import store
from app.datasets.ingest import _read_dataframe, ingest_source, SUPPORTED_EXTENSIONS
from app.datasets.providers.base import DatasetSource
from app.datasets.providers.kaggle_provider import KaggleCredentialsError, KaggleProvider
from app.datasets.providers import gdrive_sessions


# ---------------------------------------------------------------------------
# DatasetStore — real, no mocking
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_store(monkeypatch, tmp_path):
    """Points DatasetStore at a throwaway directory for each test so runs
    never pollute (or depend on) the real backend/app/data/datasets/."""
    monkeypatch.setattr(store, "DATASETS_DIR", str(tmp_path))
    monkeypatch.setattr(store, "_INDEX_PATH", str(tmp_path / "_index.json"))
    monkeypatch.setattr(store, "_index_cache", None)
    yield tmp_path


def test_save_load_roundtrip(clean_store):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    dataset_id = store.save(df, {"user_id": "u1", "name": "test.csv", "source": "local"})

    loaded = store.load_dataframe(dataset_id)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 3

    meta = store.load_meta(dataset_id)
    assert meta["user_id"] == "u1"
    assert meta["row_count"] == 3
    assert meta["col_count"] == 2
    assert meta["source"] == "local"


def test_save_mixed_type_object_column_roundtrips_exactly(clean_store):
    # Pickle (unlike pyarrow/Parquet) has no strict per-column type
    # inference, so a mixed-type object column from the manual spreadsheet
    # editor round-trips without any coercion.
    df = pd.DataFrame({"a": [1, "x", 3.5], "b": [1, 2, 3]})
    dataset_id = store.save(df, {"user_id": "u1", "name": "mixed.csv", "source": "manual"})
    loaded = store.load_dataframe(dataset_id)
    assert loaded["a"].tolist() == [1, "x", 3.5]


def test_list_for_user_scoped_and_delete(clean_store):
    id1 = store.save(pd.DataFrame({"a": [1]}), {"user_id": "u1", "name": "one", "source": "local"})
    store.save(pd.DataFrame({"a": [1]}), {"user_id": "u2", "name": "two", "source": "local"})

    listed = store.list_for_user("u1")
    assert len(listed) == 1
    assert listed[0]["dataset_id"] == id1

    assert store.delete(id1) is True
    assert store.load_meta(id1) is None
    assert store.list_for_user("u1") == []


def test_load_dataframe_missing_raises():
    with pytest.raises(store.DatasetNotFoundError):
        store.load_dataframe("does-not-exist")


def test_index_self_heals_from_meta_files(clean_store, monkeypatch):
    df = pd.DataFrame({"a": [1, 2]})
    dataset_id = store.save(df, {"user_id": "u1", "name": "a.csv", "source": "local"})

    # Simulate a lost/corrupted index: delete it and drop the in-memory
    # cache, then confirm list_for_user rebuilds from the *.meta.json files.
    os.remove(store._INDEX_PATH)
    monkeypatch.setattr(store, "_index_cache", None)

    rebuilt = store.list_for_user("u1")
    assert len(rebuilt) == 1
    assert rebuilt[0]["dataset_id"] == dataset_id


# ---------------------------------------------------------------------------
# ingest.py — extension dispatch, real (no network)
# ---------------------------------------------------------------------------

def test_read_dataframe_csv_and_json(tmp_path):
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n")
    df = _read_dataframe(str(csv_path), "d.csv")
    assert list(df.columns) == ["a", "b"]

    json_path = tmp_path / "d.json"
    json_path.write_text(json.dumps([{"a": 1, "b": 2}, {"a": 3, "b": 4}]))
    df2 = _read_dataframe(str(json_path), "d.json")
    assert list(df2.columns) == ["a", "b"]


def test_read_dataframe_unsupported_extension_raises_valueerror(tmp_path):
    bad_path = tmp_path / "d.txt"
    bad_path.write_text("not a dataset")
    with pytest.raises(ValueError, match="Unsupported file type"):
        _read_dataframe(str(bad_path), "d.txt")


def test_read_dataframe_corrupt_file_raises_valueerror_not_traceback(tmp_path):
    bad_path = tmp_path / "d.parquet"
    bad_path.write_text("this is not a real parquet file")
    with pytest.raises(ValueError, match="Failed to parse"):
        _read_dataframe(str(bad_path), "d.parquet")


def test_ingest_source_cleans_up_transient_file(clean_store, tmp_path):
    # asyncio.run() rather than @pytest.mark.asyncio — keeps this test file
    # from requiring the pytest-asyncio plugin, which isn't a declared
    # project dependency.
    csv_path = tmp_path / "upload.csv"
    csv_path.write_text("a,b\n1,2\n")
    source = DatasetSource(
        source_type="local", file_name="upload.csv", file_path=str(csv_path), file_size=10
    )

    result = asyncio.run(ingest_source(source, user_id="u1"))
    assert result["rows"] == 1
    assert result["source"] == "local"
    # The transient staged file must be gone after ingestion.
    assert not os.path.exists(csv_path)


# ---------------------------------------------------------------------------
# KaggleProvider.parse_ref — real, pure function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("owner/slug", "owner/slug"),
    ("https://www.kaggle.com/datasets/owner/slug", "owner/slug"),
    ("http://kaggle.com/datasets/owner/slug/", "owner/slug"),
    ("www.kaggle.com/datasets/owner-name/slug.name", "owner-name/slug.name"),
    ("https://www.kaggle.com/datasets/owner/slug/data?select=file.csv", "owner/slug"),
])
def test_parse_ref_valid(raw, expected):
    assert KaggleProvider.parse_ref(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not a ref at all", "https://kaggle.com/models/owner/slug"])
def test_parse_ref_invalid(raw):
    with pytest.raises(ValueError):
        KaggleProvider.parse_ref(raw)


# ---------------------------------------------------------------------------
# KaggleProvider orchestration — MOCKED (no live Kaggle token available)
# ---------------------------------------------------------------------------

def test_kaggle_is_configured_always_true():
    # Kaggle has no server-wide credential to gate on any more — every user
    # connects their own account, so the *feature* is always offered; the
    # per-user connection state is checked at the route layer instead (see
    # test_kaggle_routes_require_connection_below).
    assert KaggleProvider().is_configured() is True


def test_kaggle_client_does_not_leak_credentials_between_instances():
    """Regression test for a real footgun found while implementing this:
    kaggle.api.kaggle_api_extended.KaggleApi declares `config_values` as a
    CLASS-level mutable dict (not set in __init__). Naively writing
    `api.config_values['username'] = x` without first reassigning a fresh
    dict would mutate that shared class attribute — leaking one user's
    Kaggle credentials into every other concurrent KaggleApi() instance.
    KaggleProvider._client() must assign a brand-new dict per call."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    provider = KaggleProvider()
    api_a = provider._client("user-a", "key-a")
    api_b = provider._client("user-b", "key-b")

    assert api_a.config_values[KaggleApi.CONFIG_NAME_USER] == "user-a"
    assert api_b.config_values[KaggleApi.CONFIG_NAME_USER] == "user-b"
    # The class-level dict itself must remain untouched.
    assert KaggleApi.config_values == {}


@pytest.mark.parametrize("status_code,expected_category", [
    (401, "invalid_credentials"),
    (403, "forbidden"),
    (400, "bad_request"),
    (429, "rate_limited"),
    (500, "kaggle_unavailable"),
    (503, "kaggle_unavailable"),
    (418, "unknown"),
])
def test_validate_credentials_categorizes_http_errors(status_code, expected_category):
    """Direct unit coverage of the categorization logic itself — this is the
    actual bug-fix code, independent of the route-layer status mapping
    tested separately above. 401 and 403 are deliberately distinct: 401 means
    the credentials themselves were rejected, 403 means they were accepted
    but lack permission for the requested operation — conflating the two
    was part of the original misleading-message bug."""
    import requests
    from unittest import mock as _mock

    fake_response = _mock.Mock(status_code=status_code, url="https://api.kaggle.com/v1/x", text="some error body")
    http_error = requests.exceptions.HTTPError(response=fake_response)

    fake_api = mock.Mock()
    fake_api.dataset_list.side_effect = http_error

    provider = KaggleProvider()
    with mock.patch.object(KaggleProvider, "_client", return_value=fake_api):
        with pytest.raises(KaggleCredentialsError) as exc_info:
            provider.validate_credentials("user", "key")

    assert exc_info.value.category == expected_category


def test_validate_credentials_uses_dataset_operation_not_competitions():
    """Regression test: validation must probe dataset access (what this app
    actually needs), not competitions_list — Kaggle gates competition
    participation behind requirements (rules acceptance, phone verification
    for some competitions) unrelated to dataset API access, which could
    reject a perfectly good, dataset-capable key for the wrong reason."""
    fake_api = mock.Mock()

    provider = KaggleProvider()
    with mock.patch.object(KaggleProvider, "_client", return_value=fake_api):
        provider.validate_credentials("user", "key")

    fake_api.dataset_list.assert_called_once()
    fake_api.competitions_list.assert_not_called()


def test_validate_credentials_network_error_is_kaggle_unavailable():
    import requests

    fake_api = mock.Mock()
    fake_api.dataset_list.side_effect = requests.exceptions.ConnectionError("no route to host")

    provider = KaggleProvider()
    with mock.patch.object(KaggleProvider, "_client", return_value=fake_api):
        with pytest.raises(KaggleCredentialsError) as exc_info:
            provider.validate_credentials("user", "key")

    assert exc_info.value.category == "kaggle_unavailable"


def test_validate_credentials_unexpected_error_is_not_blamed_on_credentials():
    """A bug in our own request construction (or any non-HTTP exception)
    must never be reported as 'your credentials are wrong' — that was
    exactly the misleading behavior being fixed."""
    provider = KaggleProvider()
    with mock.patch.object(KaggleProvider, "_client", side_effect=RuntimeError("unexpected SDK error")):
        with pytest.raises(KaggleCredentialsError) as exc_info:
            provider.validate_credentials("user", "key")

    assert exc_info.value.category == "unknown"
    # Explains an unexpected error occurred — must not claim the
    # credentials themselves are wrong/invalid/rejected, since we don't
    # actually know that.
    message = str(exc_info.value).lower()
    assert "invalid" not in message and "rejected" not in message


def test_kaggle_list_files_filters_unsupported_and_never_downloads():
    provider = KaggleProvider()
    fake_file_csv = mock.Mock()
    fake_file_csv.name = "data.csv"
    fake_file_csv.total_bytes = 100
    fake_file_readme = mock.Mock()
    fake_file_readme.name = "README.md"
    fake_file_readme.total_bytes = 10

    fake_api = mock.Mock()
    fake_api.dataset_list_files.return_value = mock.Mock(dataset_files=[fake_file_csv, fake_file_readme])

    with mock.patch.object(KaggleProvider, "_client", return_value=fake_api):
        files = provider.list_files("user", "key", "owner/slug")

    assert files == [{"name": "data.csv", "size": 100}]
    fake_api.dataset_download_file.assert_not_called()


def test_kaggle_download_file_orchestrates_and_cleans_up_on_failure(tmp_path):
    provider = KaggleProvider()
    fake_api = mock.Mock()

    def _fake_download(dataset_ref, file_name, path, force):
        # Simulate Kaggle writing an unsupported file type.
        with open(os.path.join(path, file_name), "w") as f:
            f.write("not real")

    fake_api.dataset_download_file.side_effect = _fake_download

    captured_dirs = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        captured_dirs.append(d)
        return d

    with mock.patch.object(KaggleProvider, "_client", return_value=fake_api), \
         mock.patch("tempfile.mkdtemp", side_effect=_tracking_mkdtemp):
        with pytest.raises(ValueError, match="isn't supported"):
            provider.download_file("user", "key", "owner/slug", "readme.txt")

    fake_api.dataset_download_file.assert_called_once()
    # The temp dir must be cleaned up after the rejected download.
    assert captured_dirs and not os.path.exists(captured_dirs[0])


def test_kaggle_download_file_finds_url_encoded_zip_name():
    """Regression test for a real bug hit in production: Kaggle writes
    downloaded files URL-encoded (e.g. "Mental Health Dataset.csv" is saved
    to disk as "Mental%20Health%20Dataset.csv.zip"), but download_file used
    to build the expected path by concatenating the literal requested
    file_name — which never matched, so every dataset with a space (or other
    URL-escaped character) in its filename failed with "did not produce the
    expected file" even though the download itself succeeded. The fix
    discovers whatever Kaggle actually wrote in the fresh temp dir instead
    of predicting its name."""
    provider = KaggleProvider()
    fake_api = mock.Mock()
    requested_name = "Mental Health Dataset.csv"

    def _fake_download(dataset_ref, file_name, path, force):
        # Simulate Kaggle's real, observed behavior: URL-encoded name, zipped.
        zip_path = os.path.join(path, "Mental%20Health%20Dataset.csv.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(requested_name, "col1,col2\n1,2\n")

    fake_api.dataset_download_file.side_effect = _fake_download

    with mock.patch.object(KaggleProvider, "_client", return_value=fake_api):
        source = provider.download_file("user", "key", "owner/slug", requested_name)

    assert os.path.basename(source.file_path) == requested_name
    assert os.path.exists(source.file_path)
    with open(source.file_path) as f:
        assert f.read() == "col1,col2\n1,2\n"

    # Cleanup — this test doesn't go through ingest_source, which normally
    # removes the transient file after ingestion.
    shutil.rmtree(os.path.dirname(source.file_path), ignore_errors=True)


def _kaggle_test_client(monkeypatch, connection=None):
    """Builds an isolated TestClient for dataset_source_routes with
    get_current_user and integrations_store.get_connection mocked out —
    avoids needing a real encryption key or db_client round-trip for
    route-level gating tests."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.api import dataset_source_routes
    from app.auth.dependencies import get_current_user

    app = FastAPI()
    app.include_router(dataset_source_routes.router)

    async def _fake_user():
        return {"_id": "u1"}

    app.dependency_overrides[get_current_user] = _fake_user

    async def _fake_get_connection(user_id, provider):
        return connection

    monkeypatch.setattr(dataset_source_routes.integrations_store, "get_connection", _fake_get_connection)
    return TestClient(app)


def test_kaggle_routes_require_connection_when_not_connected(monkeypatch):
    """Confirms resolve/import short-circuit with a clear 'connect first'
    error before any real Kaggle call, when this user has no stored
    connection — the per-user equivalent of the old server-wide gate."""
    client = _kaggle_test_client(monkeypatch, connection=None)

    resp = client.post("/datasets/kaggle/resolve", json={"dataset_ref": "owner/slug"})
    assert resp.status_code == 400
    assert "Connect your Kaggle account" in resp.json()["detail"]

    resp2 = client.post("/datasets/kaggle/import", json={"dataset_ref": "owner/slug", "file_name": "d.csv"})
    assert resp2.status_code == 400


def test_kaggle_status_reflects_connection_state(monkeypatch):
    client = _kaggle_test_client(monkeypatch, connection=None)
    resp = client.get("/datasets/kaggle/status")
    assert resp.json()["data"] == {"connected": False, "kaggle_username": None}

    client2 = _kaggle_test_client(
        monkeypatch, connection={"credentials": {"username": "alice", "key": "k"}, "provider_user_id": "alice"}
    )
    resp2 = client2.get("/datasets/kaggle/status")
    assert resp2.json()["data"] == {"connected": True, "kaggle_username": "alice"}


def test_kaggle_connect_validates_before_storing(monkeypatch):
    from app.api import dataset_source_routes
    from app.datasets.providers.kaggle_provider import KaggleCredentialsError

    client = _kaggle_test_client(monkeypatch, connection=None)

    save_calls = []

    async def _fake_save_connection(**kwargs):
        save_calls.append(kwargs)

    def _raise_invalid(self, username, key):
        raise KaggleCredentialsError("Invalid Kaggle credentials.", category="invalid_credentials")

    monkeypatch.setattr(dataset_source_routes.integrations_store, "save_connection", _fake_save_connection)
    monkeypatch.setattr(dataset_source_routes.KaggleProvider, "validate_credentials", _raise_invalid)

    resp = client.post("/datasets/kaggle/connect", json={"username": "alice", "key": "bad-key"})
    # NOT 401 — that status is reserved for platform-JWT failures. The
    # frontend's shared unwrap() clears the user's platform token and
    # redirects to /login on ANY 401, which was the actual bug: a rejected
    # Kaggle credential was logging the user out of the whole app.
    assert resp.status_code == 400
    assert save_calls == []  # never persisted since validation failed


@pytest.mark.parametrize("category,expected_status", [
    ("invalid_credentials", 400),
    ("forbidden", 400),
    ("bad_request", 502),
    ("rate_limited", 429),
    ("kaggle_unavailable", 503),
    ("unknown", 500),
])
def test_kaggle_connect_maps_error_category_to_status(monkeypatch, category, expected_status):
    """Each failure category from KaggleProvider.validate_credentials must
    map to a distinct, accurate HTTP status — never a blanket code that
    hides whether the problem was the user's key, Kaggle being down, or a
    bug in our own request. None may be 401 (see comment above)."""
    from app.api import dataset_source_routes
    from app.datasets.providers.kaggle_provider import KaggleCredentialsError

    client = _kaggle_test_client(monkeypatch, connection=None)

    def _raise(self, username, key):
        raise KaggleCredentialsError(f"category={category}", category=category)

    monkeypatch.setattr(dataset_source_routes.KaggleProvider, "validate_credentials", _raise)

    resp = client.post("/datasets/kaggle/connect", json={"username": "alice", "key": "k"})
    assert resp.status_code == expected_status
    assert resp.status_code != 401
    assert f"category={category}" in resp.json()["detail"]


def test_kaggle_disconnect_removes_connection(monkeypatch):
    from app.api import dataset_source_routes

    client = _kaggle_test_client(monkeypatch, connection=None)
    delete_calls = []

    async def _fake_delete_connection(user_id, provider):
        delete_calls.append((user_id, provider))
        return True

    monkeypatch.setattr(dataset_source_routes.integrations_store, "delete_connection", _fake_delete_connection)

    resp = client.post("/datasets/kaggle/disconnect")
    assert resp.status_code == 200
    assert delete_calls == [("u1", "kaggle")]


# ---------------------------------------------------------------------------
# Google Drive OAuth state/session handling — MOCKED (no live Google app)
# ---------------------------------------------------------------------------

def test_gdrive_sessions_state_roundtrip_and_single_use():
    fake_flow = mock.Mock()
    state = gdrive_sessions.create_pending_flow("u1", fake_flow)

    popped = gdrive_sessions.pop_pending_flow(state)
    assert popped["user_id"] == "u1"
    assert popped["flow"] is fake_flow

    # Single-use: popping again returns None.
    assert gdrive_sessions.pop_pending_flow(state) is None


def test_gdrive_sessions_rejects_wrong_user():
    session_token = gdrive_sessions.create_session("u1", credentials=mock.Mock())
    assert gdrive_sessions.get_session(session_token, "u1") is not None
    assert gdrive_sessions.get_session(session_token, "u2") is None


def test_gdrive_sessions_expired_state_evicted(monkeypatch):
    import time as time_mod
    fake_flow = mock.Mock()
    state = gdrive_sessions.create_pending_flow("u1", fake_flow)

    real_time = time_mod.time
    monkeypatch.setattr(time_mod, "time", lambda: real_time() + gdrive_sessions._STATE_TTL_SEC + 1)
    assert gdrive_sessions.pop_pending_flow(state) is None


def test_gdrive_routes_return_503_when_not_configured(monkeypatch):
    monkeypatch.delenv("GOOGLE_DRIVE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_CLIENT_SECRET", raising=False)

    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.api.dataset_source_routes import router

    app = FastAPI()
    app.include_router(router)

    async def _fake_user():
        return {"_id": "u1"}

    from app.auth.dependencies import get_current_user
    app.dependency_overrides[get_current_user] = _fake_user

    client = TestClient(app)
    resp = client.get("/datasets/google-drive/auth-url")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# app/utils/crypto.py — real, no mocking (Fernet is deterministic-safe here:
# we only check round-tripping, never compare ciphertext to a fixed value)
# ---------------------------------------------------------------------------

def test_crypto_encrypt_decrypt_roundtrip(monkeypatch):
    from cryptography.fernet import Fernet
    from app.utils import crypto

    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    plaintext = json.dumps({"username": "alice", "key": "secret-value"})
    ciphertext = crypto.encrypt_secret(plaintext)

    assert ciphertext != plaintext
    assert crypto.decrypt_secret(ciphertext) == plaintext


def test_crypto_raises_clean_error_when_unconfigured(monkeypatch):
    from app.utils import crypto

    monkeypatch.delenv("INTEGRATION_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError, match="INTEGRATION_ENCRYPTION_KEY"):
        crypto.encrypt_secret("{}")


def test_crypto_decrypt_fails_clean_after_key_rotation(monkeypatch):
    from cryptography.fernet import Fernet
    from app.utils import crypto

    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    ciphertext = crypto.encrypt_secret("{}")

    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    with pytest.raises(RuntimeError, match="may have changed"):
        crypto.decrypt_secret(ciphertext)


# ---------------------------------------------------------------------------
# app/integrations/store.py — real db_client, pointed at an isolated
# fallback file (mirrors the `clean_store` fixture's isolation strategy)
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_integrations(monkeypatch, tmp_path):
    from cryptography.fernet import Fernet
    from app.database.database import db_client

    monkeypatch.setenv("INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setattr(db_client, "use_fallback", True)
    monkeypatch.setattr(db_client, "fallback_path", str(tmp_path / "db.json"))
    monkeypatch.setattr(db_client, "_fallback_cache", None)
    yield


def test_integrations_save_and_get_connection_roundtrip(clean_integrations):
    from app.integrations import store as integrations_store

    asyncio.run(integrations_store.save_connection(
        user_id="u1", provider="kaggle", credentials={"username": "alice", "key": "secret"},
        provider_user_id="alice",
    ))

    conn = asyncio.run(integrations_store.get_connection("u1", "kaggle"))
    assert conn["credentials"] == {"username": "alice", "key": "secret"}
    assert conn["provider_user_id"] == "alice"
    assert asyncio.run(integrations_store.is_connected("u1", "kaggle")) is True
    assert asyncio.run(integrations_store.get_connection("u1", "google_drive")) is None


def test_integrations_save_connection_upserts_not_duplicates(clean_integrations):
    """Reconnecting (e.g. rotating an API key) must update the existing row,
    never create a second one for the same (user_id, provider)."""
    from app.database.database import db_client
    from app.integrations import store as integrations_store

    asyncio.run(integrations_store.save_connection(
        user_id="u1", provider="kaggle", credentials={"username": "alice", "key": "old-key"},
        provider_user_id="alice",
    ))
    asyncio.run(integrations_store.save_connection(
        user_id="u1", provider="kaggle", credentials={"username": "alice", "key": "new-key"},
        provider_user_id="alice",
    ))

    all_docs = asyncio.run(db_client.find_many(integrations_store.COLLECTION, {"user_id": "u1"}))
    assert len(all_docs) == 1

    conn = asyncio.run(integrations_store.get_connection("u1", "kaggle"))
    assert conn["credentials"]["key"] == "new-key"


def test_integrations_isolated_between_users(clean_integrations):
    from app.integrations import store as integrations_store

    asyncio.run(integrations_store.save_connection(
        user_id="u1", provider="kaggle", credentials={"username": "alice", "key": "a-key"},
    ))
    asyncio.run(integrations_store.save_connection(
        user_id="u2", provider="kaggle", credentials={"username": "bob", "key": "b-key"},
    ))

    conn1 = asyncio.run(integrations_store.get_connection("u1", "kaggle"))
    conn2 = asyncio.run(integrations_store.get_connection("u2", "kaggle"))
    assert conn1["credentials"]["username"] == "alice"
    assert conn2["credentials"]["username"] == "bob"


def test_integrations_delete_connection(clean_integrations):
    from app.integrations import store as integrations_store

    asyncio.run(integrations_store.save_connection(
        user_id="u1", provider="kaggle", credentials={"username": "alice", "key": "k"},
    ))
    assert asyncio.run(integrations_store.delete_connection("u1", "kaggle")) is True
    assert asyncio.run(integrations_store.get_connection("u1", "kaggle")) is None


# ---------------------------------------------------------------------------
# Regression: a dead/expired third-party (Kaggle/Drive) connection must
# never surface as HTTP 401 — api-service.ts's shared unwrap() treats ANY
# 401 as "the platform JWT is invalid," clears it, and bounces the user to
# /login. That collision was the actual root cause of "Connect Kaggle"
# redirecting an already-logged-in user to the login page.
# ---------------------------------------------------------------------------

def test_gdrive_expired_session_returns_400_not_401(monkeypatch):
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app.api.dataset_source_routes import router
    from app.auth.dependencies import get_current_user

    app = FastAPI()
    app.include_router(router)

    async def _fake_user():
        return {"_id": "u1"}

    app.dependency_overrides[get_current_user] = _fake_user

    client = TestClient(app)
    resp = client.get("/datasets/google-drive/files", params={"session": "does-not-exist"})
    assert resp.status_code == 400
    assert resp.status_code != 401
