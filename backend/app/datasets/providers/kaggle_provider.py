import logging
import os
import re
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List

from app.datasets.ingest import SUPPORTED_EXTENSIONS
from app.datasets.providers.base import DatasetSource, DatasetSourceProvider

logger = logging.getLogger("regression_studio.datasets.kaggle")

_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?kaggle\.com/datasets/([\w.-]+)/([\w.-]+?)(?:/.*)?/?(?:\?.*)?$",
    re.IGNORECASE,
)
_REF_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


class KaggleCredentialsError(Exception):
    """Raised when the credentials-check call fails, carrying a `category`
    so the route layer can return an accurate reason instead of a blanket
    'invalid credentials' message that hides what actually happened."""

    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


class KaggleProvider(DatasetSourceProvider):
    """Wraps the official `kaggle` PyPI client. There is no anonymous mode
    and no third-party OAuth flow — Kaggle's API only accepts a personal
    username+API-key pair, so every user connects their OWN Kaggle account
    (see app/integrations/store.py) rather than the server sharing one
    global credential. is_configured() therefore always returns True: the
    *feature* is always offered, gating happens per-user at connection time."""

    provider_id = "kaggle"

    def is_configured(self) -> bool:
        return True

    def _client(self, username: str, key: str):
        # Imported lazily, INSIDE this method, never at module top-level —
        # `import kaggle` prints a credentials-missing warning (and some
        # code paths in this package call sys.exit(1)) whenever it can't
        # find KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json, which
        # this app deliberately never sets server-wide.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        # IMPORTANT: KaggleApi.config_values is declared as a *class-level*
        # mutable dict (`config_values: Dict[str, str] = {}` on the class
        # body, not set in __init__). Writing `api.config_values['x'] = y`
        # without reassigning first would mutate that shared class dict —
        # leaking one user's Kaggle credentials into every other concurrent
        # KaggleApi() instance/request. Assigning a brand-new dict here
        # shadows the class attribute with an instance attribute, so each
        # request gets its own isolated credential set with no shared
        # mutable state and no need for a global lock.
        api.config_values = {api.CONFIG_NAME_USER: username, api.CONFIG_NAME_KEY: key}
        # Deliberately NOT calling api.authenticate() — that method reloads
        # config_values from disk/env (KAGGLE_ config vars), which would
        # discard the per-user credentials just set above. dataset_list_files
        # / dataset_download_file / competitions_list all read
        # self.config_values directly via build_kaggle_client(), so no
        # separate "authenticate" step is needed.
        return api

    def validate_credentials(self, username: str, key: str) -> None:
        """Smoke-tests a dataset-domain call — the actual operation this app
        needs — authenticated via HTTP Basic Auth with (username, key),
        exactly how the official `kaggle` SDK authenticates every request
        (see KaggleHttpClient._try_fill_auth: `session.auth = (username,
        key)`, which `requests` turns into a standard `Authorization: Basic
        base64(user:key)` header — no custom/home-grown auth scheme here).

        IMPORTANT, verified empirically (not assumed): Kaggle's dataset
        search/list/download endpoints do NOT require valid credentials for
        PUBLIC datasets — they work identically for a garbage username/key
        pair as for a real one (confirmed by successfully downloading a real
        public file using made-up credentials). This is presumably the same
        anonymous access kaggle.com itself allows logged-out visitors for
        public content. So this call can only catch a malformed
        request/network failure/Kaggle outage — it CANNOT confirm the
        key is a real, working Kaggle credential, because Kaggle's dataset
        API doesn't check that for public content.

        We deliberately do NOT validate against competitions_list() (which
        DOES require full account authentication) — competitions have
        account-level gates (rules acceptance, phone verification for some
        competitions) that are entirely unrelated to dataset access, so
        using it here previously rejected real, working, dataset-capable
        keys for reasons that had nothing to do with what this app needs.

        Raises KaggleCredentialsError only for genuine request-level
        failures — never claims "invalid credentials" for something Kaggle's
        dataset API doesn't actually check:
          - 400     -> our request was malformed (a bug, not a bad key)
          - 429     -> Kaggle is rate-limiting us
          - 5xx / network error -> Kaggle (or the network) is unavailable
          - 401/403 -> included defensively in case Kaggle's behavior
                       differs for some accounts/datasets, but not expected
        """
        import requests as _requests

        try:
            api = self._client(username, key)
            api.dataset_list(search="titanic", page=1)
        except _requests.exceptions.HTTPError as e:
            resp = e.response
            status = resp.status_code if resp is not None else None
            url = resp.url if resp is not None else "unknown"
            # Safe to log: Kaggle's error bodies for auth failures are short
            # plain-text/JSON messages that never echo back the submitted
            # key. Never log resp.request.headers (would include the
            # Authorization header) or the username/key arguments themselves.
            body_snippet = (resp.text or "")[:300] if resp is not None else str(e)
            logger.warning(f"Kaggle credential check: HTTP {status} from {url} — {body_snippet!r}")

            if status == 401:
                category = "invalid_credentials"
                message = (
                    "Invalid Kaggle credentials. Double-check both values from "
                    "https://www.kaggle.com/settings -> API -> Create New Token."
                )
            elif status == 403:
                category = "forbidden"
                message = (
                    "Kaggle accepted this username/API key, but this account doesn't have "
                    "permission to access datasets through the API. Check your Kaggle account "
                    "settings (e.g. verified email/phone) or try a different account."
                )
            elif status == 400:
                category = "bad_request"
                message = "Kaggle rejected the request format. Please try again in a moment."
            elif status == 429:
                category = "rate_limited"
                message = "Kaggle is rate-limiting requests right now. Please wait a moment and try again."
            elif status is not None and status >= 500:
                category = "kaggle_unavailable"
                message = "Kaggle's API is temporarily unavailable. Please try again shortly."
            else:
                category = "unknown"
                message = f"Kaggle returned an unexpected error (HTTP {status}). Please try again."
            raise KaggleCredentialsError(message, category=category)
        except _requests.exceptions.RequestException as e:
            logger.warning(f"Kaggle credential check: network error — {e}")
            raise KaggleCredentialsError(
                "Could not reach Kaggle to verify your credentials. Check your network and try again.",
                category="kaggle_unavailable",
            )
        except Exception as e:
            # Anything else (a bug in our own request construction, an SDK
            # internal error, etc.) is NOT a credentials problem — say so
            # explicitly rather than blaming the user's key.
            logger.error(f"Kaggle credential check: unexpected {type(e).__name__}: {e}")
            raise KaggleCredentialsError(
                "Unable to verify your Kaggle credentials due to an unexpected error. Please try again.",
                category="unknown",
            )

    @staticmethod
    def parse_ref(raw: str) -> str:
        """Accepts 'owner/slug' or a full kaggle.com/datasets/owner/slug URL
        (with or without a trailing path segment or query string)."""
        raw = (raw or "").strip()
        if not raw:
            raise ValueError("Please provide a Kaggle dataset URL or identifier.")

        url_match = _URL_RE.match(raw)
        if url_match:
            return f"{url_match.group(1)}/{url_match.group(2)}"

        if _REF_RE.match(raw):
            return raw

        raise ValueError(
            "Could not parse a Kaggle dataset reference from "
            f"'{raw}'. Use a URL like https://www.kaggle.com/datasets/<owner>/<dataset> "
            "or the identifier 'owner/dataset'."
        )

    def list_files(self, username: str, key: str, dataset_ref: str) -> List[Dict[str, Any]]:
        """Lists files WITHOUT downloading anything, so the caller can show
        the user a picker when a dataset has multiple files rather than
        silently choosing one. Filtered to formats this app can actually
        ingest (per spec: "show the available supported files")."""
        api = self._client(username, key)
        response = api.dataset_list_files(dataset_ref)
        # kaggle>=2.x's ApiListDatasetFilesResponse exposes `.dataset_files`
        # (list of ApiDatasetFile, fields `.name` / `.total_bytes`) — NOT the
        # older 1.x `.files` / `.totalBytes` shape. Falling back to the old
        # attribute names defensively in case a pinned older version is used.
        files = getattr(response, "dataset_files", None)
        if files is None:
            files = getattr(response, "files", response)
        return [
            {
                "name": str(f.name),
                "size": int(getattr(f, "total_bytes", getattr(f, "totalBytes", 0)) or 0),
            }
            for f in files
            if os.path.splitext(str(f.name))[1].lower() in SUPPORTED_EXTENSIONS
        ]

    def download_file(self, username: str, key: str, dataset_ref: str, file_name: str) -> DatasetSource:
        """Downloads only the selected file into a fresh temp dir. Kaggle's
        client sometimes wraps a single-file download in a .zip — unzip and
        resolve the real data file inside it if so. Cleans up the temp dir
        on any failure so a rejected/bad file never lingers on disk.

        Does NOT assume the on-disk filename matches `file_name` exactly —
        verified empirically that Kaggle writes it URL-encoded (e.g. a file
        named "Mental Health Dataset.csv" is saved as
        "Mental%20Health%20Dataset.csv.zip", not "Mental Health
        Dataset.csv.zip"), so a name built by string-concatenating `file_name`
        never matches and every download with a space (or other
        URL-escaped character) in its name silently "succeeded" per Kaggle's
        SDK while this method reported the file missing. Since `tmp_dir` is
        freshly created per call, whatever Kaggle actually wrote there is
        unambiguously the download result — discover it instead of
        predicting it."""
        tmp_dir = tempfile.mkdtemp(prefix="kaggle_")
        try:
            api = self._client(username, key)
            api.dataset_download_file(dataset_ref, file_name, path=tmp_dir, force=True)

            entries = os.listdir(tmp_dir)
            if not entries:
                raise ValueError(f"Kaggle download did not produce the expected file '{file_name}'.")

            # Unwrap a single-entry zip (Kaggle's usual wrapping for larger
            # single-file downloads) before looking for the real data file.
            if len(entries) == 1 and entries[0].lower().endswith(".zip"):
                zip_path = os.path.join(tmp_dir, entries[0])
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_dir)
                os.remove(zip_path)
                entries = os.listdir(tmp_dir)

            if not entries:
                raise ValueError(f"Kaggle download did not produce the expected file '{file_name}'.")

            # Prefer an exact (or URL-decoded) name match if present — more
            # informative than "whatever's first" when a zip contained extra
            # files (e.g. a license/readme alongside the requested dataset file).
            from urllib.parse import unquote

            by_name = {entries[i]: entries[i] for i in range(len(entries))}
            match = by_name.get(file_name) or next(
                (e for e in entries if unquote(e) == file_name), None
            )
            chosen = match or entries[0]
            downloaded = os.path.join(tmp_dir, chosen)

            ext = os.path.splitext(file_name)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"'{file_name}' is a {ext or 'unknown'} file, which isn't supported. "
                    f"Choose one of: {', '.join(SUPPORTED_EXTENSIONS)}."
                )
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return DatasetSource(
            source_type="kaggle",
            file_name=file_name,
            file_path=downloaded,
            file_size=os.path.getsize(downloaded),
            mime_type=None,
            metadata={"dataset_ref": dataset_ref},
        )
