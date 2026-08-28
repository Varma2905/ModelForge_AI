import os
import tempfile
from typing import Any, Dict, List

from app.datasets.ingest import SUPPORTED_EXTENSIONS
from app.datasets.providers.base import DatasetSource, DatasetSourceProvider

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Google Drive doesn't natively store CSV/Parquet/JSON as recognizable MIME
# types beyond a handful of well-known ones; this list is intersected with
# a filename-extension check so files Drive mis-labels still show up.
_SUPPORTED_MIME_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/json",
    "application/octet-stream",  # common for .parquet
}


class GoogleDriveProvider(DatasetSourceProvider):
    """OAuth authorization-code flow against Drive API v3, read-only scope.
    Access/refresh tokens never touch the frontend — see gdrive_sessions.py,
    which keeps them server-side, in-memory, for the life of the process."""

    provider_id = "google_drive"

    def is_configured(self) -> bool:
        return bool(os.getenv("GOOGLE_DRIVE_CLIENT_ID")) and bool(os.getenv("GOOGLE_DRIVE_CLIENT_SECRET"))

    def build_flow(self):
        # Lazy import, same reasoning as KaggleProvider._client: avoid
        # paying an import cost (and any package-specific startup checks)
        # for a provider that may never be used.
        from google_auth_oauthlib.flow import Flow

        client_config = {
            "web": {
                "client_id": os.getenv("GOOGLE_DRIVE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_DRIVE_CLIENT_SECRET"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [os.getenv("GOOGLE_DRIVE_REDIRECT_URI", "")],
            }
        }
        flow = Flow.from_client_config(client_config, scopes=_SCOPES)
        flow.redirect_uri = os.getenv("GOOGLE_DRIVE_REDIRECT_URI")
        return flow

    def list_files(self, credentials: Any) -> List[Dict[str, Any]]:
        from googleapiclient.discovery import build

        service = build("drive", "v3", credentials=credentials)
        query = " or ".join(f"mimeType='{m}'" for m in _SUPPORTED_MIME_TYPES)
        response = service.files().list(
            q=f"({query}) and trashed=false",
            fields="files(id,name,size,mimeType)",
            pageSize=100,
        ).execute()

        files = response.get("files", [])
        return [
            {
                "id": f["id"],
                "name": f["name"],
                "size": int(f.get("size", 0) or 0),
                "mimeType": f.get("mimeType"),
            }
            for f in files
            if os.path.splitext(f["name"])[1].lower() in SUPPORTED_EXTENSIONS
        ]

    def download_file(self, credentials: Any, file_id: str, file_name: str) -> DatasetSource:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        service = build("drive", "v3", credentials=credentials)
        request = service.files().get_media(fileId=file_id)

        suffix = os.path.splitext(file_name)[1]
        fd, tmp_path = tempfile.mkstemp(prefix="gdrive_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        return DatasetSource(
            source_type="google_drive",
            file_name=file_name,
            file_path=tmp_path,
            file_size=os.path.getsize(tmp_path),
            mime_type=None,
            metadata={"file_id": file_id},
        )
