import os
import tempfile

from fastapi import UploadFile

from app.datasets.providers.base import DatasetSource, DatasetSourceProvider


class LocalFileProvider(DatasetSourceProvider):
    """Wraps a file the user picked/dropped in the browser. Always
    available — there's no external configuration to gate on."""

    provider_id = "local"

    def is_configured(self) -> bool:
        return True

    def stage(self, upload_file: UploadFile, contents: bytes) -> DatasetSource:
        """Writes the already-read upload bytes to a temp file so this
        provider hands off the same DatasetSource shape Kaggle/Google Drive
        do, keeping ingest_source() uniform across all sources."""
        suffix = os.path.splitext(upload_file.filename or "")[1]
        fd, tmp_path = tempfile.mkstemp(prefix="local_upload_", suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(contents)

        return DatasetSource(
            source_type="local",
            file_name=upload_file.filename or "dataset",
            file_path=tmp_path,
            file_size=len(contents),
            mime_type=upload_file.content_type,
        )
