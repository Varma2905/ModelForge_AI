import asyncio
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from app.datasets import store
from app.datasets.providers.base import DatasetSource

logger = logging.getLogger("regression_studio.datasets.ingest")

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls", ".json", ".parquet")


def _read_dataframe(file_path: str, file_name: str) -> pd.DataFrame:
    """Dispatches on file extension. Raises ValueError with a clean,
    user-facing message on an unsupported extension or a parse failure —
    callers translate this straight into an HTTPException detail, never a
    raw traceback."""
    ext = os.path.splitext(file_name)[1].lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(file_path)
        if ext == ".json":
            return pd.read_json(file_path)
        if ext == ".parquet":
            return pd.read_parquet(file_path)
    except Exception as e:
        raise ValueError(f"Failed to parse '{file_name}': {e}")

    raise ValueError(
        f"Unsupported file type '{ext or file_name}'. Supported formats: "
        f"{', '.join(SUPPORTED_EXTENSIONS)}."
    )


async def ingest_source(
    source: DatasetSource, user_id: str, name_override: Optional[str] = None
) -> Dict[str, Any]:
    """The single place a DatasetSource (from any provider) turns into a
    stored dataset. Everything downstream of this function — preprocessing,
    training, prediction, reports — only ever deals with a dataset_id and
    the DatasetStore, never with where the data originally came from."""
    try:
        df = await asyncio.to_thread(_read_dataframe, source.file_path, source.file_name)

        if df.empty or len(df.columns) == 0:
            raise ValueError(f"'{source.file_name}' has no rows/columns to analyze.")

        meta = {
            "user_id": user_id,
            "name": name_override or source.file_name,
            "source": source.source_type,
        }
        dataset_id = await asyncio.to_thread(store.save, df, meta)
    finally:
        # Never leave a transient provider download (a Kaggle zip extract, a
        # Google Drive temp file, or a staged local-upload temp file) behind
        # on disk once it's been converted into the stored dataset.
        try:
            if os.path.exists(source.file_path):
                os.remove(source.file_path)
        except OSError as e:
            logger.warning(f"Could not remove transient file {source.file_path}: {e}")

    stored_meta = store.load_meta(dataset_id)
    return {
        "dataset_id": dataset_id,
        "dataset_name": stored_meta["name"],
        "rows": stored_meta["row_count"],
        "columns": stored_meta["col_count"],
        "column_names": stored_meta["columns"],
        "data_types": stored_meta["data_types"],
        "missing_values": stored_meta["missing_values"],
        "source": stored_meta["source"],
    }
