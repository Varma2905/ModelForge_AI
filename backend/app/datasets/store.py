import json
import logging
import os
import secrets
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("regression_studio.datasets.store")

# Filesystem-only dataset storage — deliberately NOT routed through
# app/database/database.py's db_client. Mirrors the on-disk convention
# app/ml/prediction.py already uses for trained model .pkl files
# (MODELS_DIR: a flat directory of `{id}.ext` files, created at import time).
# Each dataset is two files: `{dataset_id}.data.pkl` (row data only, via
# pandas' native to_pickle/read_pickle — NOT pyarrow/Parquet: pyarrow
# registers pandas extension types process-globally at import time, which
# breaks with "A type extension with name pandas.period already defined"
# under an in-process dev-server reload, since the registration runs again
# in the same still-alive interpreter. Pickle has no such global-registry
# state, round-trips pandas dtypes exactly, and needs no extra dependency.
# .parquet files are still SUPPORTED AS AN UPLOAD FORMAT — see ingest.py's
# pd.read_parquet() — just not used for this internal storage format) and
# `{dataset_id}.meta.json` (everything else) — kept separate so metadata-only
# callers (ai_routes.py, report_routes.py, the dashboard count) never have to
# pay the cost of reading a potentially large data file.
DATASETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "datasets"))
os.makedirs(DATASETS_DIR, exist_ok=True)

_INDEX_PATH = os.path.join(DATASETS_DIR, "_index.json")
# In-memory cache of the index, mirroring DatabaseAdapter._fallback_cache in
# app/database/database.py — same single-process-only caveat applies.
_index_cache: Optional[Dict[str, Dict[str, Any]]] = None

META_FIELDS = (
    "dataset_id", "user_id", "name", "columns", "data_types", "row_count",
    "col_count", "missing_values", "is_preprocessed", "original_dataset_id",
    "preprocessing_config", "source", "created_at",
)


class DatasetNotFoundError(Exception):
    pass


def _data_path(dataset_id: str) -> str:
    return os.path.join(DATASETS_DIR, f"{dataset_id}.data.pkl")


def _meta_path(dataset_id: str) -> str:
    return os.path.join(DATASETS_DIR, f"{dataset_id}.meta.json")


def _index_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": meta.get("user_id"),
        "name": meta.get("name"),
        "row_count": meta.get("row_count"),
        "col_count": meta.get("col_count"),
        "source": meta.get("source"),
        "is_preprocessed": meta.get("is_preprocessed", False),
        "created_at": meta.get("created_at"),
    }


def _rebuild_index() -> Dict[str, Dict[str, Any]]:
    """Self-heals the index by scanning *.meta.json if _index.json is ever
    missing or corrupted, so a manual delete/copy of the data dir can't
    permanently break listing."""
    index: Dict[str, Dict[str, Any]] = {}
    for fname in os.listdir(DATASETS_DIR):
        if not fname.endswith(".meta.json"):
            continue
        dataset_id = fname[: -len(".meta.json")]
        try:
            with open(os.path.join(DATASETS_DIR, fname), "r", encoding="utf-8") as f:
                meta = json.load(f)
            index[dataset_id] = _index_summary(meta)
        except Exception as e:
            logger.warning(f"Skipping unreadable dataset meta file {fname}: {e}")
    return index


def _read_index() -> Dict[str, Dict[str, Any]]:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    if os.path.exists(_INDEX_PATH):
        try:
            with open(_INDEX_PATH, "r", encoding="utf-8") as f:
                _index_cache = json.load(f)
                return _index_cache
        except Exception as e:
            logger.warning(f"Dataset index unreadable ({e}); rebuilding from meta files.")
    _index_cache = _rebuild_index()
    _write_index(_index_cache)
    return _index_cache


def _write_index(index: Dict[str, Dict[str, Any]]) -> None:
    global _index_cache
    _index_cache = index
    with open(_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))


def save(df: pd.DataFrame, meta: Dict[str, Any], dataset_id: Optional[str] = None) -> str:
    """Writes {dataset_id}.data.pkl + {dataset_id}.meta.json, updates the
    index, and returns dataset_id (generated if not given). `meta` must at
    least carry user_id/name/source; row_count/col_count/columns/data_types/
    missing_values are recomputed here from `df` so callers can't drift out
    of sync with what's actually stored."""
    if dataset_id is None:
        dataset_id = secrets.token_hex(12)

    # Pickle (not Parquet) for internal storage — see the DATASETS_DIR
    # comment above. Unlike pyarrow's Parquet writer, this round-trips
    # object columns with mixed per-cell types (e.g. int and str in the
    # same column, common from the manual spreadsheet editor) without
    # needing a coercion fallback.
    df.to_pickle(_data_path(dataset_id))

    full_meta = {
        "dataset_id": dataset_id,
        "user_id": meta.get("user_id"),
        "name": meta.get("name") or "dataset",
        "columns": df.columns.tolist(),
        "data_types": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "row_count": len(df),
        "col_count": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "is_preprocessed": bool(meta.get("is_preprocessed", False)),
        "original_dataset_id": meta.get("original_dataset_id"),
        "preprocessing_config": meta.get("preprocessing_config", {}),
        "source": meta.get("source", "local"),
        "created_at": meta.get("created_at") or pd.Timestamp.now().isoformat(),
    }

    with open(_meta_path(dataset_id), "w", encoding="utf-8") as f:
        json.dump(full_meta, f, separators=(",", ":"))

    index = _read_index()
    index[dataset_id] = _index_summary(full_meta)
    _write_index(index)

    return dataset_id


def load_dataframe(dataset_id: str) -> pd.DataFrame:
    path = _data_path(dataset_id)
    if not os.path.exists(path):
        raise DatasetNotFoundError(f"Dataset {dataset_id} not found.")
    return pd.read_pickle(path)


def data_file_size(dataset_id: str) -> Optional[int]:
    """Size in bytes of the stored (pickled) data file — a reasonable proxy
    for "file size" in the UI. Not the original uploaded file's byte size
    (that's never persisted separately), but close enough for display."""
    path = _data_path(dataset_id)
    if not os.path.exists(path):
        return None
    return os.path.getsize(path)


def load_meta(dataset_id: str) -> Optional[Dict[str, Any]]:
    path = _meta_path(dataset_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read dataset meta {dataset_id}: {e}")
        return None


def list_for_user(user_id: str) -> List[Dict[str, Any]]:
    index = _read_index()
    results = []
    for dataset_id, summary in index.items():
        if summary.get("user_id") == user_id:
            results.append({"dataset_id": dataset_id, **summary})
    results.sort(key=lambda d: d.get("created_at") or "", reverse=True)
    return results


def delete(dataset_id: str) -> bool:
    """Removes both files + the index entry. Deliberately does NOT cascade
    to models trained from this dataset — a model keeps its own cached
    metrics/chart_data/statistical_analysis (already persisted on the model
    doc), and deferred graph regeneration (training_routes.ensure_graphs)
    degrades to returning {} gracefully, exactly like it already does today
    when a model's .pkl file is missing."""
    found = False
    for path in (_data_path(dataset_id), _meta_path(dataset_id)):
        if os.path.exists(path):
            os.remove(path)
            found = True

    index = _read_index()
    if dataset_id in index:
        del index[dataset_id]
        _write_index(index)

    return found
