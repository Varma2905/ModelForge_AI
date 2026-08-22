import asyncio
import logging
import os
import warnings
from typing import Any, Optional, Tuple

import joblib.numpy_pickle as jb_pickle
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger("regression_studio.hf.inference")

# Real tabular sklearn/joblib artifacts sampled from the Hub ranged from
# hundreds of bytes to ~430 KB. One repo Phase 1 would happily label
# "inference_only" (sklearn-tagged, has a .pkl file) turned out to have a
# 17.5 GB "model" file per the Hub's own LFS metadata — confirmed by a
# download that never completed. This cap is ~200x the largest legitimate
# model actually found, and is checked BEFORE any download starts.
MAX_MODEL_SIZE_BYTES = 100 * 1024 * 1024

_api = HfApi(token=os.getenv("HF_TOKEN") or None)

# Restricted-unpickler allowlist. Since find_class is the only place pickle
# resolves a name into a live class/callable, restricting it here blocks
# arbitrary code execution via pickle's REDUCE/__reduce__ mechanism
# regardless of what else the stream contains. Not a formal sandbox — a
# best-effort mitigation appropriate for this app's risk profile. Module
# prefixes and builtin symbols below were derived from actually
# disassembling a real Hub model's pickle stream (numpy/sklearn/joblib's
# own array-wrapper class), not guessed.
_ALLOWED_MODULE_PREFIXES = ("numpy", "scipy", "sklearn", "joblib.numpy_pickle")
_ALLOWED_BUILTIN_SYMBOLS = {
    ("builtins", "list"),
    ("builtins", "dict"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("builtins", "complex"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("_codecs", "encode"),
}


class HFExecutionError(Exception):
    """Mirrors ConnectorError (app/connectors/base.py) and HFClientError
    (app/hf/client.py) — every function in this module raises this instead
    of letting a raw pickle/joblib/huggingface_hub exception propagate."""

    def __init__(self, category: str, message: str):
        # category: "size_limit" | "download_failed" | "unsafe_content" |
        # "load_failed" | "not_predictive" | "feature_mismatch" | "inference_failed"
        self.category = category
        super().__init__(message)


def _is_allowed(module: str, name: str) -> bool:
    if (module, name) in _ALLOWED_BUILTIN_SYMBOLS:
        return True
    return any(module == prefix or module.startswith(prefix + ".") for prefix in _ALLOWED_MODULE_PREFIXES)


class _RestrictedUnpickler(jb_pickle.NumpyUnpickler):
    """Subclasses joblib's own unpickler (not plain pickle.Unpickler) —
    real joblib .pkl files inject raw numpy array bytes directly into the
    pickle stream for efficiency (joblib.numpy_pickle.NumpyArrayWrapper), a
    hybrid format a generic unpickler can't parse. This reuses joblib's
    battle-tested array reconstruction and only adds the allowlist gate."""

    def find_class(self, module: str, name: str) -> Any:
        if not _is_allowed(module, name):
            raise HFExecutionError(
                "unsafe_content",
                f"Refusing to load object of type '{module}.{name}' — not on the safe deserialization allowlist.",
            )
        return super().find_class(module, name)


def _get_artifact_size_sync(model_id: str, filename: str) -> int:
    info = _api.model_info(model_id, files_metadata=True)
    for sibling in info.siblings or []:
        if sibling.rfilename == filename:
            return sibling.size or 0
    raise HFExecutionError("download_failed", f"File '{filename}' was not found in repository '{model_id}'.")


def _download_and_load_sync(model_id: str, filename: str) -> Tuple[Any, Optional[str]]:
    size = _get_artifact_size_sync(model_id, filename)
    if size > MAX_MODEL_SIZE_BYTES:
        raise HFExecutionError(
            "size_limit",
            f"'{filename}' is {size / (1024 * 1024):.1f} MB, which exceeds the "
            f"{MAX_MODEL_SIZE_BYTES // (1024 * 1024)} MB limit for automatic execution.",
        )

    try:
        local_path = hf_hub_download(repo_id=model_id, filename=filename)
    except Exception as e:
        logger.warning(f"HF download failed for {model_id}/{filename}: {e}")
        raise HFExecutionError("download_failed", f"Failed to download '{filename}' from '{model_id}'.")

    version_warning: Optional[str] = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with open(local_path, "rb") as f:
                obj = _RestrictedUnpickler(local_path, f, ensure_native_byte_order=True).load()
            if caught:
                version_warning = "; ".join(str(w.message) for w in caught)
    except HFExecutionError:
        raise
    except Exception as e:
        logger.warning(f"HF model load failed for {model_id}/{filename}: {e}")
        raise HFExecutionError(
            "load_failed", f"Failed to load '{filename}' — it may not be a valid sklearn/joblib model file."
        )

    if not hasattr(obj, "predict"):
        raise HFExecutionError(
            "not_predictive",
            f"The loaded object ({type(obj).__name__}) does not expose a .predict() method — "
            "not usable as a regression model.",
        )

    return obj, version_warning


async def download_and_load(model_id: str, filename: str) -> Tuple[Any, Optional[str]]:
    return await asyncio.to_thread(_download_and_load_sync, model_id, filename)


def run_inference(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    expected = getattr(estimator, "n_features_in_", None)
    if expected is not None and expected != X.shape[1]:
        raise HFExecutionError(
            "feature_mismatch",
            f"This model expects {expected} input feature(s); you selected {X.shape[1]}.",
        )
    try:
        predictions = estimator.predict(X.values)
    except HFExecutionError:
        raise
    except Exception as e:
        raise HFExecutionError("inference_failed", f"Inference failed: {e}")
    # sklearn models can return a 2D column vector (observed on a real Hub
    # model) — flatten to 1D so it lines up with y_true for metrics.
    return np.asarray(predictions).reshape(-1)
