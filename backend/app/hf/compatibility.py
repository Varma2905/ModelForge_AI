from dataclasses import dataclass, field
from typing import Any, Dict, List

# A model can only be labeled "inference_only" if BOTH signals line up: a
# sklearn/skops tag/library AND an actual serialized-estimator file in the
# repo. Tagging in the wild is inconsistent (confirmed against real Hub
# search results — some sklearn models only carry the tag in `tags`, not
# `library_name`), so both `library_name` and `tags` are checked.
_TABULAR_LIBRARIES = {"sklearn", "skops"}
_TABULAR_ARTIFACT_EXTENSIONS = (".pkl", ".joblib", ".skops")
_TABULAR_PIPELINE_TAGS = {"tabular-regression"}


@dataclass
class CompatibilityResult:
    compatible: bool
    execution_mode: str  # "inference_only" | "fine_tune_required" | "unsupported"
    confidence: str  # "high" | "medium" | "low"
    reasons: List[str] = field(default_factory=list)


def _is_numeric_dtype(dtype_str: str) -> bool:
    d = dtype_str.lower()
    return "int" in d or "float" in d


def assess_regression_compatibility(
    model_info: Dict[str, Any], dataset_schema: Dict[str, Any]
) -> CompatibilityResult:
    """Deterministic, non-LLM compatibility check grounded in real Hub
    metadata (library_name/tags/pipeline_tag/files) and the target dataset's
    dtypes. Never guesses — an unrecognized model is "unsupported", not
    silently assumed compatible."""
    reasons: List[str] = []
    library_name = (model_info.get("library_name") or "").lower()
    tags = [t.lower() for t in (model_info.get("tags") or [])]
    pipeline_tag = (model_info.get("pipeline_tag") or "").lower()
    files = model_info.get("files") or []

    has_tabular_lib_signal = library_name in _TABULAR_LIBRARIES or any(t in _TABULAR_LIBRARIES for t in tags)
    has_artifact = any(f.lower().endswith(_TABULAR_ARTIFACT_EXTENSIONS) for f in files)
    has_tabular_task_signal = pipeline_tag in _TABULAR_PIPELINE_TAGS or "tabular-regression" in tags

    if has_tabular_lib_signal and has_artifact:
        reasons.append(
            f"Repository is tagged as a {library_name or 'sklearn/skops'} model and contains a "
            "serialized estimator file that can be loaded directly."
        )
        numeric_cols = [
            col for col, dtype in (dataset_schema.get("data_types") or {}).items() if _is_numeric_dtype(dtype)
        ]
        if not numeric_cols:
            reasons.append("Your dataset has no numeric columns — a sklearn/skops estimator expects numeric input.")
            return CompatibilityResult(False, "unsupported", "medium", reasons)
        reasons.append(f"Your dataset has {len(numeric_cols)} numeric column(s) usable as input features.")
        return CompatibilityResult(True, "inference_only", "medium", reasons)

    if has_tabular_task_signal:
        reasons.append(
            "Model is tagged for tabular regression on the Hub, but does not expose a directly loadable "
            "sklearn/skops artifact — running it would require custom fine-tuning or preprocessing code "
            "this platform does not execute automatically."
        )
        return CompatibilityResult(False, "fine_tune_required", "medium", reasons)

    if pipeline_tag:
        reasons.append(
            f"Model's Hub task is '{model_info.get('pipeline_tag')}', not tabular regression — it has no "
            "meaningful way to predict a numeric target from arbitrary spreadsheet columns."
        )
        return CompatibilityResult(False, "unsupported", "high", reasons)

    reasons.append(
        "Could not determine this model's task or framework from its Hub metadata — treating it as "
        "unsupported rather than guessing."
    )
    return CompatibilityResult(False, "unsupported", "low", reasons)
