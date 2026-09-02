"""Pre-render consistency validation for regression/classification/clustering
reports — the same "check before rendering" role clustering_report.py's
validate_clustering_report() already plays for clustering, extended to the
other two task types and given a uniform, non-raising return shape so
report_routes.py can inspect ALL issues (not just the first one hit) before
deciding whether to abort PDF generation.

Severity "error" issues mean the report would be actively misleading or
would crash mid-render (e.g. metrics missing entirely) — report_routes.py
aborts with HTTP 422 rather than silently generating the PDF, per the
platform's "the backend is the single source of truth, never silently
degrade" requirement. Severity "warning" issues are non-fatal notices
(e.g. a soft count mismatch) surfaced in the PDF's Data Quality Notices
panel instead.
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger("regression_studio.reports.report_validation")

_REGRESSION_REQUIRED_METRICS = ("R2", "MSE", "RMSE", "MAE")
_CLASSIFICATION_REQUIRED_METRICS = ("Accuracy", "F1", "ConfusionMatrix")


def _issue(code: str, severity: str, message: str) -> Dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def _check_common(model_info: Dict[str, Any], expected_model_type: str) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []

    model_type = model_info.get("model_type")
    if model_type != expected_model_type:
        issues.append(_issue(
            "MODEL_TYPE_MISMATCH", "error",
            f"Expected a '{expected_model_type}' report but model_info has model_type='{model_type}'.",
        ))

    dataset_profile = model_info.get("dataset_profile")
    total_rows = model_info.get("total_rows")
    if dataset_profile and total_rows is not None:
        if dataset_profile.get("total_rows") != total_rows:
            issues.append(_issue(
                "ROW_COUNT_MISMATCH", "warning",
                f"Dataset overview reports {dataset_profile.get('total_rows')} rows, but the model "
                f"was trained on {total_rows} rows.",
            ))

    features = model_info.get("features") or []
    if not features:
        issues.append(_issue("NO_FEATURES", "error", "No features are recorded for this model."))

    feature_counts = model_info.get("feature_counts")
    if feature_counts and isinstance(feature_counts.get("raw_selected"), int):
        if feature_counts["raw_selected"] != len(features):
            issues.append(_issue(
                "FEATURE_COUNT_MISMATCH", "warning",
                f"feature_counts.raw_selected ({feature_counts['raw_selected']}) does not match "
                f"len(features) ({len(features)}).",
            ))

    train_rows = model_info.get("train_rows")
    test_rows = model_info.get("test_rows")
    val_rows = model_info.get("val_rows") or 0
    if train_rows is not None and test_rows is not None and total_rows is not None:
        split_sum = train_rows + test_rows + val_rows
        if split_sum != total_rows:
            issues.append(_issue(
                "SPLIT_ROW_COUNT_MISMATCH", "warning",
                f"Train+test+val rows ({split_sum}) do not sum to total_rows ({total_rows}).",
            ))

    return issues


def validate_regression_report(model_info: Dict[str, Any]) -> List[Dict[str, str]]:
    issues = _check_common(model_info, "regression")

    metrics = model_info.get("metrics") or {}
    for key in _REGRESSION_REQUIRED_METRICS:
        value = metrics.get(key)
        if value is None or not isinstance(value, (int, float)):
            issues.append(_issue(
                "METRIC_MISSING", "error",
                f"Required regression metric '{key}' is missing or not numeric.",
            ))

    return issues


def validate_classification_report(model_info: Dict[str, Any]) -> List[Dict[str, str]]:
    issues = _check_common(model_info, "classification")

    metrics = model_info.get("metrics") or {}
    for key in _CLASSIFICATION_REQUIRED_METRICS:
        if metrics.get(key) is None:
            issues.append(_issue(
                "METRIC_MISSING", "error",
                f"Required classification metric '{key}' is missing.",
            ))

    classes = metrics.get("Classes") or model_info.get("classes") or []
    conf_matrix = metrics.get("ConfusionMatrix") or []
    if conf_matrix and classes and len(conf_matrix) != len(classes):
        issues.append(_issue(
            "CONFUSION_MATRIX_SHAPE_MISMATCH", "error",
            f"Confusion matrix has {len(conf_matrix)} rows but there are {len(classes)} classes.",
        ))
    if not classes:
        issues.append(_issue("NO_CLASSES", "error", "No classes recorded for this classification model."))

    return issues


def validate_clustering_report(model_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Thin adapter over the existing, unmodified
    clustering_report.validate_clustering_report() — that function still
    raises ValueError (test_clustering_report.py asserts pytest.raises
    against it directly), so its signature/behavior is untouched here; this
    just converts a raise into the same ValidationIssue shape the other two
    task validators return, for a uniform dispatcher."""
    from app.reports.clustering_report import validate_clustering_report as _raise_based_validate

    try:
        _raise_based_validate(model_info)
    except ValueError as e:
        return [_issue("CLUSTERING_VALIDATION_FAILED", "error", str(e))]
    return []


def validate_report(model_info: Dict[str, Any]) -> List[Dict[str, str]]:
    """Dispatches to the right task-specific validator by
    model_info["model_type"]. Unknown/missing model_type is itself an error
    rather than silently skipping validation."""
    model_type = model_info.get("model_type")
    if model_type == "regression":
        return validate_regression_report(model_info)
    if model_type == "classification":
        return validate_classification_report(model_info)
    if model_type == "clustering":
        return validate_clustering_report(model_info)
    return [_issue("UNKNOWN_MODEL_TYPE", "error", f"Unrecognized model_type: {model_type!r}")]
