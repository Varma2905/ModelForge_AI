import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("regression_studio.ml.feature_types")

# A column with more unique values than this AND a high uniqueness ratio is
# flagged as a likely identifier (customer_id, transaction_id, ...) — never
# auto-excluded, just surfaced as a warning (see classify_columns()).
_HIGH_CARDINALITY_MIN_UNIQUE = 50
_HIGH_CARDINALITY_MIN_RATIO = 0.5

# Fraction of non-null values that must successfully parse as a date for a
# string/object column to be classified as datetime. Real datasets in this
# app are never auto-parsed into datetime64 at ingest time (see
# app/datasets/ingest.py), so a column literally named "date" or
# "created_at" still arrives as a plain string column — this heuristic is
# what actually detects it.
_DATETIME_PARSE_THRESHOLD = 0.9


def _is_integer_like(series: pd.Series) -> bool:
    """True for a genuinely integer-valued column, including the common case
    of an int column that pandas silently upcast to float64 because it
    contains NaNs (e.g. [0.0, 1.0, NaN, 2.0]). Only meaningful for columns
    already classified "numerical" — used by classification target
    validation to distinguish an integer-encoded class-label column
    (0/1/2) from a genuine continuous regression target, which classify_
    columns() alone can't do (it has no unique-count-based reclassification
    for numeric dtypes)."""
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        return bool(non_null.mod(1).eq(0).all())
    return False


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and str(series.dtype) != "str":
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    # Sampling keeps this cheap on large datasets — classify_columns() runs
    # on every column of the full dataset, not just a preview slice.
    sample = non_null.sample(min(len(non_null), 200), random_state=0)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= _DATETIME_PARSE_THRESHOLD


def _is_likely_identifier(col_name: str, series: pd.Series, kind: str) -> bool:
    """Detects likely identifier columns based on name patterns and values.
    Roll, ID, Student_ID, Customer_ID, Random Roll, index-like columns.
    """
    name_lower = col_name.strip().lower()
    
    # 1. Regex check on name
    id_patterns = [
        r'^id$', r'^roll$', r'^index$', r'^no$', r'^serial$', r'^key$', r'^pk$', r'^fk$',
        r'.*(_id|id|_roll|roll|_index|index)$',
        r'^(id_|roll_)',
        r'student_id', r'customer_id', r'random_roll', r'student\s*id', r'customer\s*id', r'random\s*roll'
    ]
    for pattern in id_patterns:
        if re.search(pattern, name_lower):
            return True
            
    # 2. Value-based check: high uniqueness index-like check
    row_count = len(series)
    if row_count > 5:
        unique_count = int(series.nunique(dropna=True))
        if unique_count / row_count > 0.95:
            if kind == "numerical" or kind == "categorical":
                return True
                
    return False


def classify_columns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Classifies every column as "numerical", "categorical", or "datetime".

    Rules (see plan for rationale):
    - bool columns are "categorical", not "numerical" — a 2-level flag like
      is_married behaves better one-hot/passthrough than as a continuous
      number, and pd.api.types.is_numeric_dtype() alone conflates the two.
    - datetime: real dtype OR a heuristic parse-attempt (see
      _looks_like_datetime) since ingest never auto-parses date strings.
    - everything else numeric -> "numerical"; everything else -> "categorical".

    high_cardinality flags likely-identifier categorical columns
    (customer_id, transaction_id, ...) for a UI warning — it never causes
    exclusion on its own.
    """
    row_count = len(df)
    result: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        if _looks_like_datetime(series):
            kind = "datetime"
        elif pd.api.types.is_bool_dtype(series):
            kind = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "numerical"
        else:
            kind = "categorical"

        high_cardinality = (
            kind == "categorical"
            and unique_count > _HIGH_CARDINALITY_MIN_UNIQUE
            and row_count > 0
            and (unique_count / row_count) > _HIGH_CARDINALITY_MIN_RATIO
        )

        is_identifier = _is_likely_identifier(col, series, kind)

        result[col] = {
            "kind": kind,
            "missing_count": missing_count,
            "unique_count": unique_count,
            "high_cardinality": high_cardinality,
            "is_identifier": is_identifier,
            "is_integer_like": kind == "numerical" and _is_integer_like(series),
        }

    return result


def map_expanded_coefficients(
    coefficients: Dict[str, float],
    numeric_features: List[str],
    categorical_features: List[str],
    encoded_categorical_names: List[str],
) -> List[Dict[str, Any]]:
    """Maps a coefficients/importance dict keyed by expanded pipeline output
    names (e.g. "num__age", "cat__city_Chennai" — or plain "age",
    "city_Chennai" depending on caller) back to a display-friendly,
    correctly-attributed list.

    `encoded_categorical_names` must be the fitted OneHotEncoder's own
    `get_feature_names_out(categorical_features)` output — authoritative
    per-column attribution, never string-parsed/guessed, since categorical
    values can themselves contain underscores or be prefixes of other
    column names (e.g. "city" vs "city_type").

    A multiclass MNLogit fit (see app/ml/classification_evaluation.py)
    flattens its per-class coefficient tables into single-level keys shaped
    "{feature}__class_{k}" (one coefficient per feature PER non-reference
    class, since MNLogit's raw params come back as a DataFrame, one column
    per class) — this function transparently strips that trailing suffix
    before attribution and folds it back into the display label, so a
    multiclass classifier's coefficients attribute correctly without a
    second code path.

    Returns a list of {"feature": display_label, "source_feature": original
    column name, "value": coefficient} — replaces the old `k in features`
    filter (training_routes.py / graph_generator.py) that silently dropped
    every one-hot-derived coefficient once keys stopped matching original
    feature names exactly.
    """
    _CLASS_SUFFIX_RE = re.compile(r"__class_(.+)$")

    def _split_class_suffix(name: str) -> Tuple[str, Optional[str]]:
        m = _CLASS_SUFFIX_RE.search(name)
        if m:
            return name[: m.start()], m.group(1)
        return name, None

    # encoded_categorical_names[i] corresponds 1:1 to whatever
    # OneHotEncoder(...).get_feature_names_out(categorical_features)
    # produced — sklearn's own naming is "{original_col}_{category}", and it
    # guarantees each output name maps to exactly one input column, so exact
    # membership in this list (built directly from the fitted encoder) is
    # always unambiguous, unlike re-deriving it via string prefix matching.
    # Class suffixes are stripped here too (a multiclass caller's encoded
    # names list may itself carry them, one entry per category per class) —
    # several suffixed variants of the same category collapse to one entry.
    encoded_to_original: Dict[str, str] = {}
    if categorical_features and encoded_categorical_names:
        # OneHotEncoder emits its outputs in the same column order as its
        # input feature list, but the number of output columns per input
        # column varies (drop="first" means it's len(categories)-1 unless a
        # column is all-NaN/single-category). We recover the mapping by
        # asking the encoder itself is not available here (this function is
        # pure/string-based on purpose so it has no sklearn dependency) —
        # instead, since sklearn's naming convention is always
        # "{col}{sep}{category}" for a *specific* known col list, the
        # longest-matching-prefix rule against `categorical_features`
        # disambiguates columns whose names are prefixes of one another
        # (e.g. "city" vs "city_type").
        for raw_name in encoded_categorical_names:
            name, _ = _split_class_suffix(raw_name)
            candidates = [c for c in categorical_features if name == c or name.startswith(c + "_")]
            if not candidates:
                continue
            best = max(candidates, key=len)
            encoded_to_original[name] = best

    def _strip_prefix(name: str) -> str:
        for prefix in ("num__", "cat__"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    results: List[Dict[str, Any]] = []
    for key, value in coefficients.items():
        # Plain "const" for a binary/regression fit; "const__class_{k}" for
        # a flattened multiclass MNLogit fit (one intercept per non-reference
        # class) — both are the model's baseline/intercept term, never a
        # real feature, and must be excluded the same way in either shape.
        if key == "const" or key.startswith("const__"):
            continue
        bare = _strip_prefix(key)
        bare, class_suffix = _split_class_suffix(bare)
        class_label = f" (class {class_suffix})" if class_suffix is not None else ""

        # A coefficient/p-value can legitimately be None here — sanitize_floats()
        # (see training_routes.py / classify_routes.py) replaces any NaN a
        # near-singular or non-converged fit produced with None before storage,
        # most commonly Logit/MNLogit on a small/near-perfectly-separated
        # training split. float(None) raises TypeError, which previously
        # crashed the whole AI Insights pipeline for exactly these models —
        # pass None straight through so callers' existing "value is None"
        # handling (e.g. the explanation agents' significance checks) can
        # report it as "could not be computed" instead.
        numeric_value = float(value) if value is not None else None

        if bare in numeric_features:
            results.append({"feature": f"{bare}{class_label}", "source_feature": bare, "value": numeric_value})
            continue

        source = encoded_to_original.get(bare)
        if source is not None:
            category = bare[len(source) + 1:]
            results.append({
                "feature": f"{source} = {category}{class_label}",
                "source_feature": source,
                "value": numeric_value,
            })
            continue

        # Unrecognized key (shouldn't normally happen) — still surface it
        # rather than silently dropping it, which was the original bug.
        results.append({"feature": f"{bare}{class_label}", "source_feature": bare, "value": numeric_value})

    return results
