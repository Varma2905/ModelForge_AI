from typing import Any, Dict, Optional

import pandas as pd

from app.ml.feature_types import classify_columns

# Powers the PDF report's "Dataset Overview" and "Feature Summary" sections
# (see pdf_generator.py) — deliberately separate from the ML pipeline
# (training/preprocessing/prediction/metrics are untouched) and computed
# fresh from the actual uploaded dataframe every time a report is built, so
# it works identically for any dataset without hard-coding column names,
# counts, or types.


def build_dataset_profile(df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
    """Returns a compact profile of `df`: row/column counts, per-kind
    column counts, missing-value/duplicate-row totals, and a per-column
    summary (original names only — never one-hot-expanded pipeline output
    names). Reuses classify_columns(), the same numerical/categorical/
    datetime classifier the rest of the app already uses for variable
    selection, so a column is always typed identically everywhere.
    """
    classification = classify_columns(df)
    total_rows = len(df)

    numerical_count = sum(1 for c in classification.values() if c["kind"] == "numerical")
    categorical_count = sum(1 for c in classification.values() if c["kind"] == "categorical")
    datetime_count = sum(1 for c in classification.values() if c["kind"] == "datetime")

    features = [
        {
            "name": col,
            "kind": info["kind"],
            "missing_pct": (info["missing_count"] / total_rows * 100) if total_rows > 0 else 0.0,
            "unique_count": info["unique_count"],
        }
        for col, info in classification.items()
    ]
    # Target column first (if present), rest in original dataset order —
    # keeps the Feature Summary table's most relevant row visible even when
    # it gets capped for a very wide dataset.
    if target and target in classification:
        features.sort(key=lambda f: 0 if f["name"] == target else 1)

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "numerical_count": numerical_count,
        "categorical_count": categorical_count,
        "datetime_count": datetime_count,
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "features": features,
    }
