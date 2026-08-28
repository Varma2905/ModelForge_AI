from typing import List, Optional

# Model display names that require EXACTLY one input feature — plain
# "Simple Linear Regression" (ŷ = β₀ + β₁x) has no meaning with more than
# one predictor. The frontend offers this as a genuinely separate card from
# "Multiple Linear Regression" (see new.model.tsx), even though both map to
# the same get_regression_model() alias group (a plain sklearn
# LinearRegression) — this is the one place that distinction is actually
# enforced, on the raw display name rather than the normalized alias.
_SIMPLE_LINEAR_NAMES = {"linear regression", "simple linear regression"}

# Model display names whose entire purpose is fitting curvature over NUMERIC
# features — see regression_models.py build_pipeline(): PolynomialFeatures
# only runs inside the numeric sub-pipeline and is skipped outright when
# there are no numeric features. Selecting Polynomial Regression on an
# all-categorical feature set silently degrades to plain one-hot-encoded
# Linear Regression, which is misleading enough to reject upfront rather
# than let the user discover it by comparing coefficients afterward.
_REQUIRES_NUMERIC_FEATURE_NAMES = {"polynomial regression"}


def check_regression_model_compatibility(
    model_name: str,
    features: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
) -> Optional[str]:
    """Returns a clear, human-readable incompatibility reason if `model_name`
    can't meaningfully be trained on this feature selection, or None if it's
    compatible.

    Every OTHER model in the catalog (Ridge/Lasso/ElasticNet/tree
    ensembles/SVR/KNN/Bayesian Ridge/Huber/Quantile) works with any mix of
    numerical + categorical features once at least one is selected — the
    pipeline's ColumnTransformer always produces a fully-numeric design
    matrix via one-hot encoding, which is exactly the "numerical matrix
    after preprocessing" every one of those models actually needs. There is
    no real extra dataset constraint to enforce for them beyond having
    picked at least one feature at all, so this function deliberately
    doesn't invent one.
    """
    if not features:
        return "Select at least one input feature to train a model."

    name_lower = model_name.strip().lower()

    if name_lower in _SIMPLE_LINEAR_NAMES and len(features) > 1:
        return (
            "Linear Regression fits a single predictor (ŷ = β₀ + β₁x) and isn't "
            f"meaningful with multiple input features ({len(features)} selected). Choose "
            "'Multiple Linear Regression' for more than one feature, or select just one feature here."
        )

    if name_lower in _REQUIRES_NUMERIC_FEATURE_NAMES and not numeric_features:
        return (
            "Polynomial Regression builds its curve from numerical features and has no numerical "
            f"input to expand ({len(categorical_features)} categorical feature(s) selected, 0 numerical). "
            "Select at least one numerical feature, or choose a different model."
        )

    return None
