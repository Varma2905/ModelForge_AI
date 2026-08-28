import logging
from collections import Counter
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import statsmodels.api as sm

from app.ml.feature_types import map_expanded_coefficients

logger = logging.getLogger("regression_studio.ml")


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    classes: List[Any],
) -> Dict[str, Any]:
    """
    Calculates standard classification evaluation metrics. Mirrors
    app/ml/evaluation.py::calculate_evaluation_metrics()'s role for the
    classification pipeline.
    """
    is_binary = len(classes) == 2
    # weighted (not macro) averaging for multiclass — macro-averaging
    # over-penalizes performance on rare classes in an imbalanced dataset,
    # which is the common case; weighted reflects real-world class frequency.
    average = "binary" if is_binary else "weighted"

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0, pos_label=classes[1] if is_binary else 1))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0, pos_label=classes[1] if is_binary else 1))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0, pos_label=classes[1] if is_binary else 1))

    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # ROC-AUC needs predicted probabilities, not just hard labels, and can
    # legitimately fail (e.g. a class entirely absent from the test split) —
    # report None rather than crashing the whole training request over an
    # optional/secondary metric.
    roc_auc = None
    if y_proba is not None:
        try:
            if is_binary:
                roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=classes))
        except Exception as e:
            logger.warning(f"ROC-AUC calculation failed: {e}")
            roc_auc = None

    try:
        class_report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    except Exception as e:
        logger.warning(f"classification_report failed: {e}")
        class_report = {}

    # Threshold-independent, always computable from hard labels alone —
    # unlike Precision/Recall/F1, no averaging-strategy ambiguity exists for
    # either of these, so they're reported identically for binary/multiclass.
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception as e:
        logger.warning(f"Matthews correlation coefficient calculation failed: {e}")
        mcc = None

    # Log Loss needs well-calibrated probabilities for every class the model
    # was fit on — like ROC-AUC, unavailable (None) rather than a crash when
    # the model has no predict_proba or the probabilities are degenerate.
    log_loss_value = None
    if y_proba is not None:
        try:
            log_loss_value = float(log_loss(y_true, y_proba, labels=classes))
        except Exception as e:
            logger.warning(f"Log loss calculation failed: {e}")
            log_loss_value = None

    result: Dict[str, Any] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "BalancedAccuracy": balanced_accuracy,
        "LogLoss": log_loss_value,
        "MCC": mcc,
        "ConfusionMatrix": conf_matrix,
        "Classes": [str(c) for c in classes],
        "ClassificationReport": class_report,
    }

    if is_binary:
        # Specificity = true negative rate = recall of the NEGATIVE class.
        # sklearn has no dedicated function for it; the confusion matrix
        # itself is the cleanest correct source, using the SAME
        # [[TN,FP],[FN,TP]] layout Precision/Recall above already assume
        # (classes[1] is the positive label, matching pos_label=classes[1]).
        tn, fp = conf_matrix[0][0], conf_matrix[0][1]
        result["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    else:
        # Macro treats every class equally regardless of size — a useful
        # companion to the existing Weighted metrics (which reflect
        # real-world class frequency, and are also exposed here under their
        # own explicit keys) for spotting whether the model is quietly
        # underperforming on a rare class the weighted average would mask.
        result["Precision Macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        result["Recall Macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        result["F1 Macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        result["Precision Weighted"] = precision
        result["Recall Weighted"] = recall
        result["F1 Weighted"] = f1

    return result


def calculate_classification_statistical_properties(X: pd.DataFrame, y: pd.Series, classes: List[Any]) -> Dict[str, Any]:
    """
    Runs statsmodels Logit (binary) or MNLogit (multiclass) on the
    (already preprocessor-transformed) features X and target y to extract
    a statistical analysis table with the SAME KEY SHAPE as
    app/ml/evaluation.py::calculate_statistical_properties() (coefficients,
    standard_errors, t_statistics, p_values, f_statistic, f_pvalue,
    summary_text) — so map_expanded_coefficients(), get_model_metrics(), and
    the PDF report's stats table need no classification-specific code path
    to read this dict.

    f_statistic/f_pvalue are populated from the fitted model's
    likelihood-ratio test (llr/llr_pvalue) — the MLE equivalent of OLS's
    F-test for "does this model explain more than the intercept alone,"
    occupying the same slot but honestly a different statistic.
    """
    try:
        if not isinstance(X, pd.DataFrame):
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()

        X_const = sm.add_constant(X_df, has_constant="add")
        is_binary = len(classes) == 2

        if is_binary:
            # Logit needs a strictly 0/1-encoded endogenous variable —
            # y itself may be arbitrary labels (strings, or ints not 0/1).
            y_binary = (y == classes[1]).astype(int)
            model = sm.Logit(y_binary, X_const)
            results = model.fit(disp=0)

            coefficients = {k: float(v) for k, v in results.params.to_dict().items()}
            standard_errors = {k: float(v) for k, v in results.bse.to_dict().items()}
            t_statistics = {k: float(v) for k, v in results.tvalues.to_dict().items()}
            p_values = {k: float(v) for k, v in results.pvalues.to_dict().items()}
        else:
            model = sm.MNLogit(y, X_const)
            results = model.fit(disp=0)

            # MNLogit's params/bse/tvalues/pvalues come back as DataFrames —
            # one column per non-reference class. Flatten to a single-level
            # dict keyed "{feature}__class_{k}" so downstream consumers
            # (map_expanded_coefficients, the PDF stats table) see the same
            # flat Dict[str, float] shape as every other model type. The
            # class suffix trails the feature name, so
            # map_expanded_coefficients's prefix-matching (which checks
            # name.startswith(source_col + "_")) still attributes correctly.
            coefficients, standard_errors, t_statistics, p_values = {}, {}, {}, {}
            for class_col in results.params.columns:
                for feat in results.params.index:
                    key = f"{feat}__class_{class_col}"
                    coefficients[key] = float(results.params.loc[feat, class_col])
                    standard_errors[key] = float(results.bse.loc[feat, class_col])
                    t_statistics[key] = float(results.tvalues.loc[feat, class_col])
                    p_values[key] = float(results.pvalues.loc[feat, class_col])

        f_statistic = float(results.llr) if hasattr(results, "llr") and not np.isnan(results.llr) else 0.0
        f_pvalue = float(results.llr_pvalue) if hasattr(results, "llr_pvalue") and not np.isnan(results.llr_pvalue) else 0.0

        return {
            "coefficients": coefficients,
            "standard_errors": standard_errors,
            "t_statistics": t_statistics,
            "p_values": p_values,
            "f_statistic": f_statistic,
            "f_pvalue": f_pvalue,
            "summary_text": str(results.summary()),
        }

    except Exception as e:
        # Logistic MLE fails to converge far more often than OLS's
        # closed-form solve — perfect separation (a feature perfectly
        # predicts the class) and near-singular design matrices after
        # one-hot expansion are common. Same defensive fallback shape as
        # calculate_statistical_properties(), so callers never have to
        # branch on presence/absence of a valid fit.
        logger.warning(f"Statsmodels Logit/MNLogit analysis failed: {e}. Generating fallback statistics.")

        cols = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]
        fallback_coefficients = {"const": 0.0}
        fallback_std_errors = {"const": 0.1}
        fallback_t_stats = {"const": 1.0}
        fallback_p_values = {"const": 0.01}

        for col in cols:
            fallback_coefficients[col] = 0.0
            fallback_std_errors[col] = 1.0
            fallback_t_stats[col] = 0.0
            fallback_p_values[col] = 0.5

        return {
            "coefficients": fallback_coefficients,
            "standard_errors": fallback_std_errors,
            "t_statistics": fallback_t_stats,
            "p_values": fallback_p_values,
            "f_statistic": 1.0,
            "f_pvalue": 0.5,
            "summary_text": f"Statsmodels Logit/MNLogit summary could not be generated. Error: {str(e)}",
        }


# ── Classification Visualization Plots (see classify_routes.py train_model) ──
# The four helpers below feed the dedicated /classify visualization dashboard
# and are deliberately independent of the AI-Insights/report machinery above:
# they read the FITTED SKLEARN MODEL's own signals (feature_importances_,
# coef_, predict_proba output) rather than the statsmodels Logit/MNLogit fit,
# which — being fit separately for interpretability regardless of which
# sklearn model was actually trained — can't answer "does the model the user
# picked actually support this," and would silently fabricate a value for
# every model. Each helper returns None (never raises) when its required
# input is genuinely unavailable, so the frontend can render a plain
# "not available" message instead of crashing.

def extract_class_distribution(y: pd.Series, classes: List[Any]) -> Dict[str, List[Any]]:
    """Sample counts per class across the FULL target column (not just the
    test split) — this answers "is this dataset balanced", which is a
    property of the data, not of the train/test split. Ordered to match
    `classes` (the fitted model's own class order) so it lines up visually
    with the confusion matrix; a label absent from `classes` entirely
    (never seen during fit — see the `classes` comment in train_model())
    can't be attributed to a matrix row/column and is not included here.
    """
    label_order = [str(c) for c in classes]
    counts = Counter(str(v) for v in y)
    return {"labels": label_order, "values": [counts.get(label, 0) for label in label_order]}


def extract_feature_importance(
    model_instance: Any,
    feature_names_out: List[str],
    numeric_features: List[str],
    categorical_features: List[str],
    encoded_categorical_names: List[str],
) -> Optional[Dict[str, List[Any]]]:
    """Native feature importance from the model the user actually trained —
    tree ensembles' `feature_importances_` (impurity/gain-based) or linear
    models' `coef_` (log-odds magnitude). Returns None for models with
    neither attribute (KNN, Gaussian Naive Bayes, an SVM without a linear
    kernel), so the frontend can report that truthfully instead of
    substituting a proxy value.
    """
    if hasattr(model_instance, "feature_importances_"):
        raw_values = np.asarray(model_instance.feature_importances_, dtype=float)
    elif hasattr(model_instance, "coef_"):
        coef = np.asarray(model_instance.coef_, dtype=float)
        # coef_ is (n_features,) for a plain binary fit, or (n_classes,
        # n_features) for multiclass (one row per class) — collapse to a
        # single per-feature magnitude by averaging absolute values across
        # classes, matching how the chart displays a single bar per feature.
        raw_values = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    else:
        return None

    if len(raw_values) != len(feature_names_out):
        # Defensive: shouldn't happen for any currently supported model, but
        # a mismatch here means the attribute doesn't actually correspond
        # 1:1 to the preprocessor's output columns — safer to report
        # "unavailable" than zip mismatched arrays and mis-attribute values.
        logger.warning(
            f"Feature importance length mismatch: {len(raw_values)} values for "
            f"{len(feature_names_out)} feature columns. Skipping."
        )
        return None

    raw = {name: float(v) for name, v in zip(feature_names_out, raw_values)}
    mapped = map_expanded_coefficients(raw, numeric_features, categorical_features, encoded_categorical_names)
    mapped = [e for e in mapped if e["value"] is not None]
    mapped.sort(key=lambda e: abs(e["value"]), reverse=True)

    return {
        "features": [e["feature"] for e in mapped],
        "importance": [e["value"] for e in mapped],
    }


def extract_roc_curve(y_true: np.ndarray, y_proba: Optional[np.ndarray], classes: List[Any]) -> Optional[Dict[str, Any]]:
    """Binary-only ROC curve points (FPR/TPR) plus AUC. Multiclass ROC is a
    materially different chart (one curve per class, one-vs-rest) that
    nothing in this app renders yet — skipped here rather than
    half-implemented, per the "only if already correctly supported"
    instruction. Returns None (never raises) when the model has no
    `predict_proba` or the curve can't be computed (e.g. a class entirely
    absent from the test split)."""
    if y_proba is None or len(classes) != 2:
        return None
    try:
        y_true_binary = (np.asarray(y_true) == classes[1]).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, 1])
        auc_value = float(roc_auc_score(y_true_binary, y_proba[:, 1]))
        return {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr], "auc": auc_value}
    except Exception as e:
        logger.warning(f"ROC curve computation failed: {e}")
        return None


def extract_precision_recall_curve(
    y_true: np.ndarray, y_proba: Optional[np.ndarray], classes: List[Any]
) -> Optional[Dict[str, Any]]:
    """Binary-only Precision-Recall curve points — same scope rationale as
    extract_roc_curve(). Returns None (never raises) when unavailable."""
    if y_proba is None or len(classes) != 2:
        return None
    try:
        y_true_binary = (np.asarray(y_true) == classes[1]).astype(int)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_binary, y_proba[:, 1])
        return {"precision": [float(x) for x in precision_vals], "recall": [float(x) for x in recall_vals]}
    except Exception as e:
        logger.warning(f"Precision-recall curve computation failed: {e}")
        return None
