import logging
import os
import matplotlib
# Use the non-interactive Agg backend to prevent GUI rendering issues on the server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from app.ml.feature_types import map_expanded_coefficients

# Set seaborn style for rich aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#1e1e24',
    'axes.facecolor': '#2a2a35',
    'text.color': '#f8f9fa',
    'axes.labelcolor': '#f8f9fa',
    'xtick.color': '#adb5bd',
    'ytick.color': '#adb5bd',
    'axes.edgecolor': '#495057',
    'grid.color': '#495057',
    'font.family': 'sans-serif'
})

logger = logging.getLogger("regression_studio.visualization")

STATIC_GRAPHS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static", "graphs"))

def ensure_graph_dir(model_id: str) -> str:
    path = os.path.join(STATIC_GRAPHS_DIR, model_id)
    os.makedirs(path, exist_ok=True)
    return path

# Sampling for PLOT RENDERING ONLY — never used for model training, which
# always fits on the full split produced upstream (see training_routes.py /
# classify_routes.py). Keeps EDA plot generation fast/light on a
# large (100k+ row) dataset without silently changing what the model learns
# from. The report/UI clearly labels a plot when this kicks in.
_MAX_ROWS_FOR_PLOTS = 5000

# A high-cardinality categorical column (e.g. a customer/product ID) can
# have thousands of distinct values — plotting every one produces an
# unreadable chart. Cap to the N most frequent categories and fold the rest
# into "Other", both for the target's own class-distribution chart and for
# per-column category frequency charts.
_MAX_CLASSES_SHOWN = 15
_MAX_CATEGORICAL_COLS_PLOTTED = 4
_MAX_CATEGORIES_PER_COL = 10


def _top_n_value_counts(series: pd.Series, max_n: int) -> pd.Series:
    """Value counts capped to the top `max_n` categories, with everything
    else folded into a single 'Other' bucket — so a high-cardinality column
    never produces an unreadable (or page-breaking) chart."""
    counts = series.astype(str).value_counts()
    if len(counts) <= max_n:
        return counts
    top = counts.iloc[:max_n]
    other_total = counts.iloc[max_n:].sum()
    return pd.concat([top, pd.Series({f"Other ({len(counts) - max_n} categories)": other_total})])


def generate_dataset_graphs(
    df: pd.DataFrame,
    target: Optional[str],
    model_id: str,
    target_series: Optional[pd.Series] = None,
) -> Dict[str, str]:
    """
    Generates dataset-specific visualization graphs, entirely from the
    ORIGINAL dataset columns (never the one-hot-expanded pipeline matrix):
    1. Distribution histograms for all numerical columns.
    2. Box plots of numerical columns to show outliers.
    3. Correlation heatmap (numerical columns only).
    4. Target / class distribution — only if `target_series` is passed.
    5. Category frequency bar charts for the categorical columns present
       in `df` (top N categories + "Other" per column, first few columns
       only) — see _MAX_CATEGORICAL_COLS_PLOTTED/_MAX_CATEGORIES_PER_COL.

    Every plot is generated independently in its own try/except so one
    failure (e.g. a single all-NaN column) never blocks the rest, and every
    plot is skipped outright (not rendered as a broken/empty chart) when
    the dataset doesn't have the right data for it — a dataset with 0 or 1
    numerical columns simply gets no histogram/heatmap, a dataset with no
    categorical columns gets no frequency chart, etc.
    """
    graph_dir = ensure_graph_dir(model_id)
    paths = {}

    # Sample the plotting data ONLY — model training elsewhere always uses
    # the complete dataset, this cap exists purely so rendering a 100k+ row
    # dataset's histograms/boxplot doesn't become the request's bottleneck.
    plot_df = df
    sample_note = ""
    if len(df) > _MAX_ROWS_FOR_PLOTS:
        plot_df = df.sample(_MAX_ROWS_FOR_PLOTS, random_state=42)
        sample_note = f" (sampled {_MAX_ROWS_FOR_PLOTS:,} of {len(df):,} rows for rendering)"

    numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = plot_df.select_dtypes(exclude=[np.number]).columns.tolist()

    if numeric_cols:
        # 1. Histogram (Distributions)
        try:
            n_cols = len(numeric_cols)
            fig, axes = plt.subplots(
                nrows=(n_cols + 1) // 2,
                ncols=2,
                figsize=(12, 4 * ((n_cols + 1) // 2)),
                squeeze=False
            )
            axes = axes.flatten()

            for idx, col in enumerate(numeric_cols):
                sns.histplot(plot_df[col], kde=True, ax=axes[idx], color="#6366f1", edgecolor="#4f46e5")
                axes[idx].set_title(f"Distribution of {col}{sample_note}", fontsize=12, color="#f8f9fa")
                axes[idx].set_xlabel("")

            # Hide unused subplots
            for idx in range(n_cols, len(axes)):
                fig.delaxes(axes[idx])

            plt.tight_layout()
            hist_path = os.path.join(graph_dir, "distribution.png")
            plt.savefig(hist_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["distribution"] = hist_path
        except Exception as e:
            logger.warning(f"Failed to generate distribution graph: {e}")

        # 2. Box Plot (Outliers)
        try:
            plt.figure(figsize=(10, 6))
            # Scale/normalize numeric data temporarily for comparison on a single box plot
            df_num = plot_df[numeric_cols]
            df_norm = (df_num - df_num.mean()) / df_num.std()

            sns.boxplot(data=df_norm, orient="h", palette="crest")
            plt.title(f"Box Plot of Normalized Numerical Features (Outlier Check){sample_note}", fontsize=13, color="#f8f9fa")
            plt.xlabel("Standard Deviations from Mean")
            plt.tight_layout()

            box_path = os.path.join(graph_dir, "boxplot.png")
            plt.savefig(box_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["boxplot"] = box_path
        except Exception as e:
            logger.warning(f"Failed to generate box plot: {e}")

        # 3. Correlation Heatmap — skipped outright (not a fake/empty chart)
        # when there isn't more than one numerical column to correlate.
        try:
            if len(numeric_cols) > 1:
                plt.figure(figsize=(8, 6))
                corr = df[numeric_cols].corr()  # full data — a correlation coefficient is cheap, no need to sample
                sns.heatmap(
                    corr,
                    annot=True,
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    fmt=".2f",
                    linewidths=0.5,
                    annot_kws={"size": 10},
                    cbar_kws={"shrink": 0.8}
                )
                plt.title("Feature Correlation Heatmap", fontsize=14, color="#f8f9fa")
                plt.tight_layout()

                heatmap_path = os.path.join(graph_dir, "heatmap.png")
                plt.savefig(heatmap_path, dpi=150, facecolor='#1e1e24')
                plt.close()
                paths["correlation_heatmap"] = heatmap_path
        except Exception as e:
            logger.warning(f"Failed to generate heatmap: {e}")

    # 4. Target / Class Distribution — only when the caller actually passed
    # the target's values (ensure_graphs() does; a caller that only has
    # feature columns simply omits this plot rather than guessing).
    if target_series is not None:
        try:
            plt.figure(figsize=(8, 5))
            is_continuous = pd.api.types.is_numeric_dtype(target_series) and target_series.nunique(dropna=True) > _MAX_CLASSES_SHOWN
            if is_continuous:
                sns.histplot(target_series.dropna(), kde=True, color="#8b5cf6", edgecolor="#7c3aed")
                plt.xlabel(target or "target")
                plt.ylabel("Count")
            else:
                counts = _top_n_value_counts(target_series.dropna(), _MAX_CLASSES_SHOWN)
                sns.barplot(x=counts.values, y=counts.index.astype(str), hue=counts.index.astype(str), legend=False, palette="viridis")
                plt.xlabel("Count")
                plt.ylabel(target or "target")
            plt.title(f"Target Distribution — {target or 'target'}{sample_note if len(target_series) > _MAX_ROWS_FOR_PLOTS else ''}", fontsize=14, color="#f8f9fa")
            plt.tight_layout()

            target_dist_path = os.path.join(graph_dir, "target_distribution.png")
            plt.savefig(target_dist_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["target_distribution"] = target_dist_path
        except Exception as e:
            logger.warning(f"Failed to generate target distribution graph: {e}")

    # 5. Categorical Feature Frequency — top N categories + "Other" per
    # column, first few categorical columns only (an EDA overview plot, not
    # an exhaustive one-chart-per-column dump).
    if categorical_cols:
        try:
            cols_to_plot = categorical_cols[:_MAX_CATEGORICAL_COLS_PLOTTED]
            n = len(cols_to_plot)
            fig, axes = plt.subplots(nrows=(n + 1) // 2, ncols=2, figsize=(12, 4 * ((n + 1) // 2)), squeeze=False)
            axes = axes.flatten()

            for idx, col in enumerate(cols_to_plot):
                counts = _top_n_value_counts(plot_df[col].dropna(), _MAX_CATEGORIES_PER_COL)
                sns.barplot(x=counts.values, y=counts.index.astype(str), ax=axes[idx], hue=counts.index.astype(str), legend=False, palette="mako")
                axes[idx].set_title(f"Top categories — {col}", fontsize=11, color="#f8f9fa")
                axes[idx].set_xlabel("Count")

            for idx in range(n, len(axes)):
                fig.delaxes(axes[idx])

            plt.tight_layout()
            cat_freq_path = os.path.join(graph_dir, "categorical_frequency.png")
            plt.savefig(cat_freq_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["categorical_frequency"] = cat_freq_path
        except Exception as e:
            logger.warning(f"Failed to generate categorical frequency graph: {e}")

    return paths

def generate_regression_graphs(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    numeric_features: List[str],
    categorical_features: List[str],
    encoded_categorical_names: List[str],
    coefficients: Dict[str, float],
    model_id: str
) -> Dict[str, str]:
    """
    Generates regression-specific visualization graphs:
    1. Actual vs. Predicted scatter plot (with 45-degree reference line).
    2. Residual plot (residuals vs. predicted).
    3. Error Distribution histogram.
    4. Feature Importance bar chart (if coefficients/feature importances are available).
    """
    graph_dir = ensure_graph_dir(model_id)
    paths = {}
    
    y_act = np.array(y_actual).flatten()
    y_pred = np.array(y_predicted).flatten()
    residuals = y_act - y_pred
    
    # 1. Actual vs. Predicted
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y_act, color="#3b82f6", alpha=0.7, edgecolors="#1d4ed8")
        
        # Draw 45-degree ideal fit line
        lims = [
            np.min([plt.xlim()[0], plt.ylim()[0]]),
            np.max([plt.xlim()[1], plt.ylim()[1]])
        ]
        plt.plot(lims, lims, color="#ef4444", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
        
        plt.title("Actual vs. Predicted Values", fontsize=14, color="#f8f9fa")
        plt.xlabel("Predicted Values", fontsize=12)
        plt.ylabel("Actual Values", fontsize=12)
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        act_vs_pred_path = os.path.join(graph_dir, "actual_vs_predicted.png")
        plt.savefig(act_vs_pred_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["actual_vs_predicted"] = act_vs_pred_path
    except Exception as e:
        logger.warning(f"Failed to generate actual vs predicted graph: {e}")
        
    # 2. Residual Plot
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, color="#10b981", alpha=0.7, edgecolors="#047857")
        plt.axhline(y=0, color="#ef4444", linestyle="-", linewidth=2)
        
        plt.title("Residual Plot (Residuals vs. Predictions)", fontsize=14, color="#f8f9fa")
        plt.xlabel("Predicted Values", fontsize=12)
        plt.ylabel("Residuals (Actual - Predicted)", fontsize=12)
        plt.tight_layout()
        
        residual_path = os.path.join(graph_dir, "residuals.png")
        plt.savefig(residual_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["residuals"] = residual_path
    except Exception as e:
        logger.warning(f"Failed to generate residual plot: {e}")

    # 3. Error Distribution (Residual Histogram)
    try:
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True, color="#f59e0b", edgecolor="#d97706")
        plt.axvline(x=0, color="#ef4444", linestyle="--", linewidth=1.5)
        
        plt.title("Error Distribution (Residuals Histogram)", fontsize=14, color="#f8f9fa")
        plt.xlabel("Residual Value", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        
        error_dist_path = os.path.join(graph_dir, "error_distribution.png")
        plt.savefig(error_dist_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["error_distribution"] = error_dist_path
    except Exception as e:
        logger.warning(f"Failed to generate error distribution graph: {e}")

    # 4. Feature Importance / Coefficients Bar Chart
    try:
        # Maps expanded one-hot/scaled coefficient keys back to a display
        # label attributed to their source column (e.g. "city = Chennai") —
        # a plain `k in features` filter (the old approach) silently drops
        # every categorical-derived coefficient once keys stop matching
        # original feature names exactly.
        mapped = map_expanded_coefficients(
            coefficients, numeric_features, categorical_features, encoded_categorical_names,
        )

        if mapped:
            plt.figure(figsize=(8, 6))

            # Sort features by absolute coefficient size
            sorted_coefs = sorted(mapped, key=lambda x: abs(x["value"]), reverse=True)
            feat_names = [x["feature"] for x in sorted_coefs]
            feat_vals = [x["value"] for x in sorted_coefs]
            
            # Determine color based on positive/negative impact
            colors = ["#3b82f6" if val >= 0 else "#ef4444" for val in feat_vals]
            
            sns.barplot(x=feat_vals, y=feat_names, palette=colors if len(colors) == len(feat_names) else None, hue=feat_names, legend=False)
            plt.axvline(x=0, color="#adb5bd", linestyle="-", linewidth=1)
            
            plt.title("Feature Impact (Coefficient / Importance Size)", fontsize=14, color="#f8f9fa")
            plt.xlabel("Effect Size (Coefficient Value)", fontsize=12)
            plt.ylabel("Features", fontsize=12)
            plt.tight_layout()
            
            importance_path = os.path.join(graph_dir, "feature_importance.png")
            plt.savefig(importance_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["feature_importance"] = importance_path
    except Exception as e:
        logger.warning(f"Failed to generate feature importance graph: {e}")

    return paths


def generate_classification_graphs(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    y_proba: Optional[np.ndarray],
    classes: List[Any],
    numeric_features: List[str],
    categorical_features: List[str],
    encoded_categorical_names: List[str],
    coefficients: Dict[str, float],
    model_id: str,
) -> Dict[str, str]:
    """
    Generates classification-specific visualization graphs, mirroring
    generate_regression_graphs()'s structure/conventions exactly (same
    ensure_graph_dir, same plt.savefig(..., dpi=150, facecolor='#1e1e24')
    convention, same per-plot try/except so one failed plot never blocks
    the rest):
    1. Confusion matrix heatmap.
    2. ROC curve (binary, or one-vs-rest per class for multiclass) — only
       if y_proba is available.
    3. Precision-recall curve — same binary/multiclass looping as ROC.
    4. Feature Importance bar chart — reused verbatim from the regression
       generator, since it's coefficient-dict-agnostic already.
    """
    graph_dir = ensure_graph_dir(model_id)
    paths = {}

    y_act = np.array(y_actual).astype(str)
    y_pred = np.array(y_predicted).astype(str)
    class_labels = [str(c) for c in classes]
    is_binary = len(class_labels) == 2

    # 1. Confusion Matrix
    try:
        cm = sk_confusion_matrix(y_act, y_pred, labels=class_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Confusion Matrix", fontsize=14, color="#f8f9fa")
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("Actual Class", fontsize=12)
        plt.tight_layout()

        cm_path = os.path.join(graph_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["confusion_matrix"] = cm_path
    except Exception as e:
        logger.warning(f"Failed to generate confusion matrix graph: {e}")

    # 2. ROC Curve (needs predicted probabilities — some estimators/configs
    # don't expose predict_proba, in which case this is skipped entirely)
    if y_proba is not None:
        try:
            proba = np.asarray(y_proba)
            plt.figure(figsize=(8, 6))

            if is_binary:
                y_true_binary = (y_act == class_labels[1]).astype(int)
                fpr, tpr, _ = roc_curve(y_true_binary, proba[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color="#3b82f6", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            else:
                y_true_bin = label_binarize(y_act, classes=class_labels)
                palette = sns.color_palette("husl", len(class_labels))
                for i, cls in enumerate(class_labels):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=palette[i], linewidth=2, label=f"{cls} (AUC = {roc_auc:.3f})")

            plt.plot([0, 1], [0, 1], color="#adb5bd", linestyle="--", linewidth=1, label="Chance")
            plt.title("ROC Curve", fontsize=14, color="#f8f9fa")
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.legend(loc="lower right", fontsize=9)
            plt.tight_layout()

            roc_path = os.path.join(graph_dir, "roc_curve.png")
            plt.savefig(roc_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["roc_curve"] = roc_path
        except Exception as e:
            logger.warning(f"Failed to generate ROC curve graph: {e}")

        # 3. Precision-Recall Curve
        try:
            proba = np.asarray(y_proba)
            plt.figure(figsize=(8, 6))

            if is_binary:
                y_true_binary = (y_act == class_labels[1]).astype(int)
                precision, recall, _ = precision_recall_curve(y_true_binary, proba[:, 1])
                plt.plot(recall, precision, color="#10b981", linewidth=2)
            else:
                y_true_bin = label_binarize(y_act, classes=class_labels)
                palette = sns.color_palette("husl", len(class_labels))
                for i, cls in enumerate(class_labels):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], proba[:, i])
                    plt.plot(recall, precision, color=palette[i], linewidth=2, label=str(cls))
                plt.legend(loc="lower left", fontsize=9)

            plt.title("Precision-Recall Curve", fontsize=14, color="#f8f9fa")
            plt.xlabel("Recall", fontsize=12)
            plt.ylabel("Precision", fontsize=12)
            plt.tight_layout()

            pr_path = os.path.join(graph_dir, "precision_recall_curve.png")
            plt.savefig(pr_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["precision_recall_curve"] = pr_path
        except Exception as e:
            logger.warning(f"Failed to generate precision-recall curve graph: {e}")

    # 4. Feature Importance / Coefficients Bar Chart — reused verbatim from
    # generate_regression_graphs(), since map_expanded_coefficients() is
    # already coefficient-dict-agnostic.
    try:
        mapped = map_expanded_coefficients(
            coefficients, numeric_features, categorical_features, encoded_categorical_names,
        )

        if mapped:
            plt.figure(figsize=(8, 6))

            sorted_coefs = sorted(mapped, key=lambda x: abs(x["value"]), reverse=True)
            feat_names = [x["feature"] for x in sorted_coefs]
            feat_vals = [x["value"] for x in sorted_coefs]

            colors = ["#3b82f6" if val >= 0 else "#ef4444" for val in feat_vals]

            sns.barplot(x=feat_vals, y=feat_names, palette=colors if len(colors) == len(feat_names) else None, hue=feat_names, legend=False)
            plt.axvline(x=0, color="#adb5bd", linestyle="-", linewidth=1)

            plt.title("Feature Impact (Coefficient / Importance Size)", fontsize=14, color="#f8f9fa")
            plt.xlabel("Effect Size (Coefficient Value)", fontsize=12)
            plt.ylabel("Features", fontsize=12)
            plt.tight_layout()

            importance_path = os.path.join(graph_dir, "feature_importance.png")
            plt.savefig(importance_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["feature_importance"] = importance_path
    except Exception as e:
        logger.warning(f"Failed to generate feature importance graph: {e}")

    return paths


def generate_clustering_graphs(
    X_transformed: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    visualizations: Dict[str, Any],
    model_id: str,
) -> Dict[str, str]:
    """
    Generates clustering-specific visualization graphs:
    1. PCA 2D Cluster Scatter Plot
    2. Cluster Size Bar Chart
    3. Silhouette Plot (per-sample silhouette coefficients)
    4. Elbow Curve (for K-Means/Centroid-based)
    5. Dendrogram (for Agglomerative Clustering)
    """
    graph_dir = ensure_graph_dir(model_id)
    paths = {}

    labels = np.asarray(labels)
    unique_labels = sorted(list(set(labels.tolist())))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Helper: Set chart aesthetics to match dark mode/premium styling
    def _apply_dark_style():
        sns.set_theme(style="darkgrid")
        plt.rcParams.update({
            'figure.facecolor': '#1e1e24',
            'axes.facecolor': '#2a2a35',
            'text.color': '#f8f9fa',
            'axes.labelcolor': '#f8f9fa',
            'xtick.color': '#adb5bd',
            'ytick.color': '#adb5bd',
            'axes.edgecolor': '#495057',
            'grid.color': '#495057'
        })

    # 1. PCA 2D Scatter Plot
    try:
        if X_transformed.shape[1] >= 2:
            _apply_dark_style()
            plt.figure(figsize=(8, 6))
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_transformed)

            palette = sns.color_palette("husl", len(unique_labels))
            # If noise (-1) is present, map it to a distinct dark gray
            for idx, label in enumerate(unique_labels):
                mask = labels == label
                lbl_name = f"Cluster {label}" if label != -1 else "Noise"
                color = "#475569" if label == -1 else palette[idx]
                plt.scatter(
                    coords[mask, 0], coords[mask, 1],
                    label=lbl_name, color=color, alpha=0.7, edgecolors='w', linewidths=0.5
                )

            plt.title(f"PCA 2D Cluster Projection ({model_name})", fontsize=14, color="#f8f9fa")
            plt.xlabel("PCA Component 1", fontsize=12)
            plt.ylabel("PCA Component 2", fontsize=12)
            plt.legend(fontsize=9)
            plt.tight_layout()

            pca_path = os.path.join(graph_dir, "pca_cluster_plot.png")
            plt.savefig(pca_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["pca_cluster_plot"] = pca_path
    except Exception as e:
        logger.warning(f"Failed to generate PCA scatter plot: {e}")

    # 2. Cluster Size Bar Chart
    try:
        _apply_dark_style()
        plt.figure(figsize=(8, 5))
        unique, counts = np.unique(labels, return_counts=True)
        categories = [f"Cluster {int(lbl)}" if lbl != -1 else "Noise" for lbl in unique]

        palette = sns.color_palette("viridis", len(unique))
        sns.barplot(x=categories, y=counts, palette=palette, hue=categories, legend=False)

        plt.title("Cluster Size Distribution", fontsize=14, color="#f8f9fa")
        plt.xlabel("Cluster / Group", fontsize=12)
        plt.ylabel("Sample Count", fontsize=12)
        plt.tight_layout()

        size_path = os.path.join(graph_dir, "cluster_sizes.png")
        plt.savefig(size_path, dpi=150, facecolor='#1e1e24')
        plt.close()
        paths["cluster_sizes"] = size_path
    except Exception as e:
        logger.warning(f"Failed to generate cluster sizes graph: {e}")

    # 3. Silhouette Plot (only valid if n_clusters >= 2)
    try:
        silhouette_data = visualizations.get("silhouette_plot")
        if silhouette_data and silhouette_data.get("by_cluster") and n_clusters >= 2:
            _apply_dark_style()
            plt.figure(figsize=(8, 6))

            by_cluster = silhouette_data["by_cluster"]
            y_lower = 10
            palette = sns.color_palette("husl", len(by_cluster))

            for idx, (cluster_lbl, vals) in enumerate(sorted(by_cluster.items())):
                vals = np.array(vals)
                size_cluster_i = len(vals)
                y_upper = y_lower + size_cluster_i

                color = palette[idx]
                plt.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0, vals,
                    facecolor=color, edgecolor=color, alpha=0.7
                )

                # Label the silhouette plots with cluster numbers in the middle
                plt.text(-0.05, y_lower + 0.5 * size_cluster_i, cluster_lbl, color="#f8f9fa", fontsize=10)
                y_lower = y_upper + 10  # 10 for the gaps between clusters

            avg_silhouette = visualizations.get("metrics", {}).get("Silhouette")
            if avg_silhouette is not None:
                plt.axvline(x=avg_silhouette, color="#ef4444", linestyle="--", label=f"Average Silhouette: {avg_silhouette:.3f}")
                plt.legend(loc="upper right", fontsize=9)

            plt.title("Silhouette Analysis Per Cluster", fontsize=14, color="#f8f9fa")
            plt.xlabel("Silhouette Coefficient Values", fontsize=12)
            plt.ylabel("Cluster Label", fontsize=12)
            plt.yticks([])  # Clear the y-axis labels / ticks
            plt.xlim([-0.1, 1.0])
            plt.tight_layout()

            sil_path = os.path.join(graph_dir, "silhouette_plot.png")
            plt.savefig(sil_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["silhouette_plot"] = sil_path
    except Exception as e:
        logger.warning(f"Failed to generate silhouette plot: {e}")

    # 4. Elbow Curve (K-Means/GMM)
    try:
        elbow_data = visualizations.get("elbow")
        if elbow_data and elbow_data.get("k_values") and elbow_data.get("values"):
            _apply_dark_style()
            plt.figure(figsize=(8, 5))
            plt.plot(elbow_data["k_values"], elbow_data["values"], marker='o', color='#8b5cf6', linewidth=2)
            plt.title(f"Elbow Curve ({model_name})", fontsize=14, color="#f8f9fa")
            plt.xlabel("Number of Clusters (k)", fontsize=12)
            plt.ylabel(elbow_data.get("metric_label", "Value"), fontsize=12)
            plt.tight_layout()

            elbow_path = os.path.join(graph_dir, "elbow_plot.png")
            plt.savefig(elbow_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["elbow_plot"] = elbow_path
    except Exception as e:
        logger.warning(f"Failed to generate elbow curve: {e}")

    # 5. Dendrogram (Hierarchical/Agglomerative)
    try:
        if "agglomerative" in model_name.lower() or "hierarchical" in model_name.lower():
            _apply_dark_style()
            plt.figure(figsize=(8, 5))
            from scipy.cluster.hierarchy import dendrogram, linkage
            # Dendrogram computation
            max_samples = min(len(X_transformed), 150)
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_transformed), max_samples, replace=False) if len(X_transformed) > max_samples else np.arange(len(X_transformed))
            Z = linkage(X_transformed[idx], method='ward')
            dendrogram(Z, no_labels=True, color_threshold=0.7 * max(Z[:, 2]))
            plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=14, color="#f8f9fa")
            plt.xlabel("Sample Index (Truncated)", fontsize=12)
            plt.ylabel("Distance (Ward)", fontsize=12)
            plt.tight_layout()

            dendrogram_path = os.path.join(graph_dir, "dendrogram_plot.png")
            plt.savefig(dendrogram_path, dpi=150, facecolor='#1e1e24')
            plt.close()
            paths["dendrogram_plot"] = dendrogram_path
    except Exception as e:
        logger.warning(f"Failed to generate dendrogram plot: {e}")

    return paths
