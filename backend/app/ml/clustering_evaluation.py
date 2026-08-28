import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

logger = logging.getLogger("regression_studio.ml")

# silhouette_score is O(n^2) — fine up to a few thousand rows, but a
# genuinely large dataset (the app explicitly trains clustering on the FULL
# uploaded data, never sampling for the fit itself) can make it the
# slowest part of the request. Sampling here affects ONLY this one
# metric's computation, not the cluster assignments themselves or any
# other metric — sklearn's own `sample_size` parameter does exactly this.
_MAX_ROWS_FOR_SILHOUETTE = 5000


def calculate_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculates standard unsupervised clustering evaluation metrics. Mirrors
    calculate_evaluation_metrics() / calculate_classification_metrics()'s
    role for the clustering pipeline — every metric that's mathematically
    undefined for the actual clustering result (fewer than 2 real clusters,
    every point assigned to noise, a cluster below sklearn's minimum size)
    reports None instead of crashing or fabricating a value.
    """
    labels = np.asarray(labels)
    unique_labels = set(labels.tolist())
    has_noise = -1 in unique_labels
    n_clusters = len(unique_labels) - (1 if has_noise else 0)
    noise_points = int(np.sum(labels == -1)) if has_noise else None

    result: Dict[str, Any] = {
        "ClusterCount": n_clusters,
        "NoisePoints": noise_points,
        "Silhouette": None,
        "CalinskiHarabasz": None,
        "DaviesBouldin": None,
    }

    # These three metrics are all undefined with fewer than 2 clusters (a
    # single cluster, or DBSCAN/OPTICS finding everything is noise) — and
    # for silhouette_score specifically, sklearn also requires
    # 2 <= n_clusters <= n_samples-1.
    if n_clusters < 2:
        return result

    # DBSCAN/OPTICS noise points (-1) aren't a real cluster — excluded from
    # all three quality metrics the same way sklearn's own examples do,
    # otherwise "noise" would be scored as if it were a legitimate cluster.
    mask = labels != -1 if has_noise else np.ones(len(labels), dtype=bool)
    X_clustered = X[mask]
    labels_clustered = labels[mask]

    if len(set(labels_clustered.tolist())) < 2 or len(X_clustered) < 3:
        return result

    try:
        sample_size = _MAX_ROWS_FOR_SILHOUETTE if len(X_clustered) > _MAX_ROWS_FOR_SILHOUETTE else None
        result["Silhouette"] = float(silhouette_score(X_clustered, labels_clustered, sample_size=sample_size, random_state=42))
    except Exception as e:
        logger.warning(f"Silhouette score calculation failed: {e}")

    try:
        result["CalinskiHarabasz"] = float(calinski_harabasz_score(X_clustered, labels_clustered))
    except Exception as e:
        logger.warning(f"Calinski-Harabasz score calculation failed: {e}")

    try:
        result["DaviesBouldin"] = float(davies_bouldin_score(X_clustered, labels_clustered))
    except Exception as e:
        logger.warning(f"Davies-Bouldin score calculation failed: {e}")

    return result


def compute_cluster_sizes(labels: np.ndarray) -> Dict[str, int]:
    """{cluster_label: sample_count}, string-keyed (JSON object keys must be
    strings) and sorted by cluster label for a stable display order —
    "-1" (noise) sorts first so it's visually distinct from real clusters."""
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(unique)
    return {str(int(unique[i])): int(counts[i]) for i in order}


def compute_cluster_profiles(
    df_features: pd.DataFrame, labels: np.ndarray, numeric_features: List[str], categorical_features: List[str],
) -> List[Dict[str, Any]]:
    """Per-cluster profile using the ORIGINAL (not one-hot-expanded)
    feature columns — mean/median for numeric features, most-common value
    for categorical features, plus a short list of "important
    characteristics": the numeric features where this cluster's mean
    deviates most (by z-score against the whole dataset) from the overall
    average, which is what actually distinguishes one cluster from another.
    Every number here is a real backend-computed statistic — the AI
    insights agent narrates these, it never computes them.
    """
    labels = np.asarray(labels)
    profiles: List[Dict[str, Any]] = []

    overall_mean = df_features[numeric_features].mean() if numeric_features else pd.Series(dtype=float)
    overall_std = df_features[numeric_features].std().replace(0, np.nan) if numeric_features else pd.Series(dtype=float)

    for cluster_label in sorted(set(labels.tolist())):
        mask = labels == cluster_label
        subset = df_features[mask]
        n_samples = int(mask.sum())

        feature_means = {}
        feature_medians = {}
        z_scores: Dict[str, float] = {}
        for col in numeric_features:
            col_mean = subset[col].mean()
            feature_means[col] = float(col_mean) if pd.notna(col_mean) else None
            col_median = subset[col].median()
            feature_medians[col] = float(col_median) if pd.notna(col_median) else None
            if pd.notna(col_mean) and pd.notna(overall_std.get(col)):
                z_scores[col] = float((col_mean - overall_mean.get(col, 0)) / overall_std[col])

        categorical_modes = {}
        for col in categorical_features:
            mode = subset[col].mode(dropna=True)
            categorical_modes[col] = str(mode.iloc[0]) if not mode.empty else None

        # Top 3 numeric features by |z-score| — the ones that actually
        # separate this cluster from the dataset average, worded plainly.
        def _describe_z(z: float) -> str:
            # A chained ternary here previously fell through to "below" for
            # ANY z that wasn't > 0.5 or < -1.5 — including small positive
            # values like z=+0.35, silently mislabeling an above-average
            # feature as below-average. Explicit branches, ordered by
            # magnitude on each side of zero, so the sign is always right.
            if z > 1.5:
                return "well above"
            if z > 0.5:
                return "above"
            if z > 0:
                return "slightly above"
            if z > -0.5:
                return "slightly below"
            if z > -1.5:
                return "below"
            return "well below"

        top_features = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        characteristics = [
            f"{feat} is {_describe_z(z)} average (z={z:+.2f})"
            for feat, z in top_features
            if abs(z) > 0.3
        ]

        profiles.append({
            "cluster": str(int(cluster_label)) if cluster_label != -1 else "Noise",
            "samples": n_samples,
            "feature_means": feature_means,
            "feature_medians": feature_medians,
            "categorical_modes": categorical_modes,
            "important_characteristics": characteristics,
        })

    return profiles


def compute_pca_projection(X: np.ndarray, labels: np.ndarray) -> Optional[Dict[str, Any]]:
    """2D (and 3D, when there are enough source dimensions) PCA projection
    of the already-preprocessed feature matrix, for the interactive cluster
    scatter plot. Returns None (never a fake/empty plot) when there are
    fewer than 2 usable feature dimensions to project."""
    n_features = X.shape[1]
    if n_features < 2:
        return None

    try:
        n_components_2d = 2
        pca_2d = PCA(n_components=n_components_2d, random_state=42)
        coords_2d = pca_2d.fit_transform(X)

        result: Dict[str, Any] = {
            "x": [float(v) for v in coords_2d[:, 0]],
            "y": [float(v) for v in coords_2d[:, 1]],
            "labels": [str(int(l)) if l != -1 else "Noise" for l in labels],
            "explained_variance_2d": [float(v) for v in pca_2d.explained_variance_ratio_],
        }

        if n_features >= 3:
            pca_3d = PCA(n_components=3, random_state=42)
            coords_3d = pca_3d.fit_transform(X)
            result["z"] = [float(v) for v in coords_3d[:, 2]]
            result["explained_variance_3d"] = [float(v) for v in pca_3d.explained_variance_ratio_]

        return result
    except Exception as e:
        logger.warning(f"PCA projection failed: {e}")
        return None


def compute_silhouette_plot(X: np.ndarray, labels: np.ndarray) -> Optional[Dict[str, Any]]:
    """Per-sample silhouette values grouped by cluster (sorted descending
    within each cluster, matching the conventional silhouette-plot shape) —
    None when there are fewer than 2 real clusters, same condition as the
    aggregate Silhouette Score itself."""
    labels = np.asarray(labels)
    has_noise = -1 in labels
    mask = labels != -1 if has_noise else np.ones(len(labels), dtype=bool)
    X_clustered = X[mask]
    labels_clustered = labels[mask]

    if len(set(labels_clustered.tolist())) < 2 or len(X_clustered) < 3:
        return None
    if len(X_clustered) > _MAX_ROWS_FOR_SILHOUETTE:
        # Per-sample silhouette has no sklearn-native sampling option (unlike
        # silhouette_score) — sample here explicitly, same cap, same reason.
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_clustered), _MAX_ROWS_FOR_SILHOUETTE, replace=False)
        X_clustered = X_clustered[idx]
        labels_clustered = labels_clustered[idx]

    try:
        sample_values = silhouette_samples(X_clustered, labels_clustered)
    except Exception as e:
        logger.warning(f"Per-sample silhouette calculation failed: {e}")
        return None

    by_cluster: Dict[str, List[float]] = {}
    for cluster_label in sorted(set(labels_clustered.tolist())):
        vals = sorted(sample_values[labels_clustered == cluster_label].tolist(), reverse=True)
        by_cluster[str(int(cluster_label))] = [float(v) for v in vals]

    return {"by_cluster": by_cluster}


def compute_elbow_data(
    model_name: str, hyperparameters: Dict[str, Any], X: np.ndarray, k_range: range = range(2, 11),
) -> Optional[Dict[str, Any]]:
    """Re-fits the SAME algorithm across a range of k values, reading each
    fit's `.inertia_` — only meaningful for centroid-based algorithms that
    actually expose it (K-Means, K-Medoids, GMM's own BIC-based analogue).
    Returns None outright (not a fake flat line) for any algorithm whose
    fitted estimator has neither `.inertia_` nor `.bic()`, or if fitting
    fails for every k tried (e.g. too few samples for the largest k)."""
    from app.ml.clustering_models import get_clustering_model

    k_values: List[int] = []
    values: List[float] = []
    metric_label = "Inertia"
    for k in k_range:
        if k >= len(X):
            break
        try:
            params = dict(hyperparameters or {})
            params["n_clusters"] = k
            estimator = get_clustering_model(model_name, params)
            estimator.fit(X)
            if hasattr(estimator, "inertia_"):
                k_values.append(k)
                values.append(float(estimator.inertia_))
            elif hasattr(estimator, "bic"):
                metric_label = "BIC"
                k_values.append(k)
                values.append(float(estimator.bic(X)))
            else:
                return None
        except Exception as e:
            logger.warning(f"Elbow data point failed at k={k} for {model_name}: {e}")
            continue

    if len(k_values) < 2:
        return None
    return {"k_values": k_values, "values": values, "metric_label": metric_label}


def compute_dendrogram_data(X: np.ndarray, linkage_method: str = "ward", max_samples: int = 200) -> Optional[Dict[str, Any]]:
    """Hierarchical merge structure for a dendrogram, via scipy — Agglomerative
    Clustering only (the registry gates which algorithms call this). Capped
    to `max_samples` rows: a dendrogram with thousands of leaves is both
    unreadable and expensive to compute (O(n^2) linkage), so this is
    sampled for RENDERING ONLY, clearly labeled by the caller when it
    happens — never used to change what the model actually fit on."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    n = X.shape[0]
    sampled = False
    X_plot = X
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        X_plot = X[idx]
        sampled = True

    try:
        Z = linkage(X_plot, method=linkage_method)
        dendro = dendrogram(Z, no_plot=True)
    except Exception as e:
        logger.warning(f"Dendrogram computation failed: {e}")
        return None

    return {
        "icoord": dendro["icoord"],
        "dcoord": dendro["dcoord"],
        "sampled": sampled,
        "sample_size": len(X_plot),
        "total_size": n,
    }
