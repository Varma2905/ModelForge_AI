from typing import Any, Dict, List, Optional, TypedDict

from app.ml.regression_registry import _package_installed

# Single source of truth for "what clustering algorithms does this app
# offer, and is each one actually usable right now" — mirrors
# regression_registry.py / classification_registry.py exactly.


class ClusteringModelEntry(TypedDict):
    id: str
    display_name: str
    algorithm_type: str
    description: str
    parameters: List[Dict[str, Any]]
    requires_package: Optional[str]
    supports_noise: bool  # DBSCAN/OPTICS report a "Noise Points" metric
    supports_elbow: bool  # K-Means-family: elbow-plot data is meaningful
    supports_dendrogram: bool  # Agglomerative only


_REGISTRY: List[ClusteringModelEntry] = [
    {
        "id": "K-Means",
        "display_name": "K-Means",
        "algorithm_type": "Partitioning",
        "description": "Partitions data into k clusters by minimizing within-cluster variance around iteratively-updated centroids.",
        "parameters": [
            {"name": "n_clusters", "label": "Number of clusters (k)", "default": 3, "type": "int"},
        ],
        "requires_package": None,
        "supports_noise": False,
        "supports_elbow": True,
        "supports_dendrogram": False,
    },
    {
        "id": "K-Medoids",
        "display_name": "K-Medoids",
        "algorithm_type": "Partitioning",
        "description": "Like K-Means, but centers each cluster on an actual data point (medoid) rather than a computed mean — more robust to outliers.",
        "parameters": [
            {"name": "n_clusters", "label": "Number of clusters (k)", "default": 3, "type": "int"},
        ],
        "requires_package": "sklearn_extra",
        "supports_noise": False,
        "supports_elbow": True,
        "supports_dendrogram": False,
    },
    {
        "id": "Agglomerative Clustering",
        "display_name": "Agglomerative Clustering",
        "algorithm_type": "Hierarchical",
        "description": "Builds a hierarchy of clusters bottom-up, successively merging the closest pair until k clusters remain.",
        "parameters": [
            {"name": "n_clusters", "label": "Number of clusters", "default": 3, "type": "int"},
            {"name": "linkage", "label": "Linkage", "default": "ward", "type": "select", "options": ["ward", "complete", "average", "single"]},
        ],
        "requires_package": None,
        "supports_noise": False,
        "supports_elbow": False,
        "supports_dendrogram": True,
    },
    {
        "id": "DBSCAN",
        "display_name": "DBSCAN",
        "algorithm_type": "Density-based",
        "description": "Groups points that are densely packed together, automatically marking sparse points as noise — no need to specify the number of clusters upfront.",
        "parameters": [
            {"name": "eps", "label": "Neighborhood radius (eps)", "default": 0.5, "type": "float"},
            {"name": "min_samples", "label": "Min samples per neighborhood", "default": 5, "type": "int"},
        ],
        "requires_package": None,
        "supports_noise": True,
        "supports_elbow": False,
        "supports_dendrogram": False,
    },
    {
        "id": "OPTICS",
        "display_name": "OPTICS",
        "algorithm_type": "Density-based",
        "description": "Density-based clustering like DBSCAN, but handles clusters of varying density in the same dataset by ordering points on reachability distance.",
        "parameters": [
            {"name": "min_samples", "label": "Min samples per neighborhood", "default": 5, "type": "int"},
        ],
        "requires_package": None,
        "supports_noise": True,
        "supports_elbow": False,
        "supports_dendrogram": False,
    },
    {
        "id": "Gaussian Mixture Model",
        "display_name": "Gaussian Mixture Model",
        "algorithm_type": "Probabilistic",
        "description": "Models the data as a mixture of Gaussian distributions, giving each point a probability of belonging to each cluster rather than a hard assignment.",
        "parameters": [
            {"name": "n_clusters", "label": "Number of components", "default": 3, "type": "int"},
        ],
        "requires_package": None,
        "supports_noise": False,
        "supports_elbow": True,
        "supports_dendrogram": False,
    },
    {
        "id": "Spectral Clustering",
        "display_name": "Spectral Clustering",
        "algorithm_type": "Graph-based",
        "description": "Uses the eigenvalues of a similarity graph between points to reduce dimensionality before clustering — effective on non-convex cluster shapes K-Means can't separate.",
        "parameters": [
            {"name": "n_clusters", "label": "Number of clusters", "default": 3, "type": "int"},
        ],
        "requires_package": None,
        "supports_noise": False,
        "supports_elbow": False,
        "supports_dendrogram": False,
    },
    {
        "id": "Fuzzy C-Means",
        "display_name": "Fuzzy C-Means",
        "algorithm_type": "Fuzzy",
        "description": "Like K-Means, but each point gets a degree of membership in every cluster instead of belonging to exactly one — useful when cluster boundaries genuinely overlap.",
        "parameters": [
            {"name": "n_clusters", "label": "Number of clusters", "default": 3, "type": "int"},
        ],
        "requires_package": "fcmeans",
        "supports_noise": False,
        "supports_elbow": True,
        "supports_dendrogram": False,
    },
]

_BY_ID = {entry["id"]: entry for entry in _REGISTRY}


def list_clustering_models() -> List[Dict[str, Any]]:
    """Returns the full catalog with a live-computed status for each entry —
    'available' or 'unavailable' (an optional package isn't installed)."""
    results: List[Dict[str, Any]] = []
    for entry in _REGISTRY:
        if entry["requires_package"] and not _package_installed(entry["requires_package"]):
            status = "unavailable"
            reason = f"{entry['display_name']} is unavailable because the required package ('{entry['requires_package']}') is not installed."
        else:
            status = "available"
            reason = None

        results.append({
            "id": entry["id"],
            "display_name": entry["display_name"],
            "algorithm_type": entry["algorithm_type"],
            "description": entry["description"],
            "parameters": entry["parameters"],
            "status": status,
            "reason": reason,
        })
    return results


def get_clustering_model_entry(model_id: str) -> Optional[ClusteringModelEntry]:
    return _BY_ID.get(model_id)
