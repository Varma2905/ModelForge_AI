from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Any, Dict, List, Optional


def get_clustering_model(model_name: str, hyperparameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Returns an un-fitted scikit-learn (or optional-package) clustering
    estimator based on the model name. Mirrors get_regression_model()'s /
    get_classification_model()'s alias-branch pattern exactly (see
    app/ml/regression_models.py).

    Every estimator returned here implements .fit_predict(X) — including
    GaussianMixture, which isn't a sklearn ClusterMixin but does implement
    the same method — so the caller (cluster_routes.py) can call
    .fit_predict() uniformly regardless of which algorithm was chosen.

    Supported model names:
    - "K-Means" or "KMeans"
    - "K-Medoids" or "KMedoids" (optional: scikit-learn-extra)
    - "Agglomerative Clustering" or "Agglomerative"
    - "DBSCAN"
    - "OPTICS"
    - "Gaussian Mixture Model" or "GMM"
    - "Spectral Clustering"
    - "Fuzzy C-Means" (optional: fuzzy-c-means)
    """
    if hyperparameters is None:
        hyperparameters = {}

    name_clean = model_name.replace(" ", "").replace("-", "").lower()

    # 1. K-Means
    if name_clean in ["kmeans", "kmeanclustering", "kmeansclustering"]:
        n_clusters = hyperparameters.get("n_clusters", 3)
        random_state = hyperparameters.get("random_state", 42)
        n_init = hyperparameters.get("n_init", 10)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_clusters", "random_state", "n_init"]}
        return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, **clean_params)

    # 2. K-Medoids (optional dependency — scikit-learn-extra)
    elif name_clean in ["kmedoids", "kmedoid"]:
        from sklearn_extra.cluster import KMedoids
        n_clusters = hyperparameters.get("n_clusters", 3)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_clusters", "random_state"]}
        return KMedoids(n_clusters=n_clusters, random_state=random_state, **clean_params)

    # 3. Agglomerative Clustering
    elif name_clean in ["agglomerative", "agglomerativeclustering", "hierarchical", "hierarchicalclustering"]:
        n_clusters = hyperparameters.get("n_clusters", 3)
        linkage = hyperparameters.get("linkage", "ward")
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_clusters", "linkage"]}
        return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **clean_params)

    # 4. DBSCAN — density-based, no n_clusters (derived from the fit result)
    elif name_clean in ["dbscan"]:
        eps = hyperparameters.get("eps", 0.5)
        min_samples = hyperparameters.get("min_samples", 5)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["eps", "min_samples"]}
        return DBSCAN(eps=eps, min_samples=min_samples, **clean_params)

    # 5. OPTICS — density-based, no n_clusters
    elif name_clean in ["optics"]:
        min_samples = hyperparameters.get("min_samples", 5)
        clean_params = {k: v for k, v in hyperparameters.items() if k != "min_samples"}
        return OPTICS(min_samples=min_samples, **clean_params)

    # 6. Gaussian Mixture Model
    elif name_clean in ["gaussianmixture", "gmm", "gaussianmixturemodel"]:
        n_components = hyperparameters.get("n_clusters", hyperparameters.get("n_components", 3))
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_clusters", "n_components", "random_state"]}
        return GaussianMixture(n_components=n_components, random_state=random_state, **clean_params)

    # 7. Spectral Clustering
    elif name_clean in ["spectral", "spectralclustering"]:
        n_clusters = hyperparameters.get("n_clusters", 3)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_clusters", "random_state"]}
        return SpectralClustering(n_clusters=n_clusters, random_state=random_state, **clean_params)

    # 8. Fuzzy C-Means (optional dependency — fuzzy-c-means package, exposes
    # an sklearn-compatible .fit_predict())
    elif name_clean in ["fuzzycmeans", "fcm", "fuzzycmean"]:
        from fcmeans import FCM
        n_clusters = hyperparameters.get("n_clusters", 3)
        random_state = hyperparameters.get("random_state", 42)
        clean_params = {k: v for k, v in hyperparameters.items() if k not in ["n_clusters", "random_state"]}
        return FCM(n_clusters=n_clusters, random_state=random_state, **clean_params)

    else:
        raise ValueError(f"Unsupported model type: {model_name}. Select one of the supported clustering algorithms.")


def build_clustering_pipeline(
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]],
    numeric_features: List[str],
    categorical_features: List[str],
    scaling_method: str = "standard",
) -> Pipeline:
    """
    Builds the full preprocessing + clusterer Pipeline. Mirrors
    build_classification_pipeline() exactly, with one difference: scaling
    defaults to "standard" (not "none") when the caller didn't specify —
    almost every clustering algorithm here is distance- or density-based
    (K-Means, Agglomerative, DBSCAN, OPTICS, Spectral all measure some
    notion of point-to-point distance), so leaving numeric features on
    wildly different scales silently lets the largest-magnitude feature
    dominate every distance calculation.

        numeric features   -> SimpleImputer(median) -> scaler
        categorical features -> SimpleImputer(most_frequent) ->
                               OneHotEncoder(handle_unknown="ignore", drop="first")
                                        |
                                  ColumnTransformer(sparse_threshold=0)
                                        |
                                    clusterer

    The caller (cluster_routes.train_model) fits this on the FULL selected
    feature set — clustering is unsupervised, so there is no train/test
    split to leak across.
    """
    hyperparameters = hyperparameters or {}
    effective_scaling = scaling_method if scaling_method in ("standard", "minmax") else "standard"

    numeric_steps: List[Any] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler() if effective_scaling == "standard" else MinMaxScaler()),
    ]

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ]

    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline(categorical_steps), categorical_features))

    preprocessor = ColumnTransformer(transformers, sparse_threshold=0)
    estimator = get_clustering_model(model_name, hyperparameters)

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])
