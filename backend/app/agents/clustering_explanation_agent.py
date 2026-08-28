import logging
from typing import Any, Dict, List, Optional
from app.services.huggingface_service import get_llm

logger = logging.getLogger("regression_studio.agents")

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass


class ClusteringInterpretationAgent:
    """
    Clustering counterpart to explanation_agent.py's
    StatisticalInterpretationAgent / classification_explanation_agent.py's
    ClassificationInterpretationAgent. The ML engine (clustering_evaluation.py)
    has already computed every number this agent talks about — Silhouette,
    Calinski-Harabasz, Davies-Bouldin, cluster sizes, per-cluster feature
    means/medians — this class only narrates those real, backend-computed
    results in plain language. It never invents or recalculates a metric.
    """

    def __init__(self):
        self.llm = get_llm()

    async def explain(
        self, model_name: str, metrics: Dict[str, Any], cluster_sizes: Dict[str, int],
        cluster_profiles: List[Dict[str, Any]], features: List[str], total_rows: int,
    ) -> str:
        """Interprets clustering metrics and per-cluster profiles. Returns a
        detailed Markdown analysis of cluster quality, separation, sizes,
        and distinguishing characteristics."""
        context = f"""
        Algorithm: {model_name}
        Total Samples: {total_rows}
        Features Used: {features}
        Silhouette Score: {metrics.get('Silhouette')}
        Calinski-Harabasz Index: {metrics.get('CalinskiHarabasz')}
        Davies-Bouldin Index: {metrics.get('DaviesBouldin')}
        Cluster Count: {metrics.get('ClusterCount')}
        Noise Points: {metrics.get('NoisePoints')}
        Cluster Sizes: {cluster_sizes}
        Cluster Profiles (means/medians/characteristics per cluster): {cluster_profiles}
        """

        if self.llm:
            try:
                system_msg = SystemMessage(content=(
                    "You are analyzing an unsupervised clustering report.\n"
                    "Use ONLY the supplied calculated results. Do not invent metrics or try to calculate metrics yourself.\n"
                    "Do not introduce regression or classification concepts (like target variables, LinearRegression, train/test splits, coefficients, p-values, t-statistics) unless directly relevant.\n"
                    "Explain:\n"
                    "- Cluster quality & separation (interpret Silhouette, Davies-Bouldin, Calinski-Harabasz indexes cautiously)\n"
                    "- Cluster sizes & distribution balance\n"
                    "- Cluster profiles & important feature differences\n"
                    "- Potential interpretations\n"
                    "- Limitations & Recommendations\n"
                    "Clearly distinguish observed results from possible interpretations. Format output in clean Markdown."
                ))
                human_msg = HumanMessage(content=f"Interpret these clustering results:\n{context}")
                response = await self.llm.ainvoke([system_msg, human_msg])
                return response.content
            except Exception as e:
                logger.warning(f"ClusteringInterpretationAgent LLM failed: {e}. Falling back to statistical parser.")

        return self._generate_analytical_fallback(model_name, metrics, cluster_sizes, cluster_profiles, features, total_rows)

    def _generate_analytical_fallback(
        self, model_name: str, metrics: Dict[str, Any], cluster_sizes: Dict[str, int],
        cluster_profiles: List[Dict[str, Any]], features: List[str], total_rows: int,
    ) -> str:
        silhouette = metrics.get("Silhouette")
        ch = metrics.get("CalinskiHarabasz")
        db = metrics.get("DaviesBouldin")
        n_clusters = metrics.get("ClusterCount") or 0
        noise_points = metrics.get("NoisePoints")

        # 1. Overall cluster quality — Silhouette is the most interpretable
        # of the three (bounded -1 to 1, unlike CH/DB which are unbounded
        # and only meaningful in comparison to another run).
        if silhouette is None:
            quality_desc = (
                f"**Cluster quality could not be scored** — {model_name} found fewer than 2 real clusters "
                "in this data (Silhouette/Calinski-Harabasz/Davies-Bouldin all require at least 2), so no "
                "quality metric is available. Consider a different algorithm or hyperparameters."
            )
        elif silhouette >= 0.7:
            quality_desc = f"**Strong, well-separated clusters** (Silhouette {silhouette:.3f}). Points are clearly closer to their own cluster than to any neighboring one."
        elif silhouette >= 0.5:
            quality_desc = f"**Reasonable cluster structure** (Silhouette {silhouette:.3f}). Clusters are identifiable but with some overlap at the boundaries."
        elif silhouette >= 0.25:
            quality_desc = f"**Weak cluster structure** (Silhouette {silhouette:.3f}). Substantial overlap between clusters — consider different features, scaling, or a different algorithm."
        else:
            quality_desc = f"**Little to no real cluster structure** (Silhouette {silhouette:.3f}). The data may not naturally group into distinct clusters with the current features."

        ch_text = f" The Calinski-Harabasz Index is **{ch:,.1f}** — higher values indicate denser, better-separated clusters (only meaningful compared to another run on the same data)." if isinstance(ch, (int, float)) else ""
        db_text = f" The Davies-Bouldin Index is **{db:.3f}** — lower is better, with 0 meaning perfect separation." if isinstance(db, (int, float)) else ""

        # 2. Cluster size balance
        real_sizes = {k: v for k, v in cluster_sizes.items() if k != "-1"}
        if real_sizes:
            max_size, min_size = max(real_sizes.values()), min(real_sizes.values())
            imbalance_ratio = max_size / min_size if min_size > 0 else float("inf")
            if imbalance_ratio > 5:
                size_text = f"Cluster sizes are notably imbalanced (largest is {imbalance_ratio:.1f}x the smallest) — the smaller cluster(s) may represent a genuine minority pattern, or may be an artifact of the chosen algorithm/parameters."
            else:
                size_text = "Cluster sizes are reasonably balanced across the dataset."
        else:
            size_text = "No cluster size information is available."

        # 3. Noise (DBSCAN/OPTICS only)
        noise_text = ""
        if noise_points is not None:
            noise_pct = (noise_points / total_rows * 100) if total_rows else 0
            noise_text = (
                f"\n\n#### Noise Points\n{model_name} classified **{noise_points:,}** of {total_rows:,} samples "
                f"({noise_pct:.1f}%) as noise — points too sparse to belong to any dense cluster. "
                + ("This is a substantial fraction; consider loosening the density parameters if that seems too aggressive." if noise_pct > 25 else "This is a modest fraction, consistent with normal outliers.")
            )

        # 4. Per-cluster characteristics
        cluster_lines = []
        for profile in cluster_profiles:
            label = profile.get("cluster")
            if label == "Noise":
                continue
            samples = profile.get("samples", 0)
            chars = profile.get("important_characteristics") or []
            char_text = "; ".join(chars) if chars else "no single feature stands out strongly from the dataset average"
            cluster_lines.append(f"- **Cluster {label}** ({samples:,} samples): {char_text}")
        cluster_text = "\n".join(cluster_lines) if cluster_lines else "No per-cluster characteristics are available."

        # 5. Recommendations
        recommendations = []
        if silhouette is not None and silhouette < 0.25:
            recommendations.append("Try a different number of clusters, alternate scaling, or a density-based algorithm (DBSCAN/OPTICS) if the true clusters may be non-spherical.")
        if real_sizes and max(real_sizes.values()) / max(min(real_sizes.values(), default=1), 1) > 5:
            recommendations.append("Investigate the smaller cluster(s) individually — they may represent a meaningful minority segment worth a closer look.")
        if noise_points and total_rows and (noise_points / total_rows) > 0.25:
            recommendations.append("A high noise fraction often means the density parameters (eps/min_samples) are too strict for this data's scale — consider loosening them.")
        if not recommendations:
            recommendations.append("The current configuration looks reasonable; consider comparing against a second algorithm to confirm the cluster structure is stable.")
        recommendations_text = "\n".join(f"- {r}" for r in recommendations)

        markdown_output = f"""### Clustering Interpretation: **{model_name} Results**

#### Cluster Quality
* {quality_desc}{ch_text}{db_text}
* **Clusters Found:** {n_clusters}
* {size_text}{noise_text}

#### Cluster Characteristics
{cluster_text}

#### Recommendations
{recommendations_text}
"""
        return markdown_output
