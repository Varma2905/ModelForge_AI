import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger("regression_studio.utils.sampling")

# Chart RENDERING only — never used to limit training/evaluation data. Same
# "sample for plotting, never for fitting/scoring" convention already used by
# graph_generator.py's _MAX_ROWS_FOR_PLOTS (matplotlib EDA charts) and
# clustering_evaluation.py's compute_dendrogram_data max_samples. Metrics are
# always computed on the full test set BEFORE this ever runs.
DEFAULT_MAX_CHART_POINTS = 2000
DEFAULT_MAX_CURVE_POINTS = 200


def sample_paired_series(
    max_points: int = DEFAULT_MAX_CHART_POINTS,
    random_state: int = 42,
    **series: Sequence[Any],
) -> Dict[str, Any]:
    """Down-samples one or more equal-length, index-aligned series (e.g.
    actual/predicted/residuals, which must stay paired per test row) to at
    most `max_points`. Returns a dict with each series under its original
    keyword name plus "sampled"/"sample_size"/"total_size" — the same shape
    already used by clustering_evaluation.py's compute_dendrogram_data, so
    the frontend can use one convention for "this chart shows a subset".

    No-ops (returns the series unchanged, sampled=False) when already at or
    under max_points. Sampling is a fixed-seed RANDOM subset (not a
    head/tail slice), so it stays representative of the full distribution.
    """
    lengths = {len(v) for v in series.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"sample_paired_series: series have mismatched lengths: "
            f"{ {k: len(v) for k, v in series.items()} }"
        )
    n = lengths.pop() if lengths else 0

    if n <= max_points:
        result: Dict[str, Any] = {k: list(v) for k, v in series.items()}
        result.update(sampled=False, sample_size=n, total_size=n)
        return result

    rng = np.random.RandomState(random_state)
    idx = np.sort(rng.choice(n, max_points, replace=False))
    result = {k: [v[i] for i in idx] for k, v in series.items()}
    result.update(sampled=True, sample_size=max_points, total_size=n)
    return result


def thin_curve(
    max_points: int = DEFAULT_MAX_CURVE_POINTS,
    **series: Sequence[float],
) -> Dict[str, Any]:
    """Evenly-spaced thinning for an already-ordered curve (ROC,
    precision-recall) — unlike sample_paired_series' random subset, this
    keeps points spread evenly along the curve's existing order, since a
    curve's continuous SHAPE (not a representative distribution of
    independent samples) is what matters visually. Always keeps the first
    and last point so the curve's endpoints aren't lost. No-ops when already
    at or under max_points."""
    lengths = {len(v) for v in series.values()}
    if len(lengths) > 1:
        raise ValueError("thin_curve: series have mismatched lengths")
    n = lengths.pop() if lengths else 0

    if n <= max_points:
        result: Dict[str, Any] = {k: list(v) for k, v in series.items()}
        result.update(sampled=False, sample_size=n, total_size=n)
        return result

    idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
    result = {k: [v[i] for i in idx] for k, v in series.items()}
    result.update(sampled=True, sample_size=len(idx), total_size=n)
    return result
