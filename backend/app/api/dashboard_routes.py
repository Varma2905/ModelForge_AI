import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user
from app.database.database import db_client
from app.datasets import store as dataset_store
from app.utils.response import ok, sanitize_floats

router = APIRouter(prefix="", tags=["Dashboard"])


def _is_finite_number(value: Any) -> bool:
    # isinstance(nan, float) is True, so this needs an explicit finiteness
    # check too — a stray NaN/inf metric must not silently corrupt the
    # avg/best-model aggregation below (or reach the JSON response, which
    # Starlette rejects outright for NaN/inf).
    return isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value))


def _record_timestamp(doc: Dict[str, Any]) -> Optional[datetime]:
    """Best-effort creation time for a dashboard record. Models and reports
    always carry an explicit created_at, written with pd.Timestamp.now()
    (naive, server-local time). Datasets don't, so this falls back to the
    timestamp embedded in their Mongo-style 24-char hex _id — a real creation
    time, not a fabricated one — but that's UTC, so it's converted to the
    system's local timezone (then stripped to naive) to match created_at's
    convention. Comparing it against local "now" without this conversion
    previously showed a dataset uploaded seconds ago as hours old whenever
    the server isn't in UTC."""
    raw = doc.get("created_at")
    if raw:
        try:
            return datetime.fromisoformat(raw).replace(tzinfo=None)
        except (ValueError, TypeError):
            pass
    try:
        _id = str(doc.get("_id", ""))
        if len(_id) == 24:
            timestamp = int(_id[:8], 16)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().replace(tzinfo=None)
    except Exception:
        pass
    return None


def _pct_change(current: int, previous: int) -> Optional[float]:
    """None when there's no prior-period baseline to compare against —
    better to omit the trend than show a misleading 0%/infinite% swing."""
    if previous == 0:
        return None
    return round((current - previous) / previous * 100, 1)


def _bucket_counts(items: List[Dict[str, Any]], now: datetime) -> Tuple[int, int]:
    """(count in the last 30 days, count in the 30 days before that)."""
    recent_cutoff = now - timedelta(days=30)
    prior_cutoff = now - timedelta(days=60)
    recent = prior = 0
    for item in items:
        ts = _record_timestamp(item)
        if ts is None:
            continue
        if ts >= recent_cutoff:
            recent += 1
        elif ts >= prior_cutoff:
            prior += 1
    return recent, prior


def _bucket_avg_metric(
    models: List[Dict[str, Any]], now: datetime, metric_key: str
) -> Tuple[Optional[float], Optional[float]]:
    """Generic version of what used to be _bucket_avg_r2 — reused for R2
    (regression) and Accuracy (classification) by passing a different
    metric_key, since both are just "average of one numeric metrics.* field,
    bucketed by creation time"."""
    recent_cutoff = now - timedelta(days=30)
    prior_cutoff = now - timedelta(days=60)
    recent_vals: List[float] = []
    prior_vals: List[float] = []
    for m in models:
        val = m.get("metrics", {}).get(metric_key)
        if not _is_finite_number(val):
            continue
        ts = _record_timestamp(m)
        if ts is None:
            continue
        if ts >= recent_cutoff:
            recent_vals.append(val)
        elif ts >= prior_cutoff:
            prior_vals.append(val)
    recent_avg = sum(recent_vals) / len(recent_vals) if recent_vals else None
    prior_avg = sum(prior_vals) / len(prior_vals) if prior_vals else None
    return recent_avg, prior_avg


@router.get("/dashboard/summary")
async def dashboard_summary(current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id"]
    datasets = dataset_store.list_for_user(user_id)
    models = await db_client.find_many("models", {"user_id": user_id})
    reports = await db_client.find_many("reports", {"user_id": user_id})

    total_datasets = len(datasets)
    total_analyses = len(models)
    total_reports = len(reports)

    # No model_type key at all -> trained before this field existed ->
    # treat as regression, so every pre-existing model doc keeps behaving
    # exactly as it did before classification/clustering support was added.
    regression_models = [m for m in models if m.get("model_type", "regression") not in ("classification", "clustering")]
    classification_models = [m for m in models if m.get("model_type") == "classification"]
    clustering_models = [m for m in models if m.get("model_type") == "clustering"]
    total_classification_analyses = len(classification_models)
    total_clustering_analyses = len(clustering_models)

    valid_r2_models = [m for m in regression_models if _is_finite_number(m.get("metrics", {}).get("R2"))]
    valid_rmse_models = [m for m in regression_models if _is_finite_number(m.get("metrics", {}).get("RMSE"))]
    valid_accuracy_models = [m for m in classification_models if _is_finite_number(m.get("metrics", {}).get("Accuracy"))]
    valid_silhouette_models = [m for m in clustering_models if _is_finite_number(m.get("metrics", {}).get("Silhouette"))]

    best_model: Optional[Dict[str, Any]] = None
    if valid_r2_models:
        best = max(valid_r2_models, key=lambda m: m["metrics"]["R2"])
        best_model = {
            "model_id": best["_id"],
            "model": best["model"],
            "dataset_name": best.get("dataset_name"),
            "r2": best["metrics"]["R2"],
        }

    best_accuracy_model: Optional[Dict[str, Any]] = None
    if valid_accuracy_models:
        best_acc = max(valid_accuracy_models, key=lambda m: m["metrics"]["Accuracy"])
        best_accuracy_model = {
            "model_id": best_acc["_id"],
            "model": best_acc["model"],
            "dataset_name": best_acc.get("dataset_name"),
            "accuracy": best_acc["metrics"]["Accuracy"],
        }

    avg_r2 = (
        sum(m["metrics"]["R2"] for m in valid_r2_models) / len(valid_r2_models)
        if valid_r2_models else None
    )
    avg_rmse = (
        sum(m["metrics"]["RMSE"] for m in valid_rmse_models) / len(valid_rmse_models)
        if valid_rmse_models else None
    )
    avg_accuracy = (
        sum(m["metrics"]["Accuracy"] for m in valid_accuracy_models) / len(valid_accuracy_models)
        if valid_accuracy_models else None
    )

    best_clustering_model: Optional[Dict[str, Any]] = None
    if valid_silhouette_models:
        best_cl = max(valid_silhouette_models, key=lambda m: m["metrics"]["Silhouette"])
        best_clustering_model = {
            "model_id": best_cl["_id"],
            "model": best_cl["model"],
            "dataset_name": best_cl.get("dataset_name"),
            "silhouette": best_cl["metrics"]["Silhouette"],
        }
    avg_silhouette = (
        sum(m["metrics"]["Silhouette"] for m in valid_silhouette_models) / len(valid_silhouette_models)
        if valid_silhouette_models else None
    )

    sorted_models = sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)
    recent_analyses: List[Dict[str, Any]] = [
        {
            "model_id": m["_id"],
            "dataset_name": m.get("dataset_name"),
            "model": m["model"],
            "model_type": m.get("model_type", "regression"),
            "r2": m.get("metrics", {}).get("R2"),
            "accuracy": m.get("metrics", {}).get("Accuracy"),
            "silhouette": m.get("metrics", {}).get("Silhouette"),
            "created_at": m.get("created_at"),
        }
        for m in sorted_models[:5]
    ]

    # Month-over-month trend for the KPI cards — real counts bucketed by each
    # record's actual creation time (see _record_timestamp), never fabricated.
    now = datetime.now()
    recent_ds, prior_ds = _bucket_counts(datasets, now)
    recent_an, prior_an = _bucket_counts(models, now)
    recent_rp, prior_rp = _bucket_counts(reports, now)
    recent_avg_r2, prior_avg_r2 = _bucket_avg_metric(regression_models, now, "R2")
    avg_r2_delta = (
        round(recent_avg_r2 - prior_avg_r2, 4)
        if recent_avg_r2 is not None and prior_avg_r2 is not None
        else None
    )
    recent_avg_acc, prior_avg_acc = _bucket_avg_metric(classification_models, now, "Accuracy")
    avg_accuracy_delta = (
        round(recent_avg_acc - prior_avg_acc, 4)
        if recent_avg_acc is not None and prior_avg_acc is not None
        else None
    )

    # A unified activity feed — dataset uploads, completed analyses, and
    # generated reports, merged and sorted by real timestamp. Reports don't
    # store dataset_name directly, so it's looked up via the model they
    # were generated from.
    model_by_id = {m["_id"]: m for m in models}
    activity: List[Dict[str, Any]] = []
    for d in datasets:
        ts = _record_timestamp(d)
        if ts:
            activity.append({
                "type": "dataset",
                "title": "Dataset",
                "subtitle": d.get("name", "Untitled dataset"),
                "action": "Uploaded",
                "timestamp": ts.isoformat(),
            })
    for m in models:
        ts = _record_timestamp(m)
        if ts:
            m_type = m.get("model_type", "regression")
            activity.append({
                "type": "analysis",
                "title": "Classification Analysis" if m_type == "classification" else "Clustering Analysis" if m_type == "clustering" else "Analysis",
                "subtitle": m.get("dataset_name", "Untitled"),
                "action": "Completed",
                "timestamp": ts.isoformat(),
            })
    for r in reports:
        ts = _record_timestamp(r)
        if ts:
            source_model = model_by_id.get(r.get("model_id"), {})
            activity.append({
                "type": "report",
                "title": "Report",
                "subtitle": source_model.get("dataset_name", "Report"),
                "action": "Generated",
                "timestamp": ts.isoformat(),
            })
    activity.sort(key=lambda a: a["timestamp"], reverse=True)

    # Top Regression Models (up to 5)
    top_regression_models = [
        {
            "model_id": m["_id"],
            "model": m["model"],
            "dataset_name": m.get("dataset_name"),
            "r2": m["metrics"].get("R2"),
            "rmse": m["metrics"].get("RMSE"),
        }
        for m in sorted(valid_r2_models, key=lambda m: m["metrics"].get("R2", 0), reverse=True)[:5]
    ]

    # Top Classification Models (up to 5)
    top_classification_models = [
        {
            "model_id": m["_id"],
            "model": m["model"],
            "dataset_name": m.get("dataset_name"),
            "accuracy": m["metrics"].get("Accuracy"),
            "f1_score": m["metrics"].get("F1"),
        }
        for m in sorted(valid_accuracy_models, key=lambda m: m["metrics"].get("Accuracy", 0), reverse=True)[:5]
    ]

    # Top Clustering Models (up to 5) — ranked by Silhouette (higher is
    # better); Davies-Bouldin is shown alongside (lower is better) rather
    # than used for ranking, matching the spec's "generally" wording (these
    # three metrics don't always agree on a single best run).
    top_clustering_models = [
        {
            "model_id": m["_id"],
            "model": m["model"],
            "dataset_name": m.get("dataset_name"),
            "silhouette": m["metrics"].get("Silhouette"),
            "davies_bouldin": m["metrics"].get("DaviesBouldin"),
            "calinski_harabasz": m["metrics"].get("CalinskiHarabasz"),
            "cluster_count": m["metrics"].get("ClusterCount"),
        }
        for m in sorted(valid_silhouette_models, key=lambda m: m["metrics"].get("Silhouette", 0), reverse=True)[:5]
    ]

    # Generate 30 days of trend data (Regression vs Classification vs Clustering cumulative counts)
    trend_data = []
    base_date = datetime.now() - timedelta(days=29)
    dates = [base_date + timedelta(days=x) for x in range(30)]

    regression_by_date = [ts for m in regression_models if (ts := _record_timestamp(m)) is not None]
    classification_by_date = [ts for m in classification_models if (ts := _record_timestamp(m)) is not None]
    clustering_by_date = [ts for m in clustering_models if (ts := _record_timestamp(m)) is not None]

    for d in dates:
        day_end = d.replace(hour=23, minute=59, second=59, microsecond=999999)
        reg_count = sum(1 for ts in regression_by_date if ts <= day_end)
        clf_count = sum(1 for ts in classification_by_date if ts <= day_end)
        clu_count = sum(1 for ts in clustering_by_date if ts <= day_end)
        trend_data.append({
            "date": d.strftime("%b %d"),
            "regression": reg_count,
            "classification": clf_count,
            "clustering": clu_count,
        })

    # Sparklines (10 data points spanning the last 30 days)
    sparkline_points = 10
    sparkline_dates = [datetime.now() - timedelta(days=29) + timedelta(days=x * 3) for x in range(sparkline_points)]
    
    datasets_by_date = [ts for d in datasets if (ts := _record_timestamp(d)) is not None]
    reports_by_date = [ts for r in reports if (ts := _record_timestamp(r)) is not None]
    
    datasets_sparkline = []
    regression_sparkline = []
    classification_sparkline = []
    clustering_sparkline = []
    reports_sparkline = []

    for d in sparkline_dates:
        day_end = d.replace(hour=23, minute=59, second=59, microsecond=999999)
        datasets_sparkline.append(sum(1 for ts in datasets_by_date if ts <= day_end))
        regression_sparkline.append(sum(1 for ts in regression_by_date if ts <= day_end))
        classification_sparkline.append(sum(1 for ts in classification_by_date if ts <= day_end))
        clustering_sparkline.append(sum(1 for ts in clustering_by_date if ts <= day_end))
        reports_sparkline.append(sum(1 for ts in reports_by_date if ts <= day_end))

    r2_sparkline = [
        m["metrics"]["R2"]
        for m in sorted(valid_r2_models, key=lambda m: _record_timestamp(m) or datetime.min)
    ][-10:]
    if len(r2_sparkline) < 10:
        r2_sparkline = [0.0] * (10 - len(r2_sparkline)) + r2_sparkline

    accuracy_sparkline = [
        m["metrics"]["Accuracy"]
        for m in sorted(valid_accuracy_models, key=lambda m: _record_timestamp(m) or datetime.min)
    ][-10:]
    if len(accuracy_sparkline) < 10:
        accuracy_sparkline = [0.0] * (10 - len(accuracy_sparkline)) + accuracy_sparkline

    silhouette_sparkline = [
        m["metrics"]["Silhouette"]
        for m in sorted(valid_silhouette_models, key=lambda m: _record_timestamp(m) or datetime.min)
    ][-10:]
    if len(silhouette_sparkline) < 10:
        silhouette_sparkline = [0.0] * (10 - len(silhouette_sparkline)) + silhouette_sparkline

    recent_avg_sil, prior_avg_sil = _bucket_avg_metric(clustering_models, now, "Silhouette")
    avg_silhouette_delta = (
        round(recent_avg_sil - prior_avg_sil, 4)
        if recent_avg_sil is not None and prior_avg_sil is not None
        else None
    )

    # Defensive backstop: sanitize the whole payload in case any pre-existing
    # record still carries an unsanitized NaN/inf metric (e.g. from before
    # this endpoint's callers started sanitizing at write time).
    return ok(sanitize_floats({
        "total_datasets": total_datasets,
        "total_analyses": total_analyses,
        "total_reports": total_reports,
        "total_classification_analyses": total_classification_analyses,
        "total_clustering_analyses": total_clustering_analyses,
        "best_model": best_model,
        "best_accuracy_model": best_accuracy_model,
        "best_clustering_model": best_clustering_model,
        "avg_r2": avg_r2,
        "avg_rmse": avg_rmse,
        "avg_accuracy": avg_accuracy,
        "avg_silhouette": avg_silhouette,
        "recent_analyses": recent_analyses,
        "top_regression_models": top_regression_models,
        "top_classification_models": top_classification_models,
        "top_clustering_models": top_clustering_models,
        "trend_data": trend_data,
        "datasets_sparkline": datasets_sparkline,
        "regression_sparkline": regression_sparkline,
        "classification_sparkline": classification_sparkline,
        "clustering_sparkline": clustering_sparkline,
        "reports_sparkline": reports_sparkline,
        "r2_sparkline": r2_sparkline,
        "accuracy_sparkline": accuracy_sparkline,
        "silhouette_sparkline": silhouette_sparkline,
        "trend_datasets_pct": _pct_change(recent_ds, prior_ds),
        "trend_analyses_pct": _pct_change(recent_an, prior_an),
        "trend_reports_pct": _pct_change(recent_rp, prior_rp),
        "avg_r2_delta": avg_r2_delta,
        "avg_accuracy_delta": avg_accuracy_delta,
        "avg_silhouette_delta": avg_silhouette_delta,
        "recent_activity": activity[:8],
    }))
