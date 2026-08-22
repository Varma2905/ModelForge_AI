import math

from fastapi import APIRouter, Depends
from typing import Any, Dict, List, Optional

from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.utils.response import ok, sanitize_floats

router = APIRouter(prefix="", tags=["Dashboard"])


def _is_finite_number(value: Any) -> bool:
    # isinstance(nan, float) is True, so this needs an explicit finiteness
    # check too — a stray NaN/inf metric must not silently corrupt the
    # avg/best-model aggregation below (or reach the JSON response, which
    # Starlette rejects outright for NaN/inf).
    return isinstance(value, (int, float)) and not (math.isnan(value) or math.isinf(value))


@router.get("/dashboard/summary")
async def dashboard_summary(current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id"]
    datasets = await db_client.find_many("datasets", {"user_id": user_id})
    models = await db_client.find_many("models", {"user_id": user_id})
    reports = await db_client.find_many("reports", {"user_id": user_id})

    total_datasets = len(datasets)
    total_analyses = len(models)
    total_reports = len(reports)

    valid_r2_models = [m for m in models if _is_finite_number(m.get("metrics", {}).get("R2"))]
    valid_rmse_models = [m for m in models if _is_finite_number(m.get("metrics", {}).get("RMSE"))]

    best_model: Optional[Dict[str, Any]] = None
    if valid_r2_models:
        best = max(valid_r2_models, key=lambda m: m["metrics"]["R2"])
        best_model = {
            "model_id": best["_id"],
            "model": best["model"],
            "dataset_name": best.get("dataset_name"),
            "r2": best["metrics"]["R2"],
        }

    avg_r2 = (
        sum(m["metrics"]["R2"] for m in valid_r2_models) / len(valid_r2_models)
        if valid_r2_models else None
    )
    avg_rmse = (
        sum(m["metrics"]["RMSE"] for m in valid_rmse_models) / len(valid_rmse_models)
        if valid_rmse_models else None
    )

    sorted_models = sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)
    recent_analyses: List[Dict[str, Any]] = [
        {
            "model_id": m["_id"],
            "dataset_name": m.get("dataset_name"),
            "model": m["model"],
            "r2": m.get("metrics", {}).get("R2"),
            "created_at": m.get("created_at"),
        }
        for m in sorted_models[:5]
    ]

    # Defensive backstop: sanitize the whole payload in case any pre-existing
    # record still carries an unsanitized NaN/inf metric (e.g. from before
    # this endpoint's callers started sanitizing at write time).
    return ok(sanitize_floats({
        "total_datasets": total_datasets,
        "total_analyses": total_analyses,
        "total_reports": total_reports,
        "best_model": best_model,
        "avg_r2": avg_r2,
        "avg_rmse": avg_rmse,
        "recent_analyses": recent_analyses,
    }))
