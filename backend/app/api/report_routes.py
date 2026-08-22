import logging
import os
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse
from typing import Dict
from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.reports.pdf_generator import build_pdf_report
from app.api.ai_routes import run_ai_explanation_pipeline
from app.api.training_routes import ensure_graphs
from app.utils.response import ok

logger = logging.getLogger("regression_studio.report_routes")

router = APIRouter(prefix="", tags=["Reports & Visualization"])

STATIC_REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static", "reports"))
os.makedirs(STATIC_REPORTS_DIR, exist_ok=True)


async def _get_owned_model(model_id: str, user_id: str) -> dict:
    model_doc = await db_client.find_one("models", {"_id": model_id})
    if not model_doc or model_doc.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found."
        )
    return model_doc


# --- ROUTE 1: GET GRAPH LINKS ---
@router.get("/graphs/{model_id}")
async def get_graphs(model_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    model_doc = await _get_owned_model(model_id, current_user["_id"])

    graph_paths = await ensure_graphs(model_doc)
    base_url = str(request.base_url).rstrip("/")

    # Convert file paths to HTTP URLs served by FastAPI static mounting
    graph_urls = {}
    for name, filepath in graph_paths.items():
        if filepath and os.path.exists(filepath):
            # Extract relative path (static/graphs/model_id/filename.png)
            parts = filepath.replace("\\", "/").split("/static/")
            if len(parts) > 1:
                graph_urls[name] = f"{base_url}/static/{parts[1]}"
            else:
                # Fallback if splitting didn't work
                graph_urls[name] = f"{base_url}/static/graphs/{model_id}/{os.path.basename(filepath)}"

    return ok({
        "model_id": model_id,
        "graphs": graph_urls
    })

# Fallback generic GET to retrieve the current user's latest graphs
@router.get("/graphs")
async def get_latest_graphs(request: Request, current_user: dict = Depends(get_current_user)):
    models = await db_client.find_many("models", {"user_id": current_user["_id"]})
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No models found. Train a model first to view graphs."
        )
    sorted_models = sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
    latest_model_id = sorted_models[0]["_id"]
    return await get_graphs(latest_model_id, request, current_user)

# --- ROUTE 2: DOWNLOAD PDF REPORT ---
@router.get("/report/{model_id}")
async def download_report(model_id: str, current_user: dict = Depends(get_current_user)):
    # 1. Fetch model document, scoped to the current user
    model_doc = await _get_owned_model(model_id, current_user["_id"])

    # 2. Check if AI explanation exists. If not, generate it now.
    ai_report_markdown = model_doc.get("ai_explanation")
    if not ai_report_markdown:
        try:
            logger.info("AI explanation missing. Generating it before creating report...")
            ai_data = await run_ai_explanation_pipeline(model_id, current_user["_id"])
            ai_report_markdown = ai_data["full_report"]
            # Reload model doc to get updated fields
            model_doc = await db_client.find_one("models", {"_id": model_id})
        except Exception as e:
            ai_report_markdown = f"# AI REGRESSION STUDIO EXECUTIVE REPORT\n\nFailed to run AI Agent pipeline: {e}"

    # 3. Merge in the dataset's preprocessing_config, since the model doc itself
    #    doesn't carry it (fixes the PDF always showing fake 80/20 + mean/iqr/standard placeholders).
    dataset_doc = await db_client.find_one("datasets", {"_id": model_doc.get("dataset_id")})
    model_info = dict(model_doc)
    model_info["preprocessing"] = (dataset_doc or {}).get("preprocessing_config", {})

    # 3.5 Generate the PNG diagnostic charts now if training didn't (it no
    # longer does, by default — see training_routes.ensure_graphs).
    graph_paths = await ensure_graphs(model_doc)

    # 4. Define output path for the PDF
    output_pdf_path = os.path.join(STATIC_REPORTS_DIR, f"report_{model_id}.pdf")

    # 5. Generate the PDF
    try:
        build_pdf_report(
            model_id=model_id,
            model_info=model_info,
            graph_paths=graph_paths,
            ai_report_markdown=ai_report_markdown,
            output_pdf_path=output_pdf_path
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF document: {str(e)}"
        )

    if not os.path.exists(output_pdf_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PDF generation completed but file was not written."
        )

    return FileResponse(
        path=output_pdf_path,
        filename=f"regression_report_{model_id}.pdf",
        media_type="application/pdf"
    )

# Fallback generic GET download report for the current user's latest model
@router.get("/download-report")
async def download_latest_report(current_user: dict = Depends(get_current_user)):
    models = await db_client.find_many("models", {"user_id": current_user["_id"]})
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No models found. Train a model first to download a report."
        )
    sorted_models = sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
    latest_model_id = sorted_models[0]["_id"]
    return await download_report(latest_model_id, current_user)
