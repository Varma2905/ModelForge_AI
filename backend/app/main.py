import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regression_studio.main")

from app.database.mongodb import db_client
from app.api.auth_routes import router as auth_router
from app.api.dataset_routes import router as dataset_router
from app.api.preprocessing_routes import router as preprocess_router
from app.api.training_routes import router as training_router
from app.api.prediction_routes import router as prediction_router
from app.api.ai_routes import router as ai_router
from app.api.report_routes import router as report_router
from app.api.dashboard_routes import router as dashboard_router
from app.api.data_sources import router as data_sources_router
from app.api.hf_routes import router as hf_router
from app.api.db_query_routes import router as db_query_router
from app.api.baas_routes import baas_management_router, baas_public_router
from app.utils.crypto import warn_if_unconfigured as warn_if_encryption_key_unconfigured
from app.utils.response import err

app = FastAPI(
    title="AI Regression Studio Backend",
    description="Production-ready FastAPI backend for regression analysis, data preprocessing, graph plotting, agentic AI LLM reports, and PDF downloads.",
    version="1.0.0"
)

# Configure CORS. Override via ALLOWED_ORIGINS (comma-separated) in production.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3001,http://localhost:8001")
allowed_origins = [origin.strip() for origin in _allowed_origins.split(",") if origin.strip()]


def _is_baas_public_path(path: str) -> bool:
    return path.startswith("/baas/data") or path.startswith("/baas/end-users")


class PathAwareCORSMiddleware:
    """Applies open CORS (any origin) to the BaaS public SDK surface
    (/baas/data/*, /baas/end-users/*) and the existing strict ALLOWED_ORIGINS
    policy to everything else, including /baas/projects/* management routes.

    NOTE: mounting a separate FastAPI sub-app with its own CORSMiddleware
    does NOT achieve this — Starlette middleware wraps the whole ASGI app
    including anything routed to a Mount, so the outer CORSMiddleware
    intercepts (and, for a disallowed origin, rejects) preflight OPTIONS
    requests before they ever reach a mounted sub-app's own middleware.
    Confirmed by testing: an OPTIONS preflight to /baas/data/* with an
    arbitrary Origin returned the strict policy's 400 rejection even with a
    sub-app mounted at /baas with allow_origins=["*"]. This middleware
    instead wraps a single app twice (open + strict CORSMiddleware, both
    pointing at the same inner app) and picks which wrapped call to invoke
    per-request based on path — no sub-app/Mount involved.
    """

    def __init__(self, app, strict_origins: list[str]):
        self.strict_cors = CORSMiddleware(
            app, allow_origins=strict_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
        )
        # allow_credentials=False is correct and required here: BaaS auth is
        # header-based (X-Public-Key/X-Secret-Key/Bearer), not cookie-based,
        # so there's no CORS credentials mode to reconcile with the wildcard
        # origin (browsers reject allow_origins=["*"] + allow_credentials=True).
        self.open_cors = CORSMiddleware(
            app, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and _is_baas_public_path(scope.get("path", "")):
            await self.open_cors(scope, receive, send)
        else:
            await self.strict_cors(scope, receive, send)


app.add_middleware(PathAwareCORSMiddleware, strict_origins=allowed_origins)

# Resolve absolute path to the static directory
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "graphs"), exist_ok=True)
os.makedirs(os.path.join(static_dir, "reports"), exist_ok=True)

# Mount static folder to serve generated graph images and PDF reports
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Register API routers
app.include_router(auth_router)
app.include_router(dataset_router)
app.include_router(preprocess_router)
app.include_router(training_router)
app.include_router(prediction_router)
app.include_router(ai_router)
app.include_router(report_router)
app.include_router(dashboard_router)
app.include_router(data_sources_router)
app.include_router(hf_router)
app.include_router(db_query_router)
app.include_router(baas_management_router)
app.include_router(baas_public_router, prefix="/baas")

# Database connection events
@app.on_event("startup")
async def startup_event():
    await db_client.connect()
    warn_if_encryption_key_unconfigured()

@app.on_event("shutdown")
async def shutdown_event():
    # If using Motor/MongoDB client, close it
    if db_client.client:
        db_client.client.close()
        logger.info("MongoDB connection closed.")

# Consistent error envelope for explicit HTTPExceptions raised in route handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=err(exc.status_code, str(exc.detail)),
    )

# Global exception handler for any unhandled error
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=err(500, f"An unexpected server error occurred: {str(exc)}"),
    )

# Root endpoint for health check (left unenveloped — not consumed by the frontend as data)
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Regression Studio Backend",
        "database_mode": "JSON Fallback" if db_client.use_fallback else "MongoDB Active"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    # Run server
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
