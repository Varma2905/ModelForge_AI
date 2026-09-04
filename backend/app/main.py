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

from app.database.database import db_client
from app.api.auth_routes import router as auth_router
from app.api.dataset_routes import router as dataset_router
from app.api.dataset_source_routes import router as dataset_source_router
from app.api.preprocessing_routes import router as preprocess_router
from app.api.training_routes import router as training_router
from app.api.classify_routes import router as classify_router
from app.api.cluster_routes import router as cluster_router
from app.api.prediction_routes import router as prediction_router
from app.api.ai_routes import router as ai_router
from app.api.report_routes import router as report_router
from app.api.dashboard_routes import router as dashboard_router
from app.utils import crypto
from app.utils.response import err

app = FastAPI(
    title="ModelForge AI Studio Backend",
    description="Production-ready FastAPI backend for regression analysis, data preprocessing, graph plotting, agentic AI LLM reports, and PDF downloads.",
    version="1.0.0"
)

# Configure CORS. Override via ALLOWED_ORIGINS (comma-separated) in production.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3001,http://localhost:8001")
allowed_origins = [origin.strip() for origin in _allowed_origins.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Off by default so local HTTP dev keeps working unchanged. Set
# REQUIRE_HTTPS=true in production (behind a TLS-terminating proxy/load
# balancer that sets X-Forwarded-Proto) to reject any request — including
# database-connection credentials — that arrives over plain HTTP.
_REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "false").strip().lower() == "true"


@app.middleware("http")
async def require_https(request: Request, call_next):
    if _REQUIRE_HTTPS:
        scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
        if scheme != "https":
            return JSONResponse(status_code=400, content=err(400, "HTTPS is required."))
    return await call_next(request)

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
app.include_router(dataset_source_router)
app.include_router(preprocess_router)
app.include_router(training_router)
app.include_router(classify_router)
app.include_router(cluster_router)
app.include_router(prediction_router)
app.include_router(ai_router)
app.include_router(report_router)
app.include_router(dashboard_router)

# Database connection events
@app.on_event("startup")
async def startup_event():
    await db_client.connect()
    crypto.warn_if_unconfigured()

@app.on_event("shutdown")
async def shutdown_event():
    pass

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
        "service": "ModelForge AI Studio Backend",
        "database_mode": "Local File Database"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    # Run server
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
