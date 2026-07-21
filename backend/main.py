"""SkyPredict FastAPI backend — application entrypoint.

Loads the exported model artifacts once on startup, exposes the RUL prediction
pipeline over HTTP, and serves interactive docs at /docs.

Run from the repo root:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.routers import explainability, health, optional_module, prediction
from backend.services.artifacts import load_artifacts

logger = logging.getLogger("skypredict")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; clear them on shutdown."""
    try:
        app.state.artifacts = load_artifacts(
            settings.model_path, settings.metadata_path
        )
        logger.info("Loaded model artifacts from %s", settings.model_path)
    except FileNotFoundError as exc:
        # Start anyway so /health reports the problem instead of crashing.
        app.state.artifacts = None
        logger.error("%s", exc)
    yield
    app.state.artifacts = None


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.description,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a clear, structured message for invalid input data."""
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid input data.",
            "errors": [
                {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                for e in exc.errors()
            ],
        },
    )


app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(explainability.router)
app.include_router(optional_module.router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "docs": "/docs", "health": "/health"}


def _example_reading() -> dict | None:
    """Build a realistic EngineReading example from the fitted scaler's ranges.

    Uses the per-feature midpoint of the scaler's learned min/max, so the Swagger
    "Try it out" body is pre-filled with in-range values derived from the model —
    not hardcoded. Returns None (docs render without an example) if unavailable.
    """
    artifacts = getattr(app.state, "artifacts", None)
    if artifacts is None:
        return None
    scaler = artifacts.model[0]
    if not hasattr(scaler, "data_min_"):
        return None
    midpoints = (scaler.data_min_ + scaler.data_max_) / 2.0
    reading = {
        name: round(float(value), 4)
        for name, value in zip(artifacts.feature_names, midpoints)
    }
    return {"unit_number": 1, "time_in_cycles": 1, **reading}


def custom_openapi() -> dict:
    """Inject the model-derived example into the EngineReading schema."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    example = _example_reading()
    if example is not None:
        schema["components"]["schemas"]["EngineReading"]["example"] = example
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
