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

from backend.config import MODEL_FILES, settings
from backend.routers import explainability, health, optional_module, prediction
from backend.services.artifacts import load_artifacts

logger = logging.getLogger("skypredict")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every configured model into a registry at startup."""
    registry = {}
    for key in MODEL_FILES:
        try:
            registry[key] = load_artifacts(*settings.artifact_paths(key))
            logger.info("Loaded model '%s'", key)
        except (FileNotFoundError, ValueError) as exc:
            # Skip a missing/malformed model so the rest still serve.
            logger.error("Could not load model '%s': %s", key, exc)
    app.state.models = registry
    yield
    app.state.models = {}


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
    """Build a realistic EngineReading example from the fitted scaler.

    Inverse-transforms the scaled origin, giving each feature's central training
    value (the mean, for StandardScaler). The Swagger "Try it out" body is then
    pre-filled with in-range values derived from the model — not hardcoded.
    Returns None (docs render without an example) if unavailable.
    """
    registry = getattr(app.state, "models", {})
    artifacts = registry.get(settings.default_model)
    if artifacts is None:
        return None
    n = len(artifacts.feature_names)
    try:
        central = artifacts.model[:-1].inverse_transform([[0.0] * n])[0]
    except Exception:
        return None
    reading = {
        name: round(float(value), 4)
        for name, value in zip(artifacts.feature_names, central)
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
