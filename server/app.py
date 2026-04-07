"""Application server bootstrap for ``dataops-gym``.

This module is responsible for exposing runtime APIs, health endpoints, and
deployment-facing application setup for the environment.
"""

from __future__ import annotations

from enum import Enum
import logging
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import DataOpsEnv
from models import Action


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="dataops-env",
    version="1.0.0",
    summary="Reasoning-first semantic data cleaning benchmark.",
    description=(
        "### DataOps Gym: Clean Data, Keep Truth\n"
        "A step-based evaluation environment for testing whether agents can detect issues, "
        "fix only with evidence, abstain with `cannot_determine` under ambiguity, and stay "
        "consistent across related records.\n\n"
        "**Tagline:** *Fix data without fabricating reality.*\n\n"
        "#### Why this API matters\n"
        "- Strict JSON action schema (no free-form outputs)\n"
        "- Reward shaping that penalizes hallucinations and over-correction\n"
        "- Cross-record consistency and uncertainty-aware scoring\n"
        
    ),
    contact={
        "name": "DataOps Gym",
        "url": "https://github.com/graheetphartyal23/Dataops--GYM",
    },
    docs_url=None,
)
active_env: Optional[DataOpsEnv] = None
active_env_lock = RLock()


class ResetRequest(BaseModel):
    """Optional reset controls for reproducible task selection."""

    seed: int = Field(default=0, description="Deterministic seed for task sampling.")
    task_name: "TaskName | None" = Field(
        default=None,
        description="Optional fixed task name: easy, medium, or hard.",
    )


class TaskName(str, Enum):
    """Allowed benchmark task names."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return a safe error payload for unexpected server failures."""

    logger.exception("Unhandled server error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the Space landing page to FastAPI docs."""

    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False)
def custom_docs() -> HTMLResponse:
    """Serve Swagger UI with a dark theme override."""

    swagger = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Docs",
        swagger_ui_parameters={
            "syntaxHighlight.theme": "obsidian",
            "displayRequestDuration": True,
        },
    )
    dark_css = """
    <style>
    html, body { background: #0b1020 !important; color: #e5e7eb !important; }
    .swagger-ui, .swagger-ui .topbar { background: #0b1020 !important; }
    .swagger-ui .topbar { border-bottom: 1px solid #1f2937 !important; }
    .swagger-ui .topbar a, .swagger-ui .topbar span { color: #e5e7eb !important; }

    /* Keep top API details readable: white card + black text */
    .swagger-ui .info {
      background: #ffffff !important;
      color: #111827 !important;
      border: 1px solid #e5e7eb !important;
      border-radius: 12px !important;
      padding: 18px !important;
      margin: 18px 0 24px 0 !important;
    }
    .swagger-ui .info .title, .swagger-ui .info h1, .swagger-ui .info h2,
    .swagger-ui .info h3, .swagger-ui .info p, .swagger-ui .info li,
    .swagger-ui .info a, .swagger-ui .info .base-url, .swagger-ui .info .version {
      color: #111827 !important;
    }
    .swagger-ui .info ul { margin: 10px 0 0 18px !important; }

    /* Default + Schemas sections as white cards with black text */
    .swagger-ui .opblock-tag {
      background: #ffffff !important;
      color: #111827 !important;
      border: 1px solid #e5e7eb !important;
      border-radius: 10px !important;
      padding: 10px 12px !important;
      margin-bottom: 12px !important;
    }
    .swagger-ui .opblock {
      background: #ffffff !important;
      border: 1px solid #e5e7eb !important;
      border-radius: 10px !important;
      margin: 0 0 14px 0 !important;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
    }
    .swagger-ui .opblock .opblock-summary {
      background: #ffffff !important;
      border-bottom: 1px solid #e5e7eb !important;
    }
    .swagger-ui .opblock .opblock-summary-method,
    .swagger-ui .opblock .opblock-summary-path,
    .swagger-ui .opblock .opblock-summary-path__deprecated,
    .swagger-ui .opblock .opblock-summary-description {
      color: #111827 !important;
      fill: #111827 !important;
    }
    .swagger-ui .opblock-section-header,
    .swagger-ui .responses-inner h4,
    .swagger-ui .responses-inner h5,
    .swagger-ui .tab li,
    .swagger-ui .parameter__type,
    .swagger-ui .model-title,
    .swagger-ui .models h4 {
      color: #111827 !important;
    }
    .swagger-ui .models {
      background: #ffffff !important;
      border: 1px solid #e5e7eb !important;
      border-radius: 10px !important;
      padding: 8px !important;
    }
    .swagger-ui .model-container, .swagger-ui .model-box {
      background: #ffffff !important;
      color: #111827 !important;
      border-color: #e5e7eb !important;
    }
    .swagger-ui .model, .swagger-ui .prop-name, .swagger-ui .prop-type, .swagger-ui .prop-format {
      color: #111827 !important;
    }
    .swagger-ui .response-col_status, .swagger-ui .response-col_description,
    .swagger-ui label, .swagger-ui .parameter__name,
    .swagger-ui table tbody tr td, .swagger-ui .responses-table, .swagger-ui .parameters-col_description {
      color: #111827 !important;
      background: #ffffff !important;
      border-color: #e5e7eb !important;
    }
    .swagger-ui input, .swagger-ui textarea, .swagger-ui select {
      background: #0f172a !important;
      color: #e5e7eb !important;
      border-color: #374151 !important;
    }
    .swagger-ui .btn.execute { background: #2563eb !important; color: white !important; }
    .swagger-ui .btn { border-color: #4b5563 !important; }
    </style>
    """
    html = swagger.body.decode("utf-8").replace("</head>", f"{dark_css}</head>")
    return HTMLResponse(content=html, status_code=200)


@app.get("/health")
def health() -> Dict[str, str]:
    """Return a lightweight deployment health signal."""

    return {"status": "healthy"}


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> Dict[str, Any]:
    """Reset the environment and return the initial observation."""

    try:
        request = payload or ResetRequest()
        env = DataOpsEnv(
            seed=request.seed,
            task_name=request.task_name.value if request.task_name is not None else None,
        )
        observation = env.reset()

        global active_env
        with active_env_lock:
            previous_env = active_env
            active_env = env

        return {
            "task_name": env.state().get("task_name"),
            "observation": observation.model_dump(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to reset environment", exc_info=exc)
        raise HTTPException(status_code=500, detail="Failed to reset environment") from exc
    finally:
        if "previous_env" in locals() and previous_env is not None:
            close_method = getattr(previous_env, "close", None)
            if callable(close_method):
                close_method()


@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    """Apply a single action to the environment and return the step result."""

    try:
        with active_env_lock:
            if active_env is None:
                raise HTTPException(
                    status_code=400,
                    detail="Environment not initialized. Call /reset first.",
                )
            observation, reward, done, info = active_env.step(action)

        return {
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to execute environment step", exc_info=exc)
        raise HTTPException(status_code=500, detail="Failed to execute step") from exc


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the current internal environment state as JSON."""

    try:
        with active_env_lock:
            if active_env is None:
                raise HTTPException(
                    status_code=400,
                    detail="Environment not initialized. Call /reset first.",
                )
            return active_env.state()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch environment state", exc_info=exc)
        raise HTTPException(status_code=500, detail="Failed to fetch state") from exc


def main() -> None:
    """Run the FastAPI application with uvicorn."""

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
