"""Application server bootstrap for ``dataops-gym``.

This module is responsible for exposing runtime APIs, health endpoints, and
deployment-facing application setup for the environment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from threading import RLock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env import DataOpsEnv
from models import Action


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="dataops-env", version="1.0.0")
active_env: Optional[DataOpsEnv] = None
active_env_lock = RLock()


class ResetRequest(BaseModel):
    """Optional reset controls for reproducible task selection."""

    seed: int = Field(default=0, description="Deterministic seed for task sampling.")
    task_name: str | None = Field(
        default=None,
        description="Optional fixed task name: easy, medium, or hard.",
    )


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


@app.get("/health")
def health() -> Dict[str, str]:
    """Return a lightweight deployment health signal."""

    return {"status": "healthy"}


@app.post("/reset")
def reset(payload: ResetRequest | None = None) -> Dict[str, Any]:
    """Reset the environment and return the initial observation."""

    try:
        request = payload or ResetRequest()
        env = DataOpsEnv(seed=request.seed, task_name=request.task_name)
        observation = env.reset()

        global active_env
        with active_env_lock:
            previous_env = active_env
            active_env = env

        return {
            "task_name": env.state().get("task_name"),
            "observation": observation.model_dump(),
        }
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
