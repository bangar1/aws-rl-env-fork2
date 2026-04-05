# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Aws Rl Env Environment.

This module creates an HTTP server that exposes the AwsRlEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required. Install dependencies with 'uv sync'") from e

from models import AwsRlAction, AwsRlObservation
from server.aws_rl_env_environment import AwsRlEnvironment

# Force ENABLE_WEB_INTERFACE=false so OpenEnv creates API-only app (no Gradio)
os.environ["ENABLE_WEB_INTERFACE"] = "false"

app = create_app(
    AwsRlEnvironment,
    AwsRlAction,
    AwsRlObservation,
    env_name="aws_rl_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

# ---------------------------------------------------------------------------
# Stateful web playground endpoints
# ---------------------------------------------------------------------------
# OpenEnv's HTTP /reset and /step create a new env per request (stateless).
# The web playground needs state across requests, so we maintain a shared
# environment instance and expose /web/reset and /web/step.
# ---------------------------------------------------------------------------

_env = AwsRlEnvironment()


class WebStepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/web/reset", include_in_schema=False)
async def web_reset():
    obs = _env.reset()
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/web/solution", include_in_schema=False)
async def web_solution():
    """Return the next solution command for the current task step."""
    if not _env._current_task:
        return {"command": None, "error": "No active task. Start a new episode first."}

    from server.services.task_solutions import get_next_solution

    result = get_next_solution(
        task_id=_env._current_task.task_id,
        backend=_env._backend,
        tracker=_env._tracker,
    )
    result["task_id"] = _env._current_task.task_id
    return result


@app.get("/web/state", include_in_schema=False)
async def web_state():
    """Return the full AwsRlState for the web UI."""
    return _env.state.model_dump()


@app.post("/web/step", include_in_schema=False)
async def web_step(request: WebStepRequest = Body(...)):
    action = AwsRlAction(**request.action)
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


# ---------------------------------------------------------------------------
# Custom web UI
# ---------------------------------------------------------------------------

_server_dir = Path(__file__).parent
_templates = Jinja2Templates(directory=str(_server_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(_server_dir / "static")), name="static")


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/web")


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def web_ui(request: Request):
    return _templates.TemplateResponse(request=request, name="index.html")


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m aws_rl_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn aws_rl_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
