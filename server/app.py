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
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

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

# ---------------------------------------------------------------------------
# Parallel Concurrency with MiniStack Pool
# ---------------------------------------------------------------------------
# POOL_SIZE=1 (default) preserves the original single-MiniStack behaviour:
# no pool object, no lock, one session at a time.
# POOL_SIZE>1 spins up a MiniStackPool and binds each WebSocket session to
# its own MiniStack for that session's lifetime. Each MiniStack must already
# be running on localhost at BASE_PORT..BASE_PORT+POOL_SIZE-1 (the Dockerfile
# starts them up during container boot).
# ---------------------------------------------------------------------------

POOL_SIZE = int(os.getenv("AWS_RL_ENV_POOL_SIZE", "1"))
BASE_MINISTACK_PORT = int(os.getenv("AWS_RL_ENV_MINISTACK_BASE_PORT", "4566"))


class MiniStackPool:
    """Thread-safe free-list of MiniStack ports.

    Used when POOL_SIZE > 1 so that N concurrent WebSocket sessions each
    get their own MiniStack process. `acquire()` hands out a port from the
    free list; `release()` returns it when the session ends so the next
    session can reuse that MiniStack.
    """

    def __init__(self, ports: Iterable[int]) -> None:
        self._free: list[int] = list(ports)
        self._lock = threading.Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._free:
                raise RuntimeError("MiniStack pool exhausted")
            return self._free.pop()

    def release(self, port: int) -> None:
        with self._lock:
            self._free.append(port)

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._free)


def make_env_factory(
    pool_size: int,
    base_port: int,
) -> tuple[MiniStackPool | None, Callable[[], AwsRlEnvironment]]:
    """Build the WebSocket-session env factory.

    Returns (pool, factory).

    - pool_size <= 1: returns (None, plain AwsRlEnvironment constructor).
    - pool_size >  1: returns (MiniStackPool, factory that acquires a port,
      constructs AwsRlEnvironment bound to that port, and injects a
      release callback so env.close() returns the port to the pool).

    Extracted as a pure function so tests can exercise both branches
    without reloading the module.
    """
    if pool_size > 1:
        pool = MiniStackPool(range(base_port, base_port + pool_size))

        def factory() -> AwsRlEnvironment:
            port = pool.acquire()
            env = AwsRlEnvironment(aws_infra_url=f"http://localhost:{port}")
            env._pool_release = lambda p=port: pool.release(p)
            return env

        return pool, factory

    return None, lambda: AwsRlEnvironment()


_pool, _env_factory = make_env_factory(POOL_SIZE, BASE_MINISTACK_PORT)


app = create_app(
    _env_factory,
    AwsRlAction,
    AwsRlObservation,
    env_name="aws_rl_env",
    max_concurrent_envs=POOL_SIZE,
)

# ---------------------------------------------------------------------------
# Stateful web playground endpoints
# ---------------------------------------------------------------------------
# OpenEnv's HTTP /reset and /step create a new env per request (stateless).
# The web playground needs state across requests, so we maintain a shared
# environment instance and expose /web/reset and /web/step.
#
# Only mounted when POOL_SIZE <= 1. With a pool active, port 4566 is
# claimed by the pool and a shared web _env would collide with the
# per-session MiniStacks.
# If POOL_SIZE=8 and web mounts anyway, the module-level _env = AwsRlEnvironment()
# defaults to http://localhost:4566 — which is also in the pool's range.
# Any /web/step clobbers the MiniStack currently held by a WS session that
# acquired port 4566. State corrupts both ways: web user's bucket appears in a
# GRPO rollout; pool rollout's drift mutations show up in the web UI.


# ---------------------------------------------------------------------------

if POOL_SIZE <= 1:
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
            return {
                "command": None,
                "error": "No active task. Start a new episode first.",
            }

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
    app.mount(
        "/static", StaticFiles(directory=str(_server_dir / "static")), name="static"
    )

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
