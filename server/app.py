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

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
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
from server.services.aws_strategy import AwsStrategy
from server.services.simulator_strategy import SimulatorStrategy

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

# Clamp to >= 1. POOL_SIZE=0 or negative is interpreted as single-MiniStack
# legacy mode — the same as POOL_SIZE=1. Without the clamp OpenEnv's
# create_app() would reject max_concurrent_envs=0 at import time.
POOL_SIZE = max(int(os.getenv("AWS_RL_ENV_POOL_SIZE", "1")), 1)
BASE_MINISTACK_PORT = int(os.getenv("AWS_RL_ENV_MINISTACK_BASE_PORT", "4566"))
BACKEND_TYPE = os.getenv("BACKEND_TYPE", "simulator")  # "simulator" | "aws"

# Constant, dedicated MiniStack port for the web playground. Kept outside the
# pool's range so a WebSocket session can never acquire it, eliminating the
# state-bleed risk that previously gated the web UI when POOL_SIZE > 1.
WEB_MINISTACK_PORT = int(os.getenv("AWS_RL_ENV_WEB_MINISTACK_PORT", "4565"))

if (
    BACKEND_TYPE != "aws"
    and POOL_SIZE > 1
    and BASE_MINISTACK_PORT <= WEB_MINISTACK_PORT < BASE_MINISTACK_PORT + POOL_SIZE
):
    raise RuntimeError(
        f"AWS_RL_ENV_WEB_MINISTACK_PORT={WEB_MINISTACK_PORT} collides with pool range "
        f"[{BASE_MINISTACK_PORT}..{BASE_MINISTACK_PORT + POOL_SIZE - 1}]. "
        f"Pick a port outside the pool's range."
    )


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
    backend_type: str = "simulator",
) -> tuple[MiniStackPool | None, Callable[[], AwsRlEnvironment]]:
    """Build the WebSocket-session env factory.

    Returns (pool, factory).

    - backend_type="aws": pool is skipped; all sessions share AwsStrategy.
    - pool_size <= 1: returns (None, plain SimulatorStrategy constructor).
    - pool_size >  1: returns (MiniStackPool, factory that acquires a port,
      constructs AwsRlEnvironment bound to that port, and injects a
      release callback so env.close() returns the port to the pool).

    Extracted as a pure function so tests can exercise both branches
    without reloading the module.
    """
    if backend_type == "aws":
        return None, lambda: AwsRlEnvironment(strategy=AwsStrategy())

    if pool_size > 1:
        pool = MiniStackPool(range(base_port, base_port + pool_size))

        def factory() -> AwsRlEnvironment:
            port = pool.acquire()
            env = AwsRlEnvironment(
                strategy=SimulatorStrategy(f"http://localhost:{port}")
            )
            env._pool_release = lambda p=port: pool.release(p)
            return env

        return pool, factory

    return None, lambda: AwsRlEnvironment(strategy=SimulatorStrategy())


_pool, _env_factory = make_env_factory(POOL_SIZE, BASE_MINISTACK_PORT, BACKEND_TYPE)


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
# When POOL_SIZE > 1 the pool owns [BASE..BASE+N-1]; the web UI uses a
# dedicated MiniStack on WEB_MINISTACK_PORT (constant, outside the pool's
# range) so it can never collide with a WebSocket session. That MiniStack is
# spawned lazily on the first /web/* request — training-only deployments pay
# zero cost. Subsequent requests reuse the cached _web_env.
# ---------------------------------------------------------------------------

_web_env: AwsRlEnvironment | None = None
_web_env_lock = threading.Lock()


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _resolve_ministack_bin() -> str:
    """Find the ministack entry point. Prefer the same venv as the running
    Python (sys.executable's bin dir) before falling back to PATH — uvicorn
    invoked via /full/path/to/.venv/bin/uvicorn doesn't always have the venv
    on PATH, so a bare "ministack" lookup would FileNotFoundError.
    """
    candidate = Path(sys.executable).parent / "ministack"
    if candidate.exists():
        return str(candidate)
    on_path = shutil.which("ministack")
    if on_path:
        return on_path
    raise RuntimeError(
        "Could not find the 'ministack' executable. Install with `uv sync` "
        "or ensure the active venv's bin directory is on PATH."
    )


def _spawn_web_ministack(port: int, timeout_s: float = 10.0) -> None:
    if _port_listening(port):
        return
    subprocess.Popen(
        [_resolve_ministack_bin(), "-d"],
        env={**os.environ, "GATEWAY_PORT": str(port)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _port_listening(port):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Web MiniStack failed to bind {port} within {timeout_s}s")


def _get_web_env() -> AwsRlEnvironment:
    global _web_env
    if _web_env is not None:
        return _web_env
    with _web_env_lock:
        if _web_env is not None:
            return _web_env
        if BACKEND_TYPE == "aws":
            _web_env = AwsRlEnvironment(strategy=AwsStrategy())
        elif POOL_SIZE > 1:
            _spawn_web_ministack(WEB_MINISTACK_PORT)
            _web_env = AwsRlEnvironment(
                strategy=SimulatorStrategy(f"http://localhost:{WEB_MINISTACK_PORT}")
            )
        else:
            _web_env = AwsRlEnvironment()
        return _web_env


class WebStepRequest(BaseModel):
    action: Dict[str, Any]


@app.post("/web/reset", include_in_schema=False)
async def web_reset():
    env = await asyncio.to_thread(_get_web_env)
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/web/solution", include_in_schema=False)
async def web_solution():
    """Return the next solution command for the current task step."""
    env = await asyncio.to_thread(_get_web_env)
    if not env._current_task:
        return {
            "command": None,
            "error": "No active task. Start a new episode first.",
        }

    from server.services.task_solutions import get_next_solution

    result = get_next_solution(
        task_id=env._current_task.task_id,
        backend=env._backend,
        tracker=env._tracker,
    )
    result["task_id"] = env._current_task.task_id
    return result


@app.get("/web/state", include_in_schema=False)
async def web_state():
    """Return the full AwsRlState for the web UI."""
    env = await asyncio.to_thread(_get_web_env)
    return env.state.model_dump()


@app.post("/web/step", include_in_schema=False)
async def web_step(request: WebStepRequest = Body(...)):
    env = await asyncio.to_thread(_get_web_env)
    action = AwsRlAction(**request.action)
    obs = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


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
