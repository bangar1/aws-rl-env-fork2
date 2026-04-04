# CLAUDE.md

## Project Overview

AWS RL Environment — a Gymnasium-style reinforcement learning environment for training LLM agents on real-world AWS cloud operations. Features a vendored MiniStack emulator (34 AWS services in-memory), priority-queue curriculum learning with spaced repetition, and FastAPI server with OpenEnv-compatible HTTP/WebSocket endpoints.

## Tech Stack

- **Python 3.12** (managed with UV)
- **FastAPI + Uvicorn + OpenEnv** for the server
- **Pydantic** for data models and validation
- **Ruff** for linting/formatting, **MyPy** for type checking
- **YAML** for task definitions
- **Docker** multi-stage build

## Common Commands

```bash
make install          # Install runtime deps (uv, frozen)
make install-dev      # Install with dev extras (pytest, ruff, mypy)
make run              # Start MiniStack + FastAPI server locally
make format           # Format code with Ruff
make lint             # Check code style with Ruff
make lint-fix         # Auto-fix Ruff violations
make typecheck        # Run MyPy
make check            # Run lint + typecheck
make docker-build     # Build Docker image
make docker-run       # Run container on port 8000
make clean            # Remove build artifacts and caches
```

## Architecture

### Episode Lifecycle
1. `reset()` — wipes MiniStack, selects next task via curriculum, provisions setup commands, returns observation
2. `step(action)` — validates command (must start with `aws`), executes against MiniStack, grades, returns observation
3. Terminates when `task_achieved == True` or max steps reached

### Key Directories
- `server/` — FastAPI app, core environment, and services
- `server/services/` — Single-responsibility services (curriculum, grading, verification, etc.)
- `server/services/tasks/` — YAML task definitions across 5 tiers (warmup → expert)
- `aws_infra/` — Vendored MiniStack emulator (do NOT modify unless necessary)
- `models.py` — All Pydantic data models (root level)

### Services
- **AwsBackend** — executes AWS CLI commands against MiniStack (port 4566)
- **Curriculum** — priority-queue task selection with weakness, novelty, spaced repetition
- **TaskGrader** — evaluates completion and shapes rewards [0.0, 1.0]
- **ResourceVerifier** — ground-truth state verification against MiniStack
- **EpisodeTracker** — step history tracking and command deduplication
- **EnvironmentDesigner** — provisions initial AWS state via setup commands

## Code Conventions

- Type hints mandatory on all function signatures
- Module-level docstrings on all Python files
- Ruff formatting (black-compatible), MyPy type checking
- MyPy and Ruff both **exclude** `aws_infra/`
- Logging via `logging.getLogger(__name__)` per module
- Commit messages follow conventional format: `feat:`, `fix:`, `chore:`, `refactor:`