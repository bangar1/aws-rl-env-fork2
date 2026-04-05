"""Tests for drift detection tasks (expert tier) — verifies setup and state checks.

Drift tasks provision correct infrastructure via setup_commands, then the agent
must audit and fix any drifts. This test verifies that:
1. All setup_commands execute successfully against MiniStack
2. After setup (no drift applied), all state_checks pass
3. The grader marks the task as achieved when state is correct

Run inside Docker:
    docker exec <container> python -m pytest tests/test_drift_tasks.py -v
"""

import json

import pytest
import yaml
from pathlib import Path

from models import SuccessCriteria, Task, TaskID, TaskDifficulty, SetupCommand
from server.services.aws_backend import AwsBackend
from server.services.task_grader import TaskGrader
from server.services.episode_tracker import EpisodeTracker
from server.services.resource_verifier import ResourceVerifier

TASKS_FILE = Path(__file__).resolve().parent.parent / "server" / "services" / "tasks" / "drift.yaml"


@pytest.fixture(scope="module")
def all_drift_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return yaml.safe_load(f)


@pytest.fixture
def backend() -> AwsBackend:
    b = AwsBackend()
    b.reset_environment()
    return b


@pytest.fixture
def grader(backend: AwsBackend) -> TaskGrader:
    return TaskGrader(backend)


def _build_task(entry: dict) -> Task:
    return Task(
        task_id=TaskID(entry["task_id"]),
        difficulty=TaskDifficulty.EXPERT,
        description=entry["description"],
        success_criteria=SuccessCriteria(**entry.get("success_criteria", {})),
        setup_commands=[
            SetupCommand(command=cmd) if isinstance(cmd, str) else SetupCommand(**cmd)
            for cmd in entry.get("setup_commands", [])
        ],
        desired_state_spec=entry.get("desired_state_spec"),
        possible_drifts=[
            SetupCommand(command=d["command"]) if isinstance(d, dict) else SetupCommand(command=d)
            for d in entry.get("possible_drifts", [])
        ],
    )


def _get_task_ids(tasks: list[dict]) -> list[int]:
    return [t["task_id"] for t in tasks]


# Load task IDs at import time for parametrize
with open(TASKS_FILE) as _f:
    _ALL_ENTRIES = yaml.safe_load(_f)
    _TASK_IDS = [t["task_id"] for t in _ALL_ENTRIES]



# ---------------------------------------------------------------------------
# Test 1: All setup_commands execute successfully
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", _TASK_IDS, ids=[f"task_{t}" for t in _TASK_IDS])
def test_drift_setup_commands_execute(
    task_id: int, all_drift_tasks: list[dict], backend: AwsBackend,
) -> None:
    """Every setup_command must succeed against MiniStack."""
    backend.reset_environment()
    entry = next(t for t in all_drift_tasks if t["task_id"] == task_id)
    setup_cmds = entry.get("setup_commands", [])

    for i, cmd in enumerate(setup_cmds):
        success, stdout, stderr = backend.execute_command(cmd)
        assert success, (
            f"Setup command {i + 1}/{len(setup_cmds)} failed for task {task_id}.\n"
            f"  Command: {cmd}\n"
            f"  Stderr: {stderr}"
        )


# ---------------------------------------------------------------------------
# Test 2: After setup, all state_checks pass (no drift applied)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", _TASK_IDS, ids=[f"task_{t}" for t in _TASK_IDS])
def test_drift_state_checks_pass_after_setup(
    task_id: int, all_drift_tasks: list[dict], backend: AwsBackend,
) -> None:
    """After running setup_commands, all state_checks must pass."""
    backend.reset_environment()
    entry = next(t for t in all_drift_tasks if t["task_id"] == task_id)
    verifier = ResourceVerifier(backend)

    # Run setup
    for cmd in entry.get("setup_commands", []):
        backend.execute_command(cmd)

    # Verify each state_check
    state_checks = entry.get("success_criteria", {}).get("state_checks", [])
    for i, check in enumerate(state_checks):
        passed = verifier.check_state(check)
        assert passed, (
            f"State check {i + 1}/{len(state_checks)} failed for task {task_id}.\n"
            f"  Check: {json.dumps(check, indent=2)}"
        )


# ---------------------------------------------------------------------------
# Test 3: Grader marks task as achieved after setup + fix commands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", _TASK_IDS, ids=[f"task_{t}" for t in _TASK_IDS])
def test_drift_grading_after_setup(
    task_id: int, all_drift_tasks: list[dict], backend: AwsBackend, grader: TaskGrader,
) -> None:
    """The grader should mark the task as achieved when state is correct."""
    backend.reset_environment()
    entry = next(t for t in all_drift_tasks if t["task_id"] == task_id)
    task = _build_task(entry)

    # Run setup commands and record them as the agent's "fix" actions.
    # Commands are only run once — the tracker records the initial successful
    # provisioning, which satisfies both the state_checks and services requirements.
    tracker = EpisodeTracker()
    for cmd in entry.get("setup_commands", []):
        success, stdout, stderr = backend.execute_command(cmd)
        step = tracker.record_step(cmd, success, stdout, stderr)

    result = grader.grade(task, tracker, step)

    assert result.task_achieved, (
        f"Task {task_id} not achieved.\n"
        f"  Description: {entry['description']}\n"
        f"  Reason: {result.reason}\n"
        f"  Reward: {result.reward}"
    )


# ---------------------------------------------------------------------------
# Test 4: Each possible drift breaks at least one state_check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", _TASK_IDS, ids=[f"task_{t}" for t in _TASK_IDS])
def test_drift_mutations_break_state(
    task_id: int, all_drift_tasks: list[dict], backend: AwsBackend,
) -> None:
    """Applying each drift mutation should cause at least one state_check to fail."""
    entry = next(t for t in all_drift_tasks if t["task_id"] == task_id)
    verifier = ResourceVerifier(backend)
    state_checks = entry.get("success_criteria", {}).get("state_checks", [])
    drifts = entry.get("possible_drifts", [])

    if not drifts:
        pytest.skip("No possible drifts defined")

    for drift in drifts:
        drift_cmd = drift["command"] if isinstance(drift, dict) else drift
        drift_desc = drift.get("description", drift_cmd) if isinstance(drift, dict) else drift_cmd

        # Fresh setup
        backend.reset_environment()
        for cmd in entry.get("setup_commands", []):
            backend.execute_command(cmd)

        # Apply drift
        backend.execute_command(drift_cmd)

        # At least one state_check should now fail
        all_pass = all(verifier.check_state(check) for check in state_checks)
        assert not all_pass, (
            f"Drift did not break any state_check for task {task_id}.\n"
            f"  Drift: {drift_desc}"
        )
