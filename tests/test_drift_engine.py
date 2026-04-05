"""Unit tests for DriftEngine — tests drift selection and application logic.

Run:
    docker exec <container> python -m pytest env/tests/test_drift_engine.py -v
"""

from unittest.mock import MagicMock

import pytest

from models import Task, TaskID, TaskDifficulty, SuccessCriteria, SetupCommand
from server.services.drift_engine import DriftEngine, _MIN_DRIFTS, _MAX_DRIFTS


def _task_with_drifts(n: int) -> Task:
    """Create a task with N possible drifts."""
    return Task(
        task_id=TaskID(1),
        difficulty=TaskDifficulty.EXPERT,
        description="test",
        success_criteria=SuccessCriteria(),
        possible_drifts=[
            SetupCommand(command=f"aws cmd-{i}", description=f"drift-{i}")
            for i in range(n)
        ],
    )


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.execute_command.return_value = (True, "", "")
    return backend


@pytest.fixture
def engine(mock_backend: MagicMock) -> DriftEngine:
    return DriftEngine(mock_backend)


# ===================================================================
# apply_drift
# ===================================================================


class TestApplyDrift:
    def test_no_drifts_returns_empty(self, engine: DriftEngine) -> None:
        task = Task(
            task_id=TaskID(1),
            difficulty=TaskDifficulty.EXPERT,
            description="t",
            success_criteria=SuccessCriteria(),
        )
        assert engine.apply_drift(task) == []

    def test_single_drift_always_selected(
        self, engine: DriftEngine, mock_backend: MagicMock
    ) -> None:
        task = _task_with_drifts(1)
        applied = engine.apply_drift(task)
        assert len(applied) == 1
        assert applied[0] == "drift-0"
        mock_backend.execute_command.assert_called_once_with("aws cmd-0")

    def test_selects_between_min_and_max(self, engine: DriftEngine) -> None:
        task = _task_with_drifts(10)
        for _ in range(20):
            applied = engine.apply_drift(task)
            assert _MIN_DRIFTS <= len(applied) <= _MAX_DRIFTS

    def test_never_exceeds_pool_size(self, engine: DriftEngine) -> None:
        task = _task_with_drifts(2)
        for _ in range(20):
            applied = engine.apply_drift(task)
            assert len(applied) <= 2

    def test_selected_drifts_are_unique(self, engine: DriftEngine) -> None:
        task = _task_with_drifts(5)
        for _ in range(20):
            applied = engine.apply_drift(task)
            assert len(applied) == len(set(applied))

    def test_failed_drift_not_in_applied(
        self, engine: DriftEngine, mock_backend: MagicMock
    ) -> None:
        mock_backend.execute_command.return_value = (False, "", "error")
        task = _task_with_drifts(1)
        applied = engine.apply_drift(task)
        assert len(applied) == 0

    def test_partial_failure_only_returns_successful(
        self, engine: DriftEngine, mock_backend: MagicMock
    ) -> None:
        task = _task_with_drifts(2)
        mock_backend.execute_command.side_effect = [
            (True, "", ""),
            (False, "", "fail"),
        ]
        applied = engine.apply_drift(task)
        assert len(applied) == 1

    def test_uses_description_as_label(self, engine: DriftEngine) -> None:
        task = Task(
            task_id=TaskID(1),
            difficulty=TaskDifficulty.EXPERT,
            description="t",
            success_criteria=SuccessCriteria(),
            possible_drifts=[
                SetupCommand(command="aws test", description="My drift label"),
            ],
        )
        applied = engine.apply_drift(task)
        assert applied == ["My drift label"]

    def test_uses_command_as_fallback_label(self, engine: DriftEngine) -> None:
        task = Task(
            task_id=TaskID(1),
            difficulty=TaskDifficulty.EXPERT,
            description="t",
            success_criteria=SuccessCriteria(),
            possible_drifts=[SetupCommand(command="aws fallback-cmd")],
        )
        applied = engine.apply_drift(task)
        assert applied == ["aws fallback-cmd"]


# ===================================================================
# _pick_count
# ===================================================================


class TestPickCount:
    def test_zero_pool(self) -> None:
        assert DriftEngine._pick_count(0) == 0

    def test_one_pool(self) -> None:
        assert DriftEngine._pick_count(1) == 1

    def test_two_pool_returns_two(self) -> None:
        # pool_size=2: lo=min(2,2)=2, hi=min(3,2)=2 => always 2
        assert DriftEngine._pick_count(2) == 2

    def test_large_pool_within_bounds(self) -> None:
        for _ in range(50):
            count = DriftEngine._pick_count(10)
            assert _MIN_DRIFTS <= count <= _MAX_DRIFTS
