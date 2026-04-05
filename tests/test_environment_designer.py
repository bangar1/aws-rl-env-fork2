"""Unit tests for EnvironmentDesigner — tests provisioning and drift integration.

Run:
    docker exec <container> python -m pytest env/tests/test_environment_designer.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from models import Task, TaskID, TaskDifficulty, SuccessCriteria, SetupCommand
from server.services.environment_designer import (
    EnvironmentDesigner,
    ProvisionMethod,
    ProvisionResult,
)


def _task(
    setup_commands: list[SetupCommand] | None = None,
    possible_drifts: list[SetupCommand] | None = None,
) -> Task:
    return Task(
        task_id=TaskID(1),
        difficulty=TaskDifficulty.BEGINNER,
        description="test",
        success_criteria=SuccessCriteria(),
        setup_commands=setup_commands or [],
        possible_drifts=possible_drifts or [],
    )


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.execute_command.return_value = (True, "", "")
    return backend


@pytest.fixture
def designer(mock_backend: MagicMock) -> EnvironmentDesigner:
    return EnvironmentDesigner(mock_backend)


# ===================================================================
# apply — no setup commands
# ===================================================================


class TestApplyNoSetup:
    def test_no_commands_returns_success(self, designer: EnvironmentDesigner) -> None:
        result = designer.apply(_task())
        assert result.success
        assert result.resources_created == 0

    def test_no_commands_no_backend_calls(
        self, designer: EnvironmentDesigner, mock_backend: MagicMock
    ) -> None:
        designer.apply(_task())
        mock_backend.execute_command.assert_not_called()


# ===================================================================
# apply — CLI commands
# ===================================================================


class TestApplyCliCommands:
    def test_all_succeed(
        self, designer: EnvironmentDesigner, mock_backend: MagicMock
    ) -> None:
        task = _task(
            setup_commands=[
                SetupCommand(command="aws s3api create-bucket --bucket a"),
                SetupCommand(command="aws s3api create-bucket --bucket b"),
            ]
        )
        result = designer.apply(task)
        assert result.success
        assert result.resources_created == 2
        assert result.method == ProvisionMethod.CLI_COMMANDS
        assert mock_backend.execute_command.call_count == 2

    def test_failure_recorded_in_errors(
        self, designer: EnvironmentDesigner, mock_backend: MagicMock
    ) -> None:
        mock_backend.execute_command.side_effect = [
            (True, "", ""),
            (False, "", "bucket already exists"),
        ]
        task = _task(
            setup_commands=[
                SetupCommand(command="aws s3api create-bucket --bucket a"),
                SetupCommand(command="aws s3api create-bucket --bucket a"),
            ]
        )
        result = designer.apply(task)
        assert not result.success
        assert result.resources_created == 1
        assert len(result.errors) == 1
        assert "bucket already exists" in result.errors[0]

    def test_ignore_failure_continues(
        self, designer: EnvironmentDesigner, mock_backend: MagicMock
    ) -> None:
        mock_backend.execute_command.side_effect = [
            (False, "", "already exists"),
            (True, "", ""),
        ]
        task = _task(
            setup_commands=[
                SetupCommand(command="cmd1", ignore_failure=True),
                SetupCommand(command="cmd2"),
            ]
        )
        result = designer.apply(task)
        assert result.success  # ignored failure doesn't count
        assert result.resources_created == 1
        assert len(result.errors) == 0

    def test_multiple_failures(
        self, designer: EnvironmentDesigner, mock_backend: MagicMock
    ) -> None:
        mock_backend.execute_command.return_value = (False, "", "err")
        task = _task(
            setup_commands=[
                SetupCommand(command="cmd1"),
                SetupCommand(command="cmd2"),
                SetupCommand(command="cmd3"),
            ]
        )
        result = designer.apply(task)
        assert not result.success
        assert result.resources_created == 0
        assert len(result.errors) == 3

    def test_commands_executed_in_order(
        self, designer: EnvironmentDesigner, mock_backend: MagicMock
    ) -> None:
        task = _task(
            setup_commands=[
                SetupCommand(command="first"),
                SetupCommand(command="second"),
                SetupCommand(command="third"),
            ]
        )
        designer.apply(task)
        calls = [c.args[0] for c in mock_backend.execute_command.call_args_list]
        assert calls == ["first", "second", "third"]


# ===================================================================
# apply — drift integration
# ===================================================================


class TestApplyWithDrifts:
    def test_drifts_applied_after_setup(
        self, designer: EnvironmentDesigner, mock_backend: MagicMock
    ) -> None:
        task = _task(
            setup_commands=[SetupCommand(command="setup-cmd")],
            possible_drifts=[SetupCommand(command="drift-cmd", description="d")],
        )
        with patch.object(
            designer._drift_engine, "apply_drift", return_value=["d"]
        ) as mock_drift:
            result = designer.apply(task)
            mock_drift.assert_called_once_with(task)
        assert result.success

    def test_no_drifts_skips_drift_engine(self, designer: EnvironmentDesigner) -> None:
        task = _task(setup_commands=[SetupCommand(command="cmd")])
        with patch.object(designer._drift_engine, "apply_drift") as mock_drift:
            designer.apply(task)
            mock_drift.assert_not_called()


# ===================================================================
# ProvisionResult model
# ===================================================================


class TestProvisionResult:
    def test_defaults(self) -> None:
        r = ProvisionResult()
        assert r.success is True
        assert r.method == ProvisionMethod.CLI_COMMANDS
        assert r.resources_created == 0
        assert r.errors == []
