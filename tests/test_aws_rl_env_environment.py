"""Unit tests for AwsRlEnvironment — tests reset/step lifecycle and edge cases.

All external dependencies (AwsBackend, Curriculum, TaskGrader, etc.) are mocked
so tests run without MiniStack.

Run:
    docker exec <container> python -m pytest env/tests/test_aws_rl_env_environment.py -v
"""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from models import (
    AwsRlAction,
    AwsRlObservation,
    Task,
    TaskID,
    TaskDifficulty,
    SuccessCriteria,
)
from server.services.task_grader import GradeResult
from server.services.episode_tracker import StepRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_TASK = Task(
    task_id=TaskID(1),
    difficulty=TaskDifficulty.WARMUP,
    description="List S3 buckets",
    success_criteria=SuccessCriteria(command_contains="s3", operation="ls"),
)


def _make_env():
    """Create an AwsRlEnvironment with all dependencies mocked."""
    with (
        patch("server.aws_rl_env_environment.AwsBackend") as MockBackend,
        patch("server.aws_rl_env_environment.Curriculum") as MockCurriculum,
        patch("server.aws_rl_env_environment.TaskGrader") as MockGrader,
        patch("server.aws_rl_env_environment.EnvironmentDesigner") as MockDesigner,
        patch("server.aws_rl_env_environment.ChaosEngine") as MockChaos,
        patch("server.aws_rl_env_environment.HintProvider") as MockHint,
    ):
        from server.aws_rl_env_environment import AwsRlEnvironment

        env = AwsRlEnvironment()

        # Grab mock instances
        backend = MockBackend.return_value
        curriculum = MockCurriculum.return_value
        grader = MockGrader.return_value
        designer = MockDesigner.return_value
        chaos = MockChaos.return_value
        hint = MockHint.return_value

        # Default behaviors
        curriculum.next_task.return_value = _DUMMY_TASK
        curriculum.chaos_probability = 0.0
        backend.execute_command.return_value = (True, "output", "")
        chaos.chaos_occurred = False
        grader.grade.return_value = GradeResult(
            task_achieved=False, partial_progress=0.0, reward=0.0, reason="not done"
        )

        return env, backend, curriculum, grader, designer, chaos, hint


# ===================================================================
# reset()
# ===================================================================


class TestReset:
    def test_returns_observation(self) -> None:
        env, *_ = _make_env()
        obs = env.reset()
        assert isinstance(obs, AwsRlObservation)

    def test_resets_backend(self) -> None:
        env, backend, *_ = _make_env()
        env.reset()
        backend.reset_environment.assert_called_once()

    def test_gets_next_task_from_curriculum(self) -> None:
        env, _, curriculum, *_ = _make_env()
        env.reset()
        curriculum.next_task.assert_called_once()

    def test_applies_designer(self) -> None:
        env, _, _, _, designer, *_ = _make_env()
        env.reset()
        designer.apply.assert_called_once_with(_DUMMY_TASK)

    def test_obs_contains_task(self) -> None:
        env, *_ = _make_env()
        obs = env.reset()
        assert obs.task == _DUMMY_TASK

    def test_obs_step_count_zero(self) -> None:
        env, *_ = _make_env()
        obs = env.reset()
        assert obs.step_count == 0

    def test_obs_not_done(self) -> None:
        env, *_ = _make_env()
        obs = env.reset()
        assert obs.done is False
        assert obs.reward == 0.0

    def test_obs_command_output_is_reset_message(self) -> None:
        env, *_ = _make_env()
        obs = env.reset()
        assert "reset" in obs.command_output.lower()

    def test_custom_episode_id(self) -> None:
        env, *_ = _make_env()
        obs = env.reset(episode_id="my-ep-123")
        assert obs.episode_id == "my-ep-123"

    def test_auto_episode_id(self) -> None:
        env, *_ = _make_env()
        obs = env.reset()
        assert len(obs.episode_id) > 0  # UUID generated

    def test_resets_chaos_engine(self) -> None:
        env, _, _, _, _, chaos, _ = _make_env()
        env.reset()
        chaos.reset.assert_called_once()

    def test_consecutive_resets_get_fresh_state(self) -> None:
        env, backend, *_ = _make_env()
        obs1 = env.reset()
        obs2 = env.reset()
        assert obs1.episode_id != obs2.episode_id
        assert backend.reset_environment.call_count == 2


# ===================================================================
# step() — non-AWS command rejection
# ===================================================================


class TestStepRejection:
    def test_non_aws_command_rejected(self) -> None:
        env, *_ = _make_env()
        env.reset()
        obs = env.step(AwsRlAction(command="ls -la"))
        assert not obs.command_success
        assert "Only AWS CLI" in obs.error
        assert obs.reward == 0.0
        assert not obs.task_achieved

    def test_empty_command_rejected(self) -> None:
        env, *_ = _make_env()
        env.reset()
        obs = env.step(AwsRlAction(command=""))
        assert not obs.command_success

    def test_whitespace_only_rejected(self) -> None:
        env, *_ = _make_env()
        env.reset()
        obs = env.step(AwsRlAction(command="   "))
        assert not obs.command_success

    def test_shell_injection_rejected(self) -> None:
        env, *_ = _make_env()
        env.reset()
        obs = env.step(AwsRlAction(command="rm -rf / && aws s3 ls"))
        assert not obs.command_success

    def test_rejected_command_increments_step_count(self) -> None:
        env, *_ = _make_env()
        env.reset()
        obs = env.step(AwsRlAction(command="not-aws"))
        assert obs.step_count == 1


# ===================================================================
# step() — hint system
# ===================================================================


class TestStepHints:
    def test_hint_request_returns_hint_text(self) -> None:
        env, _, _, _, _, _, hint = _make_env()
        hint.get_hint.return_value = "Try using s3"
        env.reset()
        obs = env.step(AwsRlAction(command="aws help --task-hint"))
        assert obs.command_output == "Try using s3"
        assert obs.hint_text == "Try using s3"
        assert obs.command_success is True

    def test_hint_increments_hints_used(self) -> None:
        env, _, _, _, _, _, hint = _make_env()
        hint.get_hint.return_value = "hint"
        env.reset()
        obs1 = env.step(AwsRlAction(command="aws help --task-hint"))
        assert obs1.hints_used == 1
        obs2 = env.step(AwsRlAction(command="aws help --task-hint"))
        assert obs2.hints_used == 2

    def test_hint_not_achieved(self) -> None:
        env, _, _, _, _, _, hint = _make_env()
        hint.get_hint.return_value = "hint"
        env.reset()
        obs = env.step(AwsRlAction(command="aws help --task-hint"))
        assert not obs.task_achieved
        assert obs.done is False
        assert obs.reward == 0.0

    def test_hint_does_not_call_backend(self) -> None:
        env, backend, _, _, _, _, hint = _make_env()
        hint.get_hint.return_value = "hint"
        env.reset()
        backend.execute_command.reset_mock()
        env.step(AwsRlAction(command="aws help --task-hint"))
        backend.execute_command.assert_not_called()

    def test_hint_does_not_grade(self) -> None:
        env, _, _, grader, _, _, hint = _make_env()
        hint.get_hint.return_value = "hint"
        env.reset()
        env.step(AwsRlAction(command="aws help --task-hint"))
        grader.grade.assert_not_called()


# ===================================================================
# step() — normal AWS command execution
# ===================================================================


class TestStepExecution:
    def test_executes_command_on_backend(self) -> None:
        env, backend, *_ = _make_env()
        env.reset()
        backend.execute_command.reset_mock()
        env.step(AwsRlAction(command="aws s3 ls"))
        backend.execute_command.assert_called_once_with("aws s3 ls")

    def test_returns_stdout(self) -> None:
        env, backend, *_ = _make_env()
        backend.execute_command.return_value = (True, "bucket-list", "")
        env.reset()
        obs = env.step(AwsRlAction(command="aws s3 ls"))
        assert obs.command_output == "bucket-list"
        assert obs.command_success is True

    def test_returns_stderr_on_failure(self) -> None:
        env, backend, *_ = _make_env()
        backend.execute_command.return_value = (False, "", "access denied")
        env.reset()
        obs = env.step(AwsRlAction(command="aws s3 ls"))
        assert obs.command_success is False
        assert obs.error == "access denied"

    def test_step_count_increments(self) -> None:
        env, *_ = _make_env()
        env.reset()
        obs1 = env.step(AwsRlAction(command="aws s3 ls"))
        obs2 = env.step(AwsRlAction(command="aws s3 ls"))
        obs3 = env.step(AwsRlAction(command="aws s3 ls"))
        assert obs1.step_count == 1
        assert obs2.step_count == 2
        assert obs3.step_count == 3

    def test_strips_command_whitespace(self) -> None:
        env, backend, *_ = _make_env()
        env.reset()
        backend.execute_command.reset_mock()
        env.step(AwsRlAction(command="  aws s3 ls  "))
        backend.execute_command.assert_called_once_with("aws s3 ls")


# ===================================================================
# step() — grading
# ===================================================================


class TestStepGrading:
    def test_grades_after_execution(self) -> None:
        env, _, _, grader, *_ = _make_env()
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        grader.grade.assert_called_once()

    def test_passes_chaos_flag_to_grader(self) -> None:
        env, _, _, grader, _, chaos, _ = _make_env()
        chaos.chaos_occurred = True
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        _, kwargs = grader.grade.call_args
        assert kwargs["chaos_occurred"] is True

    def test_passes_hints_used_to_grader(self) -> None:
        env, _, _, grader, _, _, hint = _make_env()
        hint.get_hint.return_value = "h"
        env.reset()
        env.step(AwsRlAction(command="aws help --task-hint"))
        env.step(AwsRlAction(command="aws s3 ls"))
        _, kwargs = grader.grade.call_args
        assert kwargs["hints_used"] == 1

    def test_achieved_sets_done_true(self) -> None:
        env, _, _, grader, *_ = _make_env()
        grader.grade.return_value = GradeResult(
            task_achieved=True, partial_progress=1.0, reward=1.0, reason="done"
        )
        env.reset()
        obs = env.step(AwsRlAction(command="aws s3 ls"))
        assert obs.task_achieved is True
        assert obs.done is True
        assert obs.reward == 1.0

    def test_not_achieved_keeps_done_false(self) -> None:
        env, _, _, grader, *_ = _make_env()
        grader.grade.return_value = GradeResult(
            task_achieved=False, partial_progress=0.3, reward=0.2, reason="partial"
        )
        env.reset()
        obs = env.step(AwsRlAction(command="aws s3 ls"))
        assert obs.task_achieved is False
        assert obs.done is False
        assert obs.reward == 0.2

    def test_achieved_records_in_curriculum(self) -> None:
        env, _, curriculum, grader, *_ = _make_env()
        grader.grade.return_value = GradeResult(
            task_achieved=True, partial_progress=1.0, reward=1.0, reason="done"
        )
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        curriculum.record_result.assert_called_once_with(
            _DUMMY_TASK, achieved=True, reward=1.0
        )

    def test_not_achieved_does_not_record(self) -> None:
        env, _, curriculum, grader, *_ = _make_env()
        grader.grade.return_value = GradeResult(
            task_achieved=False, partial_progress=0.0, reward=0.0, reason="no"
        )
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        curriculum.record_result.assert_not_called()


# ===================================================================
# step() — chaos injection
# ===================================================================


class TestStepChaos:
    def test_chaos_injected_after_grading(self) -> None:
        env, _, curriculum, grader, _, chaos, _ = _make_env()
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        # Chaos should be called after grading
        chaos.maybe_inject.assert_called_once()

    def test_chaos_receives_probability(self) -> None:
        env, _, curriculum, _, _, chaos, _ = _make_env()
        curriculum.chaos_probability = 0.25
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        args = chaos.maybe_inject.call_args
        assert args[0][2] == 0.25  # third positional arg is probability

    def test_chaos_not_called_on_hint(self) -> None:
        env, _, _, _, _, chaos, hint = _make_env()
        hint.get_hint.return_value = "h"
        env.reset()
        env.step(AwsRlAction(command="aws help --task-hint"))
        chaos.maybe_inject.assert_not_called()

    def test_chaos_not_called_on_rejected_command(self) -> None:
        env, _, _, _, _, chaos, _ = _make_env()
        env.reset()
        env.step(AwsRlAction(command="not-aws"))
        chaos.maybe_inject.assert_not_called()


# ===================================================================
# step() without reset
# ===================================================================


class TestStepWithoutReset:
    def test_raises_without_reset(self) -> None:
        env, *_ = _make_env()
        # Don't call reset — _current_task is None
        with pytest.raises(AssertionError, match="reset"):
            env.step(AwsRlAction(command="aws s3 ls"))


# ===================================================================
# state property
# ===================================================================


class TestState:
    def test_state_has_episode_id(self) -> None:
        env, *_ = _make_env()
        env.reset(episode_id="ep-1")
        assert env.state.episode_id == "ep-1"

    def test_state_step_count_tracks(self) -> None:
        env, *_ = _make_env()
        env.reset()
        assert env.state.step_count == 0
        env.step(AwsRlAction(command="aws s3 ls"))
        assert env.state.step_count == 1
