"""Tests for EpisodeContext + regression tests for the forced-task review.

Covers:
  * EpisodeContext is frozen and derives tier / chaos_probability from the task.
  * `reset(task=<expert>)` reports tier="expert" (reviewer's exact repro).
  * Expert forced-task runs chaos with p=0.3, not 0.0 (reviewer's chaos bug).
  * Trainer-driven episodes do NOT mutate the local curriculum's record.
  * Local-mode episodes DO mutate the local curriculum's record.
  * `reset(task=<dict>)` coerces the dict back into a Task (wire format).

Run:
    python -m pytest tests/test_episode_context.py -v
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from models import (
    AwsRlAction,
    SuccessCriteria,
    Task,
    TaskDifficulty,
    TaskID,
)
from server.services.curriculum import TIER_CONFIGS
from server.services.episode_context import EpisodeContext
from server.services.task_grader import GradeResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_task(difficulty: TaskDifficulty, task_id: int = 1) -> Task:
    return Task(
        task_id=TaskID(task_id),
        difficulty=difficulty,
        description=f"{difficulty.value} task",
        success_criteria=SuccessCriteria(),
    )


_WARMUP = _make_task(TaskDifficulty.WARMUP, task_id=1)
_EXPERT = _make_task(TaskDifficulty.EXPERT, task_id=18)


def _make_env():
    """Build an AwsRlEnvironment with all heavy dependencies mocked."""
    with (
        patch("server.aws_rl_env_environment.SimulatorStrategy") as MockBackend,
        patch("server.aws_rl_env_environment.Curriculum") as MockCurriculum,
        patch("server.aws_rl_env_environment.TaskGrader") as MockGrader,
        patch("server.aws_rl_env_environment.ChaosEngine") as MockChaos,
        patch("server.aws_rl_env_environment.HintProvider"),
    ):
        from server.aws_rl_env_environment import AwsRlEnvironment

        env = AwsRlEnvironment()
        backend = MockBackend.return_value
        curriculum = MockCurriculum.return_value
        grader = MockGrader.return_value
        chaos = MockChaos.return_value

        curriculum.next_task.return_value = _WARMUP
        curriculum.current_difficulty = TaskDifficulty.WARMUP
        curriculum.chaos_probability = 0.0
        backend.execute_command.return_value = (True, "ok", "")
        backend.get_infra_state.return_value = {}
        chaos.chaos_occurred = False
        grader.grade.return_value = GradeResult(
            task_achieved=False, partial_progress=0.0, reward=0.0, reason="x"
        )

        return env, backend, curriculum, grader, chaos


# ===========================================================================
# EpisodeContext (unit)
# ===========================================================================


class TestEpisodeContextDataclass:
    def test_frozen(self) -> None:
        ctx = EpisodeContext.for_external(task=_WARMUP)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.task = _EXPERT  # type: ignore[misc]

    def test_tier_is_derived_from_task(self) -> None:
        assert EpisodeContext.for_external(_WARMUP).tier == TaskDifficulty.WARMUP
        assert EpisodeContext.for_external(_EXPERT).tier == TaskDifficulty.EXPERT

    def test_chaos_probability_matches_tier_config(self) -> None:
        expert_p = TIER_CONFIGS[TaskDifficulty.EXPERT].chaos_probability
        warmup_p = TIER_CONFIGS[TaskDifficulty.WARMUP].chaos_probability
        assert EpisodeContext.for_external(_EXPERT).chaos_probability == expert_p
        assert EpisodeContext.for_external(_WARMUP).chaos_probability == warmup_p
        # Sanity: expert must actually have nonzero chaos for the forced-task
        # bug to be visible. If this ever becomes 0.0 the regression test
        # below must be updated.
        assert expert_p > 0.0

    def test_for_external_has_no_recorder(self) -> None:
        assert EpisodeContext.for_external(_WARMUP).record_result is None

    def test_for_local_binds_curriculum_recorder(self) -> None:
        curriculum = MagicMock()
        ctx = EpisodeContext.for_local(task=_WARMUP, curriculum=curriculum)
        assert ctx.record_result is curriculum.record_result


# ===========================================================================
# Regression: the exact bugs the reviewer found
# ===========================================================================


class TestForcedTaskReportsCorrectTier:
    """state.current_tier must be 'expert' when reset(task=<expert>) is used."""

    def test_reports_expert_tier(self) -> None:
        env, *_ = _make_env()
        env.reset(task=_EXPERT)
        assert env.state.current_tier == "expert"

    def test_reports_task_tier_not_curriculum_cursor(self) -> None:
        env, _backend, curriculum, *_ = _make_env()
        # Curriculum still thinks it's warmup — irrelevant for forced task.
        curriculum.current_difficulty = TaskDifficulty.WARMUP
        env.reset(task=_EXPERT)
        assert env.state.current_tier == "expert"

    def test_local_mode_falls_back_to_curriculum(self) -> None:
        env, _backend, curriculum, *_ = _make_env()
        curriculum.current_difficulty = TaskDifficulty.INTERMEDIATE
        # next_task returns whatever; the point is current_tier should
        # equal that task's own difficulty (which is warmup from the mock).
        env.reset()
        assert env.state.current_tier == "warmup"


class TestForcedTaskUsesCorrectChaosProbability:
    """Chaos must fire at the TASK's tier probability, not the curriculum's."""

    def test_expert_forced_task_uses_expert_chaos(self) -> None:
        env, _backend, curriculum, _grader, chaos = _make_env()
        # Curriculum still advertises warmup (p=0.0). Reviewer's exact repro.
        curriculum.chaos_probability = 0.0
        env.reset(task=_EXPERT)
        env.step(AwsRlAction(command="aws s3 ls"))
        expert_p = TIER_CONFIGS[TaskDifficulty.EXPERT].chaos_probability
        chaos.maybe_inject.assert_called_once()
        _task, _tracker, p = chaos.maybe_inject.call_args.args
        assert p == expert_p
        assert p != 0.0, "Regression: expert task ran with p=0.0"

    def test_local_mode_uses_curriculum_chaos(self) -> None:
        env, _backend, curriculum, _grader, chaos = _make_env()
        # In local mode the task returned by next_task is warmup, so p=warmup's.
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        warmup_p = TIER_CONFIGS[TaskDifficulty.WARMUP].chaos_probability
        _task, _tracker, p = chaos.maybe_inject.call_args.args
        assert p == warmup_p


class TestForcedTaskDoesNotRecordLocally:
    """Local curriculum.record_result must not fire for trainer-driven episodes."""

    def test_trainer_mode_skips_record_result(self) -> None:
        env, _backend, curriculum, grader, _chaos = _make_env()
        grader.grade.return_value = GradeResult(
            task_achieved=True, partial_progress=1.0, reward=1.0, reason="done"
        )
        env.reset(task=_EXPERT)
        env.step(AwsRlAction(command="aws s3 ls"))
        curriculum.record_result.assert_not_called()

    def test_local_mode_records_on_achievement(self) -> None:
        env, _backend, curriculum, grader, _chaos = _make_env()
        grader.grade.return_value = GradeResult(
            task_achieved=True, partial_progress=1.0, reward=1.0, reason="done"
        )
        env.reset()
        env.step(AwsRlAction(command="aws s3 ls"))
        curriculum.record_result.assert_called_once()

    def test_trainer_mode_skips_record_even_across_multiple_achievements(
        self,
    ) -> None:
        env, _backend, curriculum, grader, _chaos = _make_env()
        grader.grade.return_value = GradeResult(
            task_achieved=True, partial_progress=1.0, reward=1.0, reason="done"
        )
        env.reset(task=_EXPERT)
        env.step(AwsRlAction(command="aws s3 ls"))
        env.reset(task=_EXPERT)
        env.step(AwsRlAction(command="aws s3 ls"))
        curriculum.record_result.assert_not_called()


class TestResetAcceptsTaskDict:
    """The client sends Task.model_dump() over the wire — server must coerce."""

    def test_dict_coerces_to_task(self) -> None:
        env, *_ = _make_env()
        env.reset(task=_EXPERT.model_dump())
        assert env.state.current_task is not None
        assert env.state.current_task.task_id == _EXPERT.task_id
        assert env.state.current_tier == "expert"

    def test_task_object_passed_through(self) -> None:
        env, *_ = _make_env()
        env.reset(task=_EXPERT)
        # Same task reference survives the reset
        assert env.state.current_task is _EXPERT


# ===========================================================================
# Curriculum was cleaned up — get_task_by_id is gone
# ===========================================================================


class TestCurriculumIsNoLongerPartOfTrainerControlPlane:
    def test_get_task_by_id_removed(self) -> None:
        from server.services.curriculum import Curriculum

        assert not hasattr(Curriculum, "get_task_by_id"), (
            "get_task_by_id should be removed — trainer now passes the full Task"
        )
