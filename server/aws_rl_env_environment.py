# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Aws Rl Env Environment Implementation.

An RL environment backed by a simulated AWS cloud powered by MiniStack.
The agent sends AWS CLI commands as actions and receives CLI output plus
the current resource state as observations.
"""

import logging

from typing import Any, Callable, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from models import (
    AwsRlAction,
    AwsRlObservation,
    AwsRlState,
    EpisodeID,
    StepCount,
    Task,
    TaskID,
    TaskInfo,
    TrackerState,
)
from server.services.aws_backend import AwsBackend
from server.services.chaos_engine import ChaosEngine
from server.services.curriculum import Curriculum
from server.services.environment_designer import EnvironmentDesigner
from server.services.episode_tracker import EpisodeTracker
from server.services.hint_provider import HintProvider, MAX_HINT_LEVEL
from server.services.task_grader import TaskGrader

logger = logging.getLogger(__name__)


class AwsRlEnvironment(Environment[AwsRlAction, AwsRlObservation, AwsRlState]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, aws_infra_url: Optional[str] = None) -> None:
        print("Initializing AWS RL Environment...")
        self._state = AwsRlState(episode_id=str(uuid4()), step_count=0)
        self._backend = (
            AwsBackend(aws_infra_url=aws_infra_url)
            if aws_infra_url is not None
            else AwsBackend()
        )
        self._curriculum = Curriculum()
        self._grader = TaskGrader(self._backend)
        self._designer = EnvironmentDesigner(self._backend)
        self._tracker = EpisodeTracker()
        self._chaos_engine = ChaosEngine(self._backend)
        self._hint_provider = HintProvider()
        self._current_task: Task | None = None
        self._pool_release: Optional[Callable[[], None]] = None

    def _sync_state(self) -> None:
        """Sync internal state to the AwsRlState object."""
        self._state.current_task = self._current_task
        self._state.tracker = TrackerState(
            step_count=self._tracker.step_count,
            hints_used=self._tracker.hints_used,
            progress=self._tracker.previous_progress,
            commands_executed=[s.command for s in self._tracker.command_history],
            credited_operations=[
                f"{op}:{res}" for op, res in self._tracker._credited_operations
            ],
        )
        self._state.chaos_occurred = self._chaos_engine.chaos_occurred
        self._state.current_tier = self._curriculum.current_difficulty.value
        self._state.infra_state = self._backend.get_infra_state()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[TaskID] = None,
        **kwargs: Any,
    ) -> AwsRlObservation:
        self._backend.reset_environment()
        self._state = AwsRlState(episode_id=episode_id or str(uuid4()), step_count=0)
        self._tracker.reset()
        self._chaos_engine.reset()
        if task_id is not None:
            self._current_task = self._curriculum.get_task_by_id(task_id)
        else:
            self._current_task = self._curriculum.next_task()

        self._designer.apply(self._current_task)
        self._sync_state()

        return AwsRlObservation(
            episode_id=EpisodeID(self._state.episode_id or ""),
            step_count=StepCount(self._state.step_count),
            command_success=True,
            command_output="Environment reset. Infra state wiped.",
            task=TaskInfo.from_task(self._current_task) if self._current_task else None,
            done=False,
            reward=0.0,
        )

    def _intercept_command(self, command: str) -> AwsRlObservation | None:
        """Handle anti-hack validation, hint requests, and help commands.

        Returns an observation if the command was intercepted, None otherwise.
        """
        if not command.startswith("aws "):
            return AwsRlObservation(
                episode_id=EpisodeID(self._state.episode_id or ""),
                step_count=StepCount(self._state.step_count),
                command_success=False,
                command_output="",
                error="Only AWS CLI commands (starting with 'aws') are allowed.",
                task=TaskInfo.from_task(self._current_task)
                if self._current_task
                else None,
                task_achieved=False,
                done=False,
                reward=0.0,
            )

        if command == "aws help --task-hint":
            hint_level = self._tracker.record_hint()
            clamped_level = min(hint_level, MAX_HINT_LEVEL)
            assert self._current_task is not None
            hint_text = self._hint_provider.get_hint(self._current_task, clamped_level)
            return AwsRlObservation(
                episode_id=EpisodeID(self._state.episode_id or ""),
                step_count=StepCount(self._state.step_count),
                command_success=True,
                command_output=hint_text,
                task=TaskInfo.from_task(self._current_task)
                if self._current_task
                else None,
                task_achieved=False,
                done=False,
                reward=0.0,
                hints_used=self._tracker.hints_used,
                hint_text=hint_text,
            )

        parts = command.split()
        if len(parts) == 3 and parts[0] == "aws":
            service_name = None
            if parts[2] == "help":
                service_name = parts[1]
            elif parts[1] == "help":
                service_name = parts[2]

            if service_name is not None:
                svc_success, help_text = self._backend.get_service_help(service_name)
                return AwsRlObservation(
                    episode_id=EpisodeID(self._state.episode_id or ""),
                    step_count=StepCount(self._state.step_count),
                    command_success=svc_success,
                    command_output=help_text if svc_success else "",
                    error="" if svc_success else help_text,
                    task=TaskInfo.from_task(self._current_task)
                    if self._current_task
                    else None,
                    task_achieved=False,
                    done=False,
                    reward=0.0,
                )

        return None

    def step(
        self,
        action: AwsRlAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AwsRlObservation:
        assert self._current_task is not None, "Call reset() before step()"
        self._state.step_count += 1

        command = action.command.strip()
        intercepted = self._intercept_command(command)
        if intercepted is not None:
            return intercepted

        success, stdout, stderr = self._backend.execute_command(command)

        # Record in tracker
        latest_step = self._tracker.record_step(command, success, stdout, stderr)

        # Grade the task (pass cumulative chaos flag and hint count)
        grade_result = self._grader.grade(
            self._current_task,
            self._tracker,
            latest_step,
            chaos_occurred=self._chaos_engine.chaos_occurred,
            hints_used=self._tracker.hints_used,
        )
        task_achieved = grade_result.task_achieved
        reward = grade_result.reward

        if task_achieved:
            self._curriculum.record_result(
                self._current_task, achieved=True, reward=reward
            )

        # Inject chaos AFTER grading — disrupts state for future steps
        self._chaos_engine.maybe_inject(
            self._current_task,
            self._tracker,
            self._curriculum.chaos_probability,
        )

        self._sync_state()

        return AwsRlObservation(
            episode_id=EpisodeID(self._state.episode_id or ""),
            step_count=StepCount(self._state.step_count),
            command_success=success,
            command_output=stdout,
            error=stderr,
            task=TaskInfo.from_task(self._current_task) if self._current_task else None,
            task_achieved=task_achieved,
            partial_progress=self._tracker.previous_progress,
            done=task_achieved,
            reward=reward,
            hints_used=self._tracker.hints_used,
        )

    @property
    def state(self) -> AwsRlState:
        return self._state

    def close(self) -> None:
        if self._pool_release is None:
            return
        try:
            self._backend.reset_environment()
        except Exception:
            logger.exception("Failed to scrub MiniStack state during close")
        try:
            self._pool_release()
        finally:
            self._pool_release = None
