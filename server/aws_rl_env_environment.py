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

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import AwsRlAction, AwsRlObservation, EpisodeID, StepCount, Task
from server.services.aws_backend import AwsBackend
from server.services.curriculum import Curriculum

logger = logging.getLogger(__name__)


class AwsRlEnvironment(Environment[AwsRlAction, AwsRlObservation, State]):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        print("Initializing AWS RL Environment...")
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._backend = AwsBackend()
        self._curriculum = Curriculum()
        self._current_task: Task | None = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AwsRlObservation:
        self._backend.reset_environment()
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._current_task = self._curriculum.next_task()

        return AwsRlObservation(
            episode_id=EpisodeID(self._state.episode_id or ""),
            step_count=StepCount(self._state.step_count),
            command_success=True,
            command_output="Environment reset. MiniStack state wiped.",
            task=self._current_task,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: AwsRlAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AwsRlObservation:
        self._state.step_count += 1

        success, stdout, stderr = self._backend.execute_command(action.command)
        reward = 1.0 if success else -1.0

        # TODO: evaluate success_criteria to determine task_achieved
        task_achieved = False

        if self._current_task and task_achieved:
            self._curriculum.record_result(self._current_task, achieved=True)

        return AwsRlObservation(
            episode_id=EpisodeID(self._state.episode_id or ""),
            step_count=StepCount(self._state.step_count),
            command_success=success,
            command_output=stdout,
            error=stderr,
            task=self._current_task,
            task_achieved=task_achieved,
            done=False,
            reward=reward,
        )

    @property
    def state(self) -> State:
        return self._state
