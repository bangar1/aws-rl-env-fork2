# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Aws Rl Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import AwsRlAction, AwsRlObservation, EpisodeID, StepCount


class AwsRlEnv(EnvClient[AwsRlAction, AwsRlObservation, State]):
    """
    Client for the Aws Rl Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with AwsRlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.command_output)
        ...
        ...     result = client.step(AwsRlAction(command="aws s3 ls"))
        ...     print(result.observation.command_output)

    Example with Docker:
        >>> client = AwsRlEnv.from_docker_image("aws_rl_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AwsRlAction(command="aws s3 ls"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AwsRlAction) -> Dict:
        """Convert AwsRlAction to JSON payload for step message."""
        return {"command": action.command}

    def _parse_result(self, payload: Dict) -> StepResult[AwsRlObservation]:
        """Parse server response into StepResult[AwsRlObservation]."""
        obs_data = payload.get("observation", {})
        observation = AwsRlObservation(
            episode_id=EpisodeID(obs_data.get("episode_id", "")),
            step_count=StepCount(obs_data.get("step_count", 0)),
            command_success=obs_data.get("command_success", False),
            command_output=obs_data.get("command_output", ""),
            error=obs_data.get("error", ""),
            task=obs_data.get("task"),
            task_achieved=obs_data.get("task_achieved", False),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
