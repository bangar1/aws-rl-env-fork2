"""
Data models for the Aws Rl Env Environment.
"""

from enum import Enum
from typing import Any, NewType, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------

TaskID = NewType("TaskID", int)
EpisodeID = NewType("EpisodeID", str)
StepCount = NewType("StepCount", int)


class AwsService(str, Enum):
    S3 = "s3"
    EC2 = "ec2"
    DYNAMODB = "dynamodb"
    LAMBDA = "lambda"


# ---------------------------------------------------------------------------
# RL Task Definition
# ---------------------------------------------------------------------------


class TaskDifficulty(Enum):
    WARMUP = "warmup"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Task(BaseModel):
    """Defines a task the RL agent must accomplish in the AWS environment."""

    task_id: TaskID = Field(..., ge=0, description="Unique task identifier")
    difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.WARMUP, description="Task difficulty level"
    )
    description: str = Field(..., description="Human-readable task description")
    success_criteria: dict[str, Any] = Field(
        default_factory=dict,
        description="Machine-readable criteria to evaluate task completion",
    )


# ---------------------------------------------------------------------------
# Action & Observation
# ---------------------------------------------------------------------------


class AwsRlAction(Action):
    """Action for the Aws Rl Env environment — an AWS CLI command to execute against MiniStack."""

    command: str = Field(
        ...,
        description="AWS CLI command to execute, e.g. 'aws s3 ls', 'aws ec2 describe-instances'",
    )


class AwsRlObservation(Observation):
    """Observation returned after each step in the AWS RL environment."""

    episode_id: EpisodeID = Field(..., description="Unique identifier for the episode")
    step_count: StepCount = Field(..., ge=0, description="Current step count in the episode")
    command_success: bool = Field(
        ..., description="Whether the CLI command executed successfully"
    )
    command_output: str = Field(
        default="", description="Stdout from the executed AWS CLI command"
    )
    error: str = Field(default="", description="Stderr if the command failed")
    resources: dict[AwsService, Union[dict, list, str]] = Field(
        default_factory=dict,
        description="Current resource state from MiniStack, keyed by service name",
    )
    task: Task | None = Field(
        default=None, description="The task the agent is trying to accomplish"
    )
    task_achieved: bool = Field(
        default=False, description="Whether the task has been achieved"
    )
