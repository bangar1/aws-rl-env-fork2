"""
Data models for the Aws Rl Env Environment.
"""

from enum import Enum
from typing import NewType, Union

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
    SQS = "sqs"
    SNS = "sns"
    IAM = "iam"
    APIGATEWAY = "apigateway"
    SECRETSMANAGER = "secretsmanager"


# ---------------------------------------------------------------------------
# RL Task Definition
# ---------------------------------------------------------------------------


class TaskDifficulty(str, Enum):
    WARMUP = "warmup"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TierConfig(BaseModel):
    """Configuration for a single difficulty tier's promotion and mastery rules."""

    min_episodes: int = Field(
        ..., ge=0, description="Minimum episodes before promotion eligible"
    )
    advance_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Tier success rate to advance"
    )
    mastery_window: int = Field(
        default=10, ge=1, description="Sliding window size for success rate"
    )
    mastery_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Per-task graduation threshold"
    )
    fast_track_rate: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Success rate for early promotion after 3 episodes",
    )


class SpacedRepState(BaseModel):
    """Tracks spaced repetition schedule for a graduated task."""

    interval: int = Field(default=3, ge=1, description="Episodes until next re-test")
    last_graduated_episode: int = Field(
        default=0, ge=0, description="Episode number when task was last graduated"
    )


class SetupCommand(BaseModel):
    """A single AWS CLI command executed during environment setup before the agent acts."""

    command: str = Field(..., description="AWS CLI command to execute")
    description: str | None = Field(
        default=None,
        description="Human-readable explanation of what this command sets up",
    )
    ignore_failure: bool = Field(
        default=False,
        description="If True, continue setup even if this command fails",
    )


class ResourceExistsCheck(BaseModel):
    """Checks that a specific named resource exists in MiniStack."""

    service: AwsService = Field(
        ..., description="AWS service to verify the resource in"
    )
    name: str = Field(..., description="Exact resource name to verify")


class StepCriteria(BaseModel):
    """A single required step in a multi-step task."""

    operation: str = Field(..., description="AWS CLI operation, e.g. 'create-bucket'")
    resource: str | None = Field(
        default=None, description="Resource name the operation must target"
    )


class StateCheck(BaseModel):
    """An assertion about the environment's end-state, evaluated via AWS CLI."""

    command: str = Field(..., description="AWS CLI command to run for verification")
    output_contains: str | None = Field(
        default=None, description="Substring that must appear in stdout"
    )
    json_path: str | None = Field(
        default=None,
        description="JSON path to extract from stdout, e.g. '$.Table.Name'",
    )
    expected: int | float | str | bool | None = Field(
        default=None, description="Expected value at json_path"
    )


class SuccessCriteria(BaseModel):
    """Machine-readable criteria to evaluate task completion.

    Different tiers populate different fields:
    - Warmup: command_contains + operation
    - Beginner: command_contains + operation + resource_exists
    - Intermediate: steps
    - Advanced: services + steps
    - Expert: services + state_checks + steps (optional)
    """

    command_contains: str | None = Field(
        default=None, description="Substring the agent's command must contain"
    )
    operation: str | None = Field(
        default=None, description="AWS CLI operation the agent must invoke"
    )
    resource_exists: ResourceExistsCheck | None = Field(
        default=None, description="Resource that must exist after the agent acts"
    )
    steps: list[StepCriteria] = Field(
        default_factory=list, description="Ordered sequence of required operations"
    )
    services: list[AwsService] = Field(
        default_factory=list, description="AWS services the agent must interact with"
    )
    state_checks: list[StateCheck] = Field(
        default_factory=list,
        description="End-state assertions — source of truth for expert/SRE tasks",
    )


class Task(BaseModel):
    """Defines a task the RL agent must accomplish in the AWS environment."""

    task_id: TaskID = Field(..., ge=0, description="Unique task identifier")
    difficulty: TaskDifficulty = Field(
        default=TaskDifficulty.WARMUP, description="Task difficulty level"
    )
    description: str = Field(..., description="Human-readable task description")
    success_criteria: SuccessCriteria = Field(
        default_factory=SuccessCriteria,
        description="Machine-readable criteria to evaluate task completion",
    )
    setup_commands: list[SetupCommand] = Field(
        default_factory=list,
        description="Commands to run during reset to set up initial state (e.g. for SRE tasks)",
    )
    cost_budget: float | None = Field(
        default=None,
        description="Optional simulated cost budget in USD. Enables cost-based reward shaping.",
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
    step_count: StepCount = Field(
        ..., ge=0, description="Current step count in the episode"
    )
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
    cost_incurred: float = Field(
        default=0.0, description="Cumulative simulated cost of operations this episode"
    )
