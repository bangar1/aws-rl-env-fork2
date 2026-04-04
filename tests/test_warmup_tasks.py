"""Tests for warmup-tier tasks — verifies every task executes and grades correctly.

Each test sends the correct AWS CLI command for a warmup task against MiniStack
and asserts the grader returns task_achieved=True with reward=1.0.

Run inside Docker:
    docker exec aws-rl-env python -m pytest tests/test_warmup_tasks.py -v
"""

import pytest
import yaml
from pathlib import Path

from models import SuccessCriteria, Task, TaskID, TaskDifficulty, SetupCommand
from server.services.aws_backend import AwsBackend
from server.services.task_grader import TaskGrader
from server.services.episode_tracker import EpisodeTracker

TASKS_FILE = Path(__file__).resolve().parent.parent / "server" / "services" / "tasks" / "warmup.yaml"

# Mapping of task_id -> correct AWS CLI command
WARMUP_COMMANDS: dict[int, str] = {
    0: "aws s3 ls",
    1: "aws ec2 describe-instances",
    2: "aws dynamodb list-tables",
    3: "aws lambda list-functions",
    4: "aws sqs list-queues",
    5: "aws sns list-topics",
    27: "aws iam list-users",
    28: "aws secretsmanager list-secrets",
    29: "aws ecs list-clusters",
    30: "aws rds describe-db-instances",
    31: "aws elasticache describe-cache-clusters",
    32: "aws athena list-named-queries",
    33: "aws glue get-databases",
    34: "aws firehose list-delivery-streams",
    35: "aws emr list-clusters",
    36: "aws apigatewayv2 get-apis",
    37: "aws route53 list-hosted-zones",
    38: "aws elbv2 describe-load-balancers",
    39: "aws ec2 describe-volumes",
    40: "aws efs describe-file-systems",
    41: "aws cognito-idp list-user-pools --max-results 10",
    42: "aws ssm describe-parameters",
    43: "aws events list-rules",
    44: "aws cloudformation list-stacks",
    45: "aws apigateway get-rest-apis",
}


@pytest.fixture(scope="module")
def backend() -> AwsBackend:
    return AwsBackend()


@pytest.fixture(scope="module")
def grader(backend: AwsBackend) -> TaskGrader:
    return TaskGrader(backend)


@pytest.fixture(scope="module")
def warmup_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return yaml.safe_load(f)


def _build_task(entry: dict) -> Task:
    """Build a Task model from a raw YAML entry."""
    return Task(
        task_id=TaskID(entry["task_id"]),
        difficulty=TaskDifficulty.WARMUP,
        description=entry["description"],
        success_criteria=SuccessCriteria(**entry.get("success_criteria", {})),
        setup_commands=[
            SetupCommand(command=cmd) if isinstance(cmd, str) else SetupCommand(**cmd)
            for cmd in entry.get("setup_commands", [])
        ],
    )


def test_all_warmup_tasks_have_commands(warmup_tasks: list[dict]) -> None:
    """Every warmup task in the YAML must have a corresponding test command."""
    missing = [t["task_id"] for t in warmup_tasks if t["task_id"] not in WARMUP_COMMANDS]
    assert not missing, f"No test command mapped for task_ids: {missing}"


@pytest.mark.parametrize(
    "task_id",
    sorted(WARMUP_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(WARMUP_COMMANDS.keys())],
)
def test_warmup_task_grading(
    task_id: int,
    warmup_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """Send the correct command for a warmup task and verify it grades as achieved."""
    entry = next((t for t in warmup_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in warmup.yaml"

    task = _build_task(entry)
    cmd = WARMUP_COMMANDS[task_id]

    # Execute against MiniStack
    success, stdout, stderr = backend.execute_command(cmd)
    assert success, f"Command failed: {cmd}\nstderr: {stderr}"

    # Grade the step
    tracker = EpisodeTracker()
    step = tracker.record_step(cmd, success, stdout, stderr)
    result = grader.grade(task, tracker, step)

    assert result.task_achieved, (
        f"Task {task_id} not achieved.\n"
        f"  Command: {cmd}\n"
        f"  Reason: {result.reason}\n"
        f"  Reward: {result.reward}"
    )
    assert result.reward == 1.0, f"Expected reward=1.0, got {result.reward}"


@pytest.mark.parametrize(
    "task_id",
    sorted(WARMUP_COMMANDS.keys()),
    ids=[f"task_{tid}_wrong_cmd" for tid in sorted(WARMUP_COMMANDS.keys())],
)
def test_warmup_task_rejects_wrong_command(
    task_id: int,
    warmup_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """A wrong command should not achieve a warmup task."""
    entry = next((t for t in warmup_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in warmup.yaml"

    task = _build_task(entry)

    # Use a deliberately wrong command (different service)
    wrong_cmd = "aws sts get-caller-identity"

    success, stdout, stderr = backend.execute_command(wrong_cmd)
    tracker = EpisodeTracker()
    step = tracker.record_step(wrong_cmd, success, stdout, stderr)
    result = grader.grade(task, tracker, step)

    assert not result.task_achieved, (
        f"Task {task_id} should NOT be achieved with wrong command '{wrong_cmd}'"
    )
    assert result.reward < 1.0
