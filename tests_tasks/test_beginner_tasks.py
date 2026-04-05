"""Tests for beginner-tier tasks — verifies resource creation and grading.

Beginner tasks require the agent to create a specific AWS resource. The grader
checks both command matching AND that the resource actually exists in MiniStack
via the ResourceVerifier.

Each test resets MiniStack, runs the correct create command, and asserts the
grader returns task_achieved=True with reward=1.0.

Run inside Docker:
    docker exec aws-rl-env python -m pytest tests/test_beginner_tasks.py -v
"""

import pytest
import yaml
from pathlib import Path

from models import SuccessCriteria, Task, TaskID, TaskDifficulty, SetupCommand
from server.services.aws_backend import AwsBackend
from server.services.task_grader import TaskGrader
from server.services.episode_tracker import EpisodeTracker

TASKS_FILE = (
    Path(__file__).resolve().parent.parent
    / "server"
    / "services"
    / "tasks"
    / "beginner.yaml"
)

# Mapping of task_id -> correct AWS CLI command to create the resource
BEGINNER_COMMANDS: dict[int, str] = {
    6: "aws s3api create-bucket --bucket my-test-bucket",
    7: (
        "aws dynamodb create-table --table-name users "
        "--key-schema AttributeName=user_id,KeyType=HASH "
        "--attribute-definitions AttributeName=user_id,AttributeType=S "
        "--billing-mode PAY_PER_REQUEST"
    ),
    8: "aws sqs create-queue --queue-name task-queue",
    9: "aws sns create-topic --name notifications",
    10: (
        "aws lambda create-function --function-name hello-world "
        "--runtime python3.12 --role arn:aws:iam::000000000000:role/lambda-role "
        "--handler index.handler --code S3Bucket=dummy,S3Key=dummy.zip"
    ),
    46: (
        "aws iam create-role --role-name lambda-exec-role "
        '--assume-role-policy-document \'{"Version":"2012-10-17",'
        '"Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},'
        '"Action":"sts:AssumeRole"}]}\''
    ),
    47: (
        "aws secretsmanager create-secret --name db-credentials "
        '--secret-string \'{"username":"admin","password":"secret123"}\''
    ),
    48: "aws ecs create-cluster --cluster-name web-cluster",
    49: (
        "aws rds create-db-instance --db-instance-identifier app-database "
        "--engine mysql --db-instance-class db.t3.micro "
        "--master-username admin --master-user-password Password123"
    ),
    50: (
        "aws elasticache create-cache-cluster --cache-cluster-id session-cache "
        "--engine redis --cache-node-type cache.t3.micro --num-cache-nodes 1"
    ),
    51: (
        "aws route53 create-hosted-zone --name example.internal "
        "--caller-reference unique-ref-123"
    ),
    52: (
        "aws elbv2 create-load-balancer --name web-alb "
        "--subnets subnet-00000001 subnet-00000002"
    ),
    53: "aws ec2 create-volume --size 20 --availability-zone us-east-1a",
    54: "aws efs create-file-system --creation-token shared-storage",
    55: "aws cognito-idp create-user-pool --pool-name app-users",
    56: (
        "aws ssm put-parameter --name /config/app/database-url "
        "--type String --value mysql://localhost:3306/mydb"
    ),
    57: 'aws events put-rule --name daily-cleanup --schedule-expression "rate(1 day)"',
    58: (
        "aws cloudformation create-stack --stack-name vpc-stack "
        '--template-body \'{"AWSTemplateFormatVersion":"2010-09-09","Resources":{}}\''
    ),
    59: "aws apigateway create-rest-api --name orders-api",
    60: "aws apigatewayv2 create-api --name payments-api --protocol-type HTTP",
    61: 'aws glue create-database --database-input \'{"Name":"analytics-db"}\'',
    62: "aws firehose create-delivery-stream --delivery-stream-name log-stream",
    63: (
        "aws iam create-policy --policy-name s3-read-policy "
        '--policy-document \'{"Version":"2012-10-17",'
        '"Statement":[{"Effect":"Allow","Action":"s3:GetObject","Resource":"*"}]}\''
    ),
    64: "aws iam create-user --user-name deploy-bot",
    65: (
        "aws lambda create-function --function-name data-processor "
        "--runtime python3.12 --handler index.handler "
        "--role arn:aws:iam::000000000000:role/lambda-exec-role "
        "--code S3Bucket=dummy,S3Key=dummy.zip"
    ),
}


@pytest.fixture(scope="module")
def backend() -> AwsBackend:
    return AwsBackend()


@pytest.fixture(scope="module")
def grader(backend: AwsBackend) -> TaskGrader:
    return TaskGrader(backend)


@pytest.fixture(scope="module")
def beginner_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return yaml.safe_load(f)


def _build_task(entry: dict) -> Task:
    """Build a Task model from a raw YAML entry."""
    return Task(
        task_id=TaskID(entry["task_id"]),
        difficulty=TaskDifficulty.BEGINNER,
        description=entry["description"],
        success_criteria=SuccessCriteria(**entry.get("success_criteria", {})),
        setup_commands=[
            SetupCommand(command=cmd) if isinstance(cmd, str) else SetupCommand(**cmd)
            for cmd in entry.get("setup_commands", [])
        ],
    )


def test_all_beginner_tasks_have_commands(beginner_tasks: list[dict]) -> None:
    """Every beginner task in the YAML must have a corresponding test command."""
    missing = [
        t["task_id"] for t in beginner_tasks if t["task_id"] not in BEGINNER_COMMANDS
    ]
    assert not missing, f"No test command mapped for task_ids: {missing}"


@pytest.mark.parametrize(
    "task_id",
    sorted(BEGINNER_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(BEGINNER_COMMANDS.keys())],
)
def test_beginner_task_command_executes(
    task_id: int,
    backend: AwsBackend,
) -> None:
    """The create command must execute successfully against MiniStack."""
    backend.reset_environment()
    cmd = BEGINNER_COMMANDS[task_id]
    success, stdout, stderr = backend.execute_command(cmd)
    assert success, (
        f"Command failed for task {task_id}.\n  Command: {cmd}\n  Stderr: {stderr}"
    )


@pytest.mark.parametrize(
    "task_id",
    sorted(BEGINNER_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(BEGINNER_COMMANDS.keys())],
)
def test_beginner_task_grading(
    task_id: int,
    beginner_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """Create the resource and verify the grader marks the task as achieved."""
    entry = next((t for t in beginner_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in beginner.yaml"

    # Reset MiniStack for a clean slate
    backend.reset_environment()

    task = _build_task(entry)
    cmd = BEGINNER_COMMANDS[task_id]

    # Execute the create command
    success, stdout, stderr = backend.execute_command(cmd)
    assert success, (
        f"Command failed for task {task_id}.\n  Command: {cmd}\n  Stderr: {stderr}"
    )

    # Grade the step
    tracker = EpisodeTracker()
    step = tracker.record_step(cmd, success, stdout, stderr)
    result = grader.grade(task, tracker, step)

    assert result.task_achieved, (
        f"Task {task_id} not achieved.\n"
        f"  Description: {entry['description']}\n"
        f"  Command: {cmd}\n"
        f"  Reason: {result.reason}\n"
        f"  Reward: {result.reward}"
    )
    assert result.reward == 1.0, f"Expected reward=1.0, got {result.reward}"


@pytest.mark.parametrize(
    "task_id",
    sorted(BEGINNER_COMMANDS.keys()),
    ids=[f"task_{tid}_wrong_cmd" for tid in sorted(BEGINNER_COMMANDS.keys())],
)
def test_beginner_task_rejects_wrong_command(
    task_id: int,
    beginner_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """A wrong command should not achieve a beginner task."""
    entry = next((t for t in beginner_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in beginner.yaml"

    backend.reset_environment()
    task = _build_task(entry)

    # Use a deliberately wrong command (list instead of create)
    wrong_cmd = "aws sts get-caller-identity"
    success, stdout, stderr = backend.execute_command(wrong_cmd)
    tracker = EpisodeTracker()
    step = tracker.record_step(wrong_cmd, success, stdout, stderr)
    result = grader.grade(task, tracker, step)

    assert not result.task_achieved, (
        f"Task {task_id} should NOT be achieved with wrong command '{wrong_cmd}'"
    )
    assert result.reward < 1.0
