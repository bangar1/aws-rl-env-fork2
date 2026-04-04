"""Tests for intermediate-tier tasks — verifies multi-step command sequences and grading.

Intermediate tasks require the agent to execute multiple AWS CLI commands in order.
The grader checks that each step's operation + resource has been executed successfully
via the EpisodeTracker.

Each test resets MiniStack, executes the full command sequence, and asserts the grader
returns task_achieved=True with reward=1.0.

Run inside Docker:
    docker exec aws-rl-env python -m pytest tests/test_intermediate_tasks.py -v
"""

import json

import pytest
import yaml
from pathlib import Path

from models import SuccessCriteria, Task, TaskID, TaskDifficulty, SetupCommand
from server.services.aws_backend import AwsBackend
from server.services.task_grader import TaskGrader
from server.services.episode_tracker import EpisodeTracker

TASKS_FILE = Path(__file__).resolve().parent.parent / "server" / "services" / "tasks" / "intermediate.yaml"

# Mapping of task_id -> ordered list of AWS CLI commands to complete the task
INTERMEDIATE_COMMANDS: dict[int, list[str]] = {
    11: [
        "aws s3api create-bucket --bucket data-pipeline",
        "aws s3api put-object --bucket data-pipeline --key test.txt --content-type text/plain",
    ],
    12: [
        (
            "aws dynamodb create-table --table-name orders "
            "--key-schema AttributeName=order_id,KeyType=HASH "
            "--attribute-definitions AttributeName=order_id,AttributeType=S "
            "--billing-mode PAY_PER_REQUEST"
        ),
        (
            "aws dynamodb put-item --table-name orders "
            '--item \'{"order_id":{"S":"001"},"status":{"S":"pending"}}\''
        ),
    ],
    13: [
        "aws sns create-topic --name alerts",
        "aws sqs create-queue --queue-name alert-inbox",
        (
            "aws sns subscribe --topic-arn arn:aws:sns:us-east-1:000000000000:alerts "
            "--protocol sqs "
            "--notification-endpoint arn:aws:sqs:us-east-1:000000000000:alert-inbox"
        ),
    ],
    14: [
        (
            "aws iam create-role --role-name lambda-exec-role "
            "--assume-role-policy-document "
            "'{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\","
            "\"Principal\":{\"Service\":\"lambda.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'"
        ),
        (
            "aws iam attach-role-policy --role-name lambda-exec-role "
            "--policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
        ),
    ],
    66: [
        "aws s3api create-bucket --bucket app-assets",
        (
            "aws iam create-policy --policy-name app-assets-read-policy "
            "--policy-document "
            "'{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\","
            "\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::app-assets/*\"}]}'"
        ),
    ],
    67: [
        (
            "aws dynamodb create-table --table-name user-sessions "
            "--key-schema AttributeName=session_id,KeyType=HASH "
            "--attribute-definitions AttributeName=session_id,AttributeType=S "
            "--billing-mode PAY_PER_REQUEST"
        ),
        "aws s3api create-bucket --bucket session-exports",
    ],
    68: [
        (
            "aws iam create-role --role-name data-processor-role "
            "--assume-role-policy-document "
            "'{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\","
            "\"Principal\":{\"Service\":\"lambda.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'"
        ),
        (
            "aws lambda create-function --function-name data-processor "
            "--runtime python3.12 --handler index.handler "
            "--role arn:aws:iam::000000000000:role/data-processor-role "
            "--code S3Bucket=dummy,S3Key=dummy.zip"
        ),
    ],
    69: [
        "aws sqs create-queue --queue-name order-events",
        "aws sns create-topic --name order-notifications",
        (
            "aws sns subscribe "
            "--topic-arn arn:aws:sns:us-east-1:000000000000:order-notifications "
            "--protocol sqs "
            "--notification-endpoint arn:aws:sqs:us-east-1:000000000000:order-events"
        ),
    ],
    70: [
        (
            "aws secretsmanager create-secret --name db-credentials "
            "--secret-string '{\"username\":\"admin\",\"password\":\"secret123\"}'"
        ),
        (
            "aws iam create-role --role-name secret-reader-role "
            "--assume-role-policy-document "
            "'{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\","
            "\"Principal\":{\"Service\":\"lambda.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'"
        ),
    ],
    71: [
        (
            "aws ssm put-parameter --name /app/config/db-host "
            "--type String --value db.internal.local"
        ),
        (
            "aws lambda create-function --function-name config-loader "
            "--runtime python3.12 --handler index.handler "
            "--role arn:aws:iam::000000000000:role/lambda-exec-role "
            "--code S3Bucket=dummy,S3Key=dummy.zip"
        ),
    ],
    72: [
        (
            "aws lambda create-function --function-name scheduled-task "
            "--runtime python3.12 --handler index.handler "
            "--role arn:aws:iam::000000000000:role/lambda-exec-role "
            "--code S3Bucket=dummy,S3Key=dummy.zip"
        ),
        'aws events put-rule --name every-five-minutes --schedule-expression "rate(5 minutes)"',
        (
            "aws events put-targets --rule every-five-minutes "
            "--targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:scheduled-task"
        ),
    ],
    73: [
        (
            "aws iam create-role --role-name ecs-task-role "
            "--assume-role-policy-document "
            "'{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\","
            "\"Principal\":{\"Service\":\"ecs-tasks.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'"
        ),
        (
            "aws iam attach-role-policy --role-name ecs-task-role "
            "--policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
        ),
    ],
    74: [
        (
            "aws secretsmanager create-secret --name rds-master-password "
            "--secret-string "
            "'{\"host\":\"db.local\",\"port\":\"3306\",\"username\":\"admin\",\"password\":\"secret\"}'"
        ),
        (
            "aws rds create-db-instance --db-instance-identifier app-database "
            "--engine mysql --db-instance-class db.t3.micro "
            "--master-username admin --master-user-password secret"
        ),
    ],
    75: [
        (
            "aws elbv2 create-target-group --name web-targets "
            "--protocol HTTP --port 80 --vpc-id vpc-00000001"
        ),
        (
            "aws route53 create-hosted-zone --name app.example.com "
            "--caller-reference unique-ref-75"
        ),
    ],
    76: [
        "aws cognito-idp create-user-pool --pool-name app-users",
        # second command placeholder — needs dynamic user-pool-id (see DYNAMIC_TASKS)
    ],
    77: [
        "aws efs create-file-system --creation-token app-storage",
        (
            "aws ec2 create-security-group --group-name efs-mount-sg "
            '--description "Allow NFS access for EFS mount"'
        ),
    ],
    78: [
        "aws ec2 create-volume --size 20 --availability-zone us-east-1a --volume-type gp3 "
        "--tag-specifications ResourceType=volume,Tags=[{Key=Name,Value=data-volume}]",
        # second command placeholder — needs dynamic volume-id (see DYNAMIC_TASKS)
    ],
    79: [
        (
            "aws elasticache create-cache-subnet-group "
            "--cache-subnet-group-name cache-subnets "
            '--cache-subnet-group-description "Cache subnets" '
            "--subnet-ids subnet-00000001 subnet-00000002"
        ),
        (
            "aws elasticache create-cache-cluster --cache-cluster-id session-cache "
            "--engine redis --cache-node-type cache.t3.micro --num-cache-nodes 1"
        ),
    ],
    80: [
        "aws glue create-database --database-input '{\"Name\":\"analytics-db\"}'",
        (
            "aws glue create-crawler --name raw-data-crawler "
            "--role arn:aws:iam::000000000000:role/glue-role "
            "--database-name analytics-db "
            "--targets '{\"S3Targets\":[{\"Path\":\"s3://data-bucket/raw/\"}]}'"
        ),
    ],
    81: [
        (
            "aws cloudformation create-stack --stack-name vpc-stack "
            "--template-body '{\"AWSTemplateFormatVersion\":\"2010-09-09\",\"Resources\":{}}'"
        ),
        "aws cloudformation describe-stacks --stack-name vpc-stack",
    ],
    82: [
        "aws apigatewayv2 create-api --name products-api --protocol-type HTTP",
        # second command placeholder — needs dynamic api-id (see DYNAMIC_TASKS)
    ],
    83: [
        "aws s3api create-bucket --bucket firehose-delivery",
        (
            "aws firehose create-delivery-stream --delivery-stream-name event-stream "
            "--s3-destination-configuration "
            "RoleARN=arn:aws:iam::000000000000:role/firehose-role,"
            "BucketARN=arn:aws:s3:::firehose-delivery"
        ),
    ],
    84: [
        "aws sqs create-queue --queue-name task-queue",
        # second command placeholder — needs dynamic queue-url (see DYNAMIC_TASKS)
    ],
    85: [
        (
            "aws dynamodb create-table --table-name products "
            "--key-schema AttributeName=product_id,KeyType=HASH "
            "AttributeName=category,KeyType=RANGE "
            "--attribute-definitions AttributeName=product_id,AttributeType=S "
            "AttributeName=category,AttributeType=S "
            "--billing-mode PAY_PER_REQUEST"
        ),
        (
            "aws dynamodb put-item --table-name products "
            "--item '{\"product_id\":{\"S\":\"P001\"},\"category\":{\"S\":\"electronics\"},"
            "\"name\":{\"S\":\"Wireless Mouse\"}}'"
        ),
    ],
    86: [
        (
            "aws iam create-role --role-name firehose-delivery-role "
            "--assume-role-policy-document "
            "'{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\","
            "\"Principal\":{\"Service\":\"firehose.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'"
        ),
        (
            "aws iam create-policy --policy-name s3-write-policy "
            "--policy-document "
            "'{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\","
            "\"Action\":\"s3:PutObject\",\"Resource\":\"*\"}]}'"
        ),
        (
            "aws iam attach-role-policy --role-name firehose-delivery-role "
            "--policy-arn arn:aws:iam::000000000000:policy/s3-write-policy"
        ),
    ],
}


def _resolve_dynamic_commands(
    task_id: int, outputs: list[str]
) -> list[str]:
    """Generate additional commands for tasks that need dynamic IDs from prior outputs.

    Returns extra commands to append after the static ones have run.
    """
    if task_id == 76:
        # create-user-pool-client needs the user-pool-id from create-user-pool output
        data = json.loads(outputs[0])
        pool_id = data["UserPool"]["Id"]
        return [
            f"aws cognito-idp create-user-pool-client --user-pool-id {pool_id} "
            f"--client-name web-app-client"
        ]
    if task_id == 78:
        # create-tags needs the volume-id from create-volume output
        data = json.loads(outputs[0])
        vol_id = data["VolumeId"]
        return [
            f"aws ec2 create-tags --resources {vol_id} "
            f"--tags Key=Name,Value=data-volume"
        ]
    if task_id == 82:
        # create-route needs the api-id from create-api output
        data = json.loads(outputs[0])
        api_id = data["ApiId"]
        return [
            f'aws apigatewayv2 create-route --api-id {api_id} '
            f'--route-key "GET /products-api"'
        ]
    if task_id == 84:
        # send-message needs the queue-url from create-queue output
        data = json.loads(outputs[0])
        queue_url = data["QueueUrl"]
        return [
            f"aws sqs send-message --queue-url {queue_url} "
            f'--message-body \'{{"task":"process","id":"task-queue-001"}}\''
        ]
    return []


# Tasks that have placeholder entries and need dynamic command resolution
_DYNAMIC_TASK_IDS = {76, 78, 82, 84}


@pytest.fixture(scope="module")
def backend() -> AwsBackend:
    return AwsBackend()


@pytest.fixture(scope="module")
def grader(backend: AwsBackend) -> TaskGrader:
    return TaskGrader(backend)


@pytest.fixture(scope="module")
def intermediate_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return yaml.safe_load(f)


def _build_task(entry: dict) -> Task:
    """Build a Task model from a raw YAML entry."""
    return Task(
        task_id=TaskID(entry["task_id"]),
        difficulty=TaskDifficulty.INTERMEDIATE,
        description=entry["description"],
        success_criteria=SuccessCriteria(**entry.get("success_criteria", {})),
        setup_commands=[
            SetupCommand(command=cmd) if isinstance(cmd, str) else SetupCommand(**cmd)
            for cmd in entry.get("setup_commands", [])
        ],
    )


def _execute_all_commands(
    task_id: int, backend: AwsBackend
) -> list[tuple[str, bool, str, str]]:
    """Execute static commands, resolve dynamic follow-ups, return all (cmd, ok, out, err)."""
    static_cmds = INTERMEDIATE_COMMANDS[task_id]
    results: list[tuple[str, bool, str, str]] = []

    for cmd in static_cmds:
        success, stdout, stderr = backend.execute_command(cmd)
        results.append((cmd, success, stdout, stderr))

    if task_id in _DYNAMIC_TASK_IDS:
        outputs = [r[2] for r in results]  # stdout values
        extra_cmds = _resolve_dynamic_commands(task_id, outputs)
        for cmd in extra_cmds:
            success, stdout, stderr = backend.execute_command(cmd)
            results.append((cmd, success, stdout, stderr))

    return results


def test_all_intermediate_tasks_have_commands(intermediate_tasks: list[dict]) -> None:
    """Every intermediate task in the YAML must have a corresponding test command sequence."""
    missing = [t["task_id"] for t in intermediate_tasks if t["task_id"] not in INTERMEDIATE_COMMANDS]
    assert not missing, f"No test commands mapped for task_ids: {missing}"


@pytest.mark.parametrize(
    "task_id",
    sorted(INTERMEDIATE_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(INTERMEDIATE_COMMANDS.keys())],
)
def test_intermediate_task_commands_execute(
    task_id: int,
    backend: AwsBackend,
) -> None:
    """All commands in the sequence must execute successfully against MiniStack."""
    backend.reset_environment()
    results = _execute_all_commands(task_id, backend)
    for i, (cmd, success, stdout, stderr) in enumerate(results):
        assert success, (
            f"Command {i + 1}/{len(results)} failed for task {task_id}.\n"
            f"  Command: {cmd}\n"
            f"  Stderr: {stderr}"
        )


@pytest.mark.parametrize(
    "task_id",
    sorted(INTERMEDIATE_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(INTERMEDIATE_COMMANDS.keys())],
)
def test_intermediate_task_grading(
    task_id: int,
    intermediate_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """Execute the full command sequence and verify the grader marks the task as achieved."""
    entry = next((t for t in intermediate_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in intermediate.yaml"

    backend.reset_environment()
    task = _build_task(entry)
    results = _execute_all_commands(task_id, backend)

    tracker = EpisodeTracker()
    for cmd, success, stdout, stderr in results:
        step = tracker.record_step(cmd, success, stdout, stderr)

    # Grade using the last step
    result = grader.grade(task, tracker, step)

    all_cmds = [r[0] for r in results]
    assert result.task_achieved, (
        f"Task {task_id} not achieved.\n"
        f"  Description: {entry['description']}\n"
        f"  Commands: {all_cmds}\n"
        f"  Reason: {result.reason}\n"
        f"  Reward: {result.reward}"
    )
    assert result.reward == 1.0, f"Expected reward=1.0, got {result.reward}"


@pytest.mark.parametrize(
    "task_id",
    sorted(INTERMEDIATE_COMMANDS.keys()),
    ids=[f"task_{tid}_partial" for tid in sorted(INTERMEDIATE_COMMANDS.keys())],
)
def test_intermediate_task_partial_gives_no_completion(
    task_id: int,
    intermediate_tasks: list[dict],
    backend: AwsBackend,
    grader: TaskGrader,
) -> None:
    """Executing only the first command of a multi-step task should not achieve it."""
    entry = next((t for t in intermediate_tasks if t["task_id"] == task_id), None)
    assert entry is not None

    # Check actual step count from YAML criteria
    steps = entry.get("success_criteria", {}).get("steps", [])
    if len(steps) < 2:
        pytest.skip("Single-step task — partial test not applicable")

    backend.reset_environment()
    task = _build_task(entry)

    # Execute only the first static command
    cmd = INTERMEDIATE_COMMANDS[task_id][0]
    success, stdout, stderr = backend.execute_command(cmd)
    tracker = EpisodeTracker()
    step = tracker.record_step(cmd, success, stdout, stderr)
    result = grader.grade(task, tracker, step)

    assert not result.task_achieved, (
        f"Task {task_id} should NOT be achieved with only the first command.\n"
        f"  Command: {cmd}\n"
        f"  Reason: {result.reason}"
    )
    assert result.reward < 1.0
