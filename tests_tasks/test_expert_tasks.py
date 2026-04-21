"""Tests for expert-tier tasks — verifies SRE incident resolution and security audit grading.

Expert tasks require setup commands to provision initial (broken/vulnerable) state,
then the agent must diagnose and fix issues via multi-step AWS CLI commands.
The grader uses state_checks as ground truth for task completion.

Each test resets MiniStack, provisions the setup state, executes the solution
command sequence, and asserts the grader returns task_achieved=True with reward=1.0.

Run inside Docker:
    docker exec -w /app/env aws-rl-env python -m pytest tests/test_expert_tasks.py -v
"""

import json
import re

import pytest
import yaml
from pathlib import Path

from models import SuccessCriteria, Task, TaskID, TaskDifficulty, SetupCommand
from server.services.simulator_strategy import SimulatorStrategy
from server.services.task_grader import TaskGrader
from server.services.episode_tracker import EpisodeTracker

TASKS_FILE = (
    Path(__file__).resolve().parent.parent
    / "server"
    / "services"
    / "tasks"
    / "expert.yaml"
)

# ---------------------------------------------------------------------------
# Solution commands for each expert task — ordered list of AWS CLI commands
# that resolve the SRE incident or pass the security audit.
# Diagnostic commands (list/describe) are included where needed to satisfy
# the services requirement in grading.
# ---------------------------------------------------------------------------

EXPERT_COMMANDS: dict[int, list[str]] = {
    # -- Task 18: SRE — Lambda missing SQS permissions + event source mapping --
    18: [
        "aws sqs get-queue-url --queue-name incoming-orders",
        (
            "aws iam attach-role-policy --role-name broken-lambda-role "
            "--policy-arn arn:aws:iam::aws:policy/AmazonSQSFullAccess"
        ),
        (
            "aws lambda create-event-source-mapping "
            "--function-name order-processor "
            "--event-source-arn arn:aws:sqs:us-east-1:000000000000:incoming-orders "
            "--batch-size 10"
        ),
    ],
    # -- Task 19: SRE — S3 versioning + lifecycle rule -------------------------
    19: [
        (
            "aws s3api put-bucket-versioning --bucket app-config-store "
            "--versioning-configuration Status=Enabled"
        ),
        (
            "aws s3api put-bucket-lifecycle-configuration --bucket app-config-store "
            "--lifecycle-configuration "
            '\'{"Rules":[{"ID":"cleanup-old-versions","Status":"Enabled",'
            '"NoncurrentVersionExpiration":{"NoncurrentDays":30},'
            '"Filter":{"Prefix":""}}]}\''
        ),
    ],
    # -- Task 20: SRE — DynamoDB throughput + SNS subscription -----------------
    20: [
        (
            "aws dynamodb update-table --table-name session-store "
            "--provisioned-throughput ReadCapacityUnits=50,WriteCapacityUnits=50"
        ),
        "aws sqs create-queue --queue-name ops-alert-inbox",
        (
            "aws sns subscribe "
            "--topic-arn arn:aws:sns:us-east-1:000000000000:ops-alerts "
            "--protocol sqs "
            "--notification-endpoint arn:aws:sqs:us-east-1:000000000000:ops-alert-inbox"
        ),
    ],
    # -- Task 21: Security — Replace overly permissive S3 bucket policy --------
    21: [
        "aws s3api get-bucket-policy --bucket public-assets",
        (
            "aws s3api put-bucket-policy --bucket public-assets "
            "--policy "
            '\'{"Version":"2012-10-17","Statement":[{"Effect":"Allow",'
            '"Principal":{"AWS":"arn:aws:iam::000000000000:role/app-role"},'
            '"Action":"s3:GetObject",'
            '"Resource":"arn:aws:s3:::public-assets/*"}]}\''
        ),
    ],
    # -- Task 22: Security — Replace overly broad IAM inline policy ------------
    22: [
        "aws iam get-role-policy --role-name app-role --policy-name app-access",
        (
            "aws iam put-role-policy --role-name app-role "
            "--policy-name app-access "
            "--policy-document "
            '\'{"Version":"2012-10-17","Statement":[{"Effect":"Allow",'
            '"Action":["dynamodb:GetItem","dynamodb:PutItem"],'
            '"Resource":"arn:aws:dynamodb:us-east-1:000000000000:table/users"}]}\''
        ),
    ],
    # -- Task 23: Security — Move plaintext password to Secrets Manager --------
    23: [
        (
            "aws secretsmanager create-secret "
            "--name data-processor/db-password "
            "--secret-string hunter2"
        ),
        (
            "aws lambda update-function-configuration "
            "--function-name data-processor "
            "--environment "
            "Variables={SECRET_ARN=arn:aws:secretsmanager:us-east-1:000000000000:secret:data-processor/db-password}"
        ),
    ],
    # -- Task 109: SRE — Lambda timeout + CloudWatch alarm ---------------------
    109: [
        (
            "aws lambda update-function-configuration "
            "--function-name payment-webhook --timeout 30"
        ),
        (
            "aws cloudwatch put-metric-alarm --alarm-name payment-webhook-errors "
            "--metric-name Errors --namespace AWS/Lambda --statistic Sum "
            "--period 60 --evaluation-periods 1 --threshold 5 "
            "--comparison-operator GreaterThanThreshold "
            "--dimensions Name=FunctionName,Value=payment-webhook"
        ),
    ],
    # -- Task 110: SRE — ECS service role policy + desired count ---------------
    110: [
        (
            "aws iam attach-role-policy --role-name ecs-service-role "
            "--policy-arn arn:aws:iam::aws:policy/AmazonECS_FullAccess"
        ),
        (
            "aws ecs update-service --cluster prod-cluster "
            "--service api-service --desired-count 3"
        ),
    ],
    # -- Task 111: SRE — Start RDS + fix security group -----------------------
    111: [
        "aws rds start-db-instance --db-instance-identifier analytics-db",
        (
            "aws ec2 create-security-group --group-name analytics-db-sg-fixed "
            '--description "Restricted MySQL access"'
        ),
        # authorize-security-group-ingress resolved dynamically (needs group-id)
        (
            "aws rds modify-db-instance --db-instance-identifier analytics-db "
            "--vpc-security-group-ids analytics-db-sg-fixed"
        ),
    ],
    # -- Task 113: SRE — SQS visibility timeout (redrive resolved dynamically) -
    113: [
        (
            "aws sqs set-queue-attributes "
            "--queue-url http://localhost:4566/000000000000/order-processing "
            "--attributes VisibilityTimeout=120"
        ),
        # RedrivePolicy resolved dynamically (JSON format issue with shorthand)
    ],
    # -- Task 114: SRE — Route53 DNS record update (zone-id from setup) --------
    114: [
        # change-resource-record-sets resolved dynamically (needs zone ID)
    ],
    # -- Task 115: SRE — ALB target group health check fix (DYNAMIC) -----------
    115: [
        # Resolved dynamically after setup — needs target group ARN
    ],
    # -- Task 116: Security — Lambda resource policy fix -----------------------
    116: [
        "aws iam list-roles",
        (
            "aws lambda remove-permission "
            "--function-name public-api-handler "
            "--statement-id open-access"
        ),
        (
            "aws lambda add-permission "
            "--function-name public-api-handler "
            "--statement-id restricted-access "
            "--action lambda:InvokeFunction "
            "--principal apigateway.amazonaws.com "
            "--source-arn arn:aws:execute-api:us-east-1:000000000000:*"
        ),
    ],
    # -- Task 117: Security — S3 encryption + deny unencrypted uploads ---------
    117: [
        (
            "aws s3api put-bucket-encryption --bucket data-lake-raw "
            "--server-side-encryption-configuration "
            '\'{"Rules":[{"ApplyServerSideEncryptionByDefault":'
            '{"SSEAlgorithm":"AES256"}}]}\''
        ),
        (
            "aws s3api put-bucket-policy --bucket data-lake-raw "
            "--policy "
            '\'{"Version":"2012-10-17","Statement":[{"Effect":"Deny",'
            '"Principal":"*","Action":"s3:PutObject",'
            '"Resource":"arn:aws:s3:::data-lake-raw/*",'
            '"Condition":{"StringNotEquals":'
            '{"s3:x-amz-server-side-encryption":"AES256"}}}]}\''
        ),
    ],
    # -- Task 118: Security — DynamoDB PITR + TTL ------------------------------
    118: [
        (
            "aws dynamodb update-continuous-backups "
            "--table-name financial-transactions "
            "--point-in-time-recovery-specification PointInTimeRecoveryEnabled=true"
        ),
        (
            "aws dynamodb update-time-to-live "
            "--table-name financial-transactions "
            "--time-to-live-specification Enabled=true,AttributeName=expiry_timestamp"
        ),
    ],
    # -- Task 119: Security — SSM SecureString + Secrets Manager ---------------
    119: [
        (
            "aws ssm put-parameter --name /app/database/password-secure "
            "--value SuperSecret123 --type SecureString"
        ),
        (
            "aws secretsmanager create-secret "
            "--name app/database-credentials "
            "--secret-string "
            '\'{"username":"admin","password":"SuperSecret123"}\''
        ),
    ],
    # -- Task 120: Security — IAM user managed + inline policy fix ------------
    120: [
        (
            "aws iam detach-user-policy --user-name deploy-bot "
            "--policy-arn arn:aws:iam::aws:policy/IAMFullAccess"
        ),
        (
            "aws iam delete-user-policy --user-name deploy-bot "
            "--policy-name admin-access"
        ),
        (
            "aws iam put-user-policy --user-name deploy-bot "
            "--policy-name deploy-only "
            "--policy-document "
            '\'{"Version":"2012-10-17","Statement":[{"Effect":"Allow",'
            '"Action":["s3:PutObject","codedeploy:*"],'
            '"Resource":"*"}]}\''
        ),
    ],
    # -- Task 121: SRE — EventBridge rule enable + Lambda target ---------------
    121: [
        "aws lambda get-function --function-name etl-runner",
        (
            "aws events put-rule --name nightly-etl-trigger "
            '--schedule-expression "cron(0 2 * * ? *)" '
            "--state ENABLED"
        ),
        (
            "aws events put-targets --rule nightly-etl-trigger "
            "--targets Id=1,Arn=arn:aws:lambda:us-east-1:000000000000:function:etl-runner"
        ),
    ],
    # -- Task 122: SRE — Firehose delivery stream prefix fix -------------------
    122: [
        "aws s3api head-bucket --bucket clickstream-archive",
        (
            "aws firehose delete-delivery-stream "
            "--delivery-stream-name clickstream-delivery"
        ),
        (
            "aws firehose create-delivery-stream "
            "--delivery-stream-name clickstream-delivery "
            "--s3-destination-configuration "
            '\'{"RoleARN":"arn:aws:iam::000000000000:role/firehose-role",'
            '"BucketARN":"arn:aws:s3:::clickstream-archive",'
            '"Prefix":"clickstream/year=!{timestamp:yyyy}/month=!{timestamp:MM}/"}\''
        ),
    ],
    # -- Task 123: SRE — SNS subscription DLQ + retention (DYNAMIC) ------------
    123: [
        "aws sqs create-queue --queue-name order-notifications-dlq",
        (
            "aws sqs set-queue-attributes "
            "--queue-url http://localhost:4566/000000000000/order-notifications-dlq "
            "--attributes MessageRetentionPeriod=1209600"
        ),
        # Dynamic: set-subscription-attributes resolved after setup
    ],
    # -- Task 124: Security — Encrypted EFS + NFS security group ---------------
    124: [
        (
            "aws efs create-file-system --creation-token shared-data-encrypted "
            "--encrypted --tags Key=Name,Value=shared-data-encrypted"
        ),
        (
            "aws ec2 create-security-group --group-name efs-mount-sg "
            '--description "NFS access for EFS"'
        ),
        # authorize-security-group-ingress resolved dynamically (needs group-id)
    ],
    # -- Task 125: SRE — Glue job script location fix --------------------------
    125: [
        (
            "aws s3api head-object --bucket glue-scripts-bucket "
            "--key scripts/daily-transform.py"
        ),
        (
            "aws glue update-job --job-name daily-transform "
            "--job-update "
            '\'{"Role":"arn:aws:iam::000000000000:role/glue-role",'
            '"Command":{"Name":"glueetl",'
            '"ScriptLocation":"s3://glue-scripts-bucket/scripts/daily-transform.py",'
            '"PythonVersion":"3"}}\''
        ),
    ],
    # -- Task 126: Security — Cognito password policy fix (pool-id dynamic) ----
    126: [
        # update-user-pool resolved dynamically (needs pool ID from setup)
    ],
    # -- Task 127: SRE — CloudFormation stack recovery -------------------------
    127: [
        "aws s3api create-bucket --bucket legacy-data-backup",
        "aws cloudformation delete-stack --stack-name legacy-infra",
        (
            "aws cloudformation create-stack --stack-name legacy-infra-v2 "
            "--template-body "
            '\'{"AWSTemplateFormatVersion":"2010-09-09","Resources":{"Table":'
            '{"Type":"AWS::DynamoDB::Table","Properties":{"TableName":"legacy-config",'
            '"AttributeDefinitions":[{"AttributeName":"id","AttributeType":"S"}],'
            '"KeySchema":[{"AttributeName":"id","KeyType":"HASH"}],'
            '"BillingMode":"PAY_PER_REQUEST"}}}}\''
        ),
    ],
}

# Tasks that need dynamic command resolution from setup state
_DYNAMIC_TASK_IDS = {111, 113, 114, 115, 123, 124, 126}

# ---------------------------------------------------------------------------
# MiniStack Compatibility — patching setup commands
# ---------------------------------------------------------------------------


def _patch_setup_command(cmd: str, state: dict[str, str]) -> str:
    """Patch setup commands for MiniStack compatibility."""
    # Replace hardcoded Route53 zone-001 with tracked zone ID
    if "zone-001" in cmd and "route53_zone_id" in state:
        cmd = cmd.replace("zone-001", state["route53_zone_id"])

    # Replace --group-name with --group-id for authorize-security-group-ingress
    if "authorize-security-group-ingress" in cmd:
        for key, val in state.items():
            if key.startswith("sg_"):
                group_name = key[3:]
                if f"--group-name {group_name}" in cmd:
                    cmd = cmd.replace(
                        f"--group-name {group_name}",
                        f"--group-id {val}",
                    )

    return cmd


def _track_state(cmd: str, stdout: str, state: dict[str, str]) -> None:
    """Track dynamic IDs from command outputs for subsequent commands."""
    try:
        data = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        return

    # Track Route53 hosted zone ID
    if "create-hosted-zone" in cmd and isinstance(data, dict):
        hz = data.get("HostedZone", {})
        zone_id = hz.get("Id", "")
        if "/" in zone_id:
            zone_id = zone_id.split("/")[-1]
        if zone_id:
            state["route53_zone_id"] = zone_id

    # Track security group IDs
    if "create-security-group" in cmd and isinstance(data, dict):
        group_id = data.get("GroupId", "")
        if group_id:
            match = re.search(r"--group-name\s+(\S+)", cmd)
            if match:
                state[f"sg_{match.group(1)}"] = group_id

    # Track Cognito user pool ID
    if "create-user-pool" in cmd and isinstance(data, dict):
        pool = data.get("UserPool", {})
        pool_id = pool.get("Id", "")
        if pool_id:
            state["cognito_pool_id"] = pool_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _execute_setup(
    task_entry: dict, backend: SimulatorStrategy
) -> tuple[list[tuple[str, bool, str, str]], dict[str, str]]:
    """Execute setup commands with patching; return results and tracked state."""
    results: list[tuple[str, bool, str, str]] = []
    state: dict[str, str] = {}

    for cmd in task_entry.get("setup_commands", []):
        cmd = _patch_setup_command(cmd, state)
        success, stdout, stderr = backend.execute_command(cmd)
        results.append((cmd, success, stdout, stderr))
        if success:
            _track_state(cmd, stdout, state)

    return results, state


def _resolve_dynamic_commands(
    task_id: int, backend: SimulatorStrategy, state: dict[str, str]
) -> list[str]:
    """Generate commands that depend on dynamic IDs from setup state."""
    if task_id == 111:
        # authorize-security-group-ingress needs group-id
        sg_id = state.get("sg_analytics-db-sg-fixed", "")
        if not sg_id:
            # Try to get it from the create output
            _, stdout, _ = backend.execute_command(
                "aws ec2 describe-security-groups --group-names analytics-db-sg-fixed"
            )
            try:
                data = json.loads(stdout)
                sg_id = data["SecurityGroups"][0]["GroupId"]
            except (json.JSONDecodeError, KeyError, IndexError):
                sg_id = ""
        return [
            f"aws ec2 authorize-security-group-ingress "
            f"--group-id {sg_id} "
            f"--protocol tcp --port 3306 --cidr 10.0.1.0/24"
        ]

    if task_id == 113:
        # RedrivePolicy needs JSON format to avoid shorthand parsing issues
        redrive = json.dumps(
            {
                "deadLetterTargetArn": "arn:aws:sqs:us-east-1:000000000000:order-processing-dlq",
                "maxReceiveCount": "5",
            }
        )
        attrs = json.dumps({"RedrivePolicy": redrive})
        return [
            f"aws sqs set-queue-attributes "
            f"--queue-url http://localhost:4566/000000000000/order-processing "
            f"--attributes '{attrs}'"
        ]

    if task_id == 114:
        # Route53 zone-id from setup
        zone_id = state.get("route53_zone_id", "zone-001")
        change_batch = json.dumps(
            {
                "Changes": [
                    {
                        "Action": "UPSERT",
                        "ResourceRecordSet": {
                            "Name": "api.example.com",
                            "Type": "A",
                            "TTL": 300,
                            "ResourceRecords": [{"Value": "10.0.1.50"}],
                        },
                    }
                ]
            }
        )
        return [
            f"aws route53 change-resource-record-sets "
            f"--hosted-zone-id {zone_id} "
            f"--change-batch '{change_batch}'"
        ]

    if task_id == 115:
        # Need target group ARN for modify-target-group
        success, stdout, _ = backend.execute_command(
            "aws elbv2 describe-target-groups --names web-targets"
        )
        try:
            data = json.loads(stdout)
            tg_arn = data["TargetGroups"][0]["TargetGroupArn"]
        except (json.JSONDecodeError, KeyError, IndexError):
            tg_arn = "unknown"
        return [
            f"aws elbv2 modify-target-group --target-group-arn {tg_arn} "
            f"--health-check-path /health --health-check-port 80 "
            f"--health-check-interval-seconds 15 --healthy-threshold-count 2"
        ]

    if task_id == 123:
        # Need subscription ARN for set-subscription-attributes
        success, stdout, _ = backend.execute_command(
            "aws sns list-subscriptions-by-topic "
            "--topic-arn arn:aws:sns:us-east-1:000000000000:order-notifications"
        )
        try:
            data = json.loads(stdout)
            sub_arn = data["Subscriptions"][0]["SubscriptionArn"]
        except (json.JSONDecodeError, KeyError, IndexError):
            sub_arn = "unknown"
        redrive = json.dumps(
            {
                "deadLetterTargetArn": "arn:aws:sqs:us-east-1:000000000000:order-notifications-dlq"
            }
        )
        return [
            f"aws sns set-subscription-attributes --subscription-arn {sub_arn} "
            f"--attribute-name RedrivePolicy "
            f"--attribute-value '{redrive}'"
        ]

    if task_id == 124:
        # authorize-security-group-ingress needs group-id
        sg_id = state.get("sg_efs-mount-sg", "")
        if not sg_id:
            _, stdout, _ = backend.execute_command(
                "aws ec2 describe-security-groups --group-names efs-mount-sg"
            )
            try:
                data = json.loads(stdout)
                sg_id = data["SecurityGroups"][0]["GroupId"]
            except (json.JSONDecodeError, KeyError, IndexError):
                sg_id = ""
        return [
            f"aws ec2 authorize-security-group-ingress "
            f"--group-id {sg_id} "
            f"--protocol tcp --port 2049 --cidr 10.0.2.0/24"
        ]

    if task_id == 126:
        # Cognito user-pool-id from setup
        pool_id = state.get("cognito_pool_id", "us-east-1_customer-auth")
        policies = json.dumps(
            {
                "PasswordPolicy": {
                    "MinimumLength": 12,
                    "RequireUppercase": True,
                    "RequireLowercase": True,
                    "RequireNumbers": True,
                    "RequireSymbols": True,
                    "TemporaryPasswordValidityDays": 1,
                }
            }
        )
        return [
            f"aws cognito-idp update-user-pool "
            f"--user-pool-id {pool_id} "
            f"--policies '{policies}'"
        ]

    return []


def _execute_all_commands(
    task_id: int, backend: SimulatorStrategy, state: dict[str, str] | None = None
) -> list[tuple[str, bool, str, str]]:
    """Execute static + dynamic solution commands, return all (cmd, ok, out, err)."""
    if state is None:
        state = {}

    static_cmds = EXPERT_COMMANDS[task_id]
    results: list[tuple[str, bool, str, str]] = []

    for cmd in static_cmds:
        success, stdout, stderr = backend.execute_command(cmd)
        results.append((cmd, success, stdout, stderr))
        # Track security group IDs from solution commands too
        if success:
            _track_state(cmd, stdout, state)

    if task_id in _DYNAMIC_TASK_IDS:
        extra_cmds = _resolve_dynamic_commands(task_id, backend, state)
        for cmd in extra_cmds:
            success, stdout, stderr = backend.execute_command(cmd)
            results.append((cmd, success, stdout, stderr))

    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend() -> SimulatorStrategy:
    return SimulatorStrategy()


@pytest.fixture(scope="module")
def grader(backend: SimulatorStrategy) -> TaskGrader:
    return TaskGrader(backend)


@pytest.fixture(scope="module")
def expert_tasks() -> list[dict]:
    with open(TASKS_FILE) as f:
        return yaml.safe_load(f)


def _build_task(entry: dict, state: dict[str, str] | None = None) -> Task:
    """Build a Task model, patching state_check commands with dynamic IDs."""
    task = Task(
        task_id=TaskID(entry["task_id"]),
        difficulty=TaskDifficulty.EXPERT,
        description=entry["description"],
        success_criteria=SuccessCriteria(**entry.get("success_criteria", {})),
        setup_commands=[
            SetupCommand(command=cmd) if isinstance(cmd, str) else SetupCommand(**cmd)
            for cmd in entry.get("setup_commands", [])
        ],
    )

    # Patch state_check commands with dynamic IDs from setup
    if state:
        for check in task.success_criteria.state_checks:
            if "route53_zone_id" in state and "zone-001" in check.command:
                check.command = check.command.replace(
                    "zone-001", state["route53_zone_id"]
                )
            if "cognito_pool_id" in state:
                pool_id = state["cognito_pool_id"]
                check.command = check.command.replace(
                    "us-east-1_customer-auth", pool_id
                )

    return task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_expert_tasks_have_commands(expert_tasks: list[dict]) -> None:
    """Every expert task in the YAML must have a corresponding test command sequence."""
    missing = [
        t["task_id"] for t in expert_tasks if t["task_id"] not in EXPERT_COMMANDS
    ]
    assert not missing, f"No test commands mapped for task_ids: {missing}"


@pytest.mark.parametrize(
    "task_id",
    sorted(EXPERT_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(EXPERT_COMMANDS.keys())],
)
def test_expert_task_setup_executes(
    task_id: int,
    expert_tasks: list[dict],
    backend: SimulatorStrategy,
) -> None:
    """All setup commands must execute successfully to provision initial state."""
    entry = next((t for t in expert_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in expert.yaml"

    backend.reset_environment()
    results, _ = _execute_setup(entry, backend)
    for i, (cmd, success, stdout, stderr) in enumerate(results):
        assert success, (
            f"Setup command {i + 1}/{len(results)} failed for task {task_id}.\n"
            f"  Command: {cmd}\n"
            f"  Stderr: {stderr}"
        )


@pytest.mark.parametrize(
    "task_id",
    sorted(EXPERT_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(EXPERT_COMMANDS.keys())],
)
def test_expert_task_commands_execute(
    task_id: int,
    expert_tasks: list[dict],
    backend: SimulatorStrategy,
) -> None:
    """All solution commands must execute successfully after setup."""
    entry = next((t for t in expert_tasks if t["task_id"] == task_id), None)
    assert entry is not None

    backend.reset_environment()
    _, state = _execute_setup(entry, backend)
    results = _execute_all_commands(task_id, backend, state)
    for i, (cmd, success, stdout, stderr) in enumerate(results):
        assert success, (
            f"Command {i + 1}/{len(results)} failed for task {task_id}.\n"
            f"  Command: {cmd}\n"
            f"  Stderr: {stderr}"
        )


@pytest.mark.parametrize(
    "task_id",
    sorted(EXPERT_COMMANDS.keys()),
    ids=[f"task_{tid}" for tid in sorted(EXPERT_COMMANDS.keys())],
)
def test_expert_task_grading(
    task_id: int,
    expert_tasks: list[dict],
    backend: SimulatorStrategy,
    grader: TaskGrader,
) -> None:
    """Execute setup + full solution and verify the grader marks the task as achieved."""
    entry = next((t for t in expert_tasks if t["task_id"] == task_id), None)
    assert entry is not None, f"task_id {task_id} not found in expert.yaml"

    backend.reset_environment()
    _, state = _execute_setup(entry, backend)
    task = _build_task(entry, state)
    results = _execute_all_commands(task_id, backend, state)

    tracker = EpisodeTracker()
    for cmd, success, stdout, stderr in results:
        step = tracker.record_step(cmd, success, stdout, stderr)

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
    sorted(EXPERT_COMMANDS.keys()),
    ids=[f"task_{tid}_setup_only" for tid in sorted(EXPERT_COMMANDS.keys())],
)
def test_expert_task_setup_only_gives_no_completion(
    task_id: int,
    expert_tasks: list[dict],
    backend: SimulatorStrategy,
    grader: TaskGrader,
) -> None:
    """Running only setup commands (no agent fix actions) should not achieve the task."""
    entry = next((t for t in expert_tasks if t["task_id"] == task_id), None)
    assert entry is not None

    backend.reset_environment()
    _, state = _execute_setup(entry, backend)
    task = _build_task(entry, state)

    # Agent does a no-op command to produce a StepRecord
    tracker = EpisodeTracker()
    success, stdout, stderr = backend.execute_command("aws sts get-caller-identity")
    step = tracker.record_step("aws sts get-caller-identity", success, stdout, stderr)

    result = grader.grade(task, tracker, step)
    assert not result.task_achieved, (
        f"Task {task_id} should NOT be achieved with only setup + no-op.\n"
        f"  Reason: {result.reason}"
    )
    assert result.reward < 1.0


@pytest.mark.parametrize(
    "task_id",
    sorted(EXPERT_COMMANDS.keys()),
    ids=[f"task_{tid}_partial" for tid in sorted(EXPERT_COMMANDS.keys())],
)
def test_expert_task_partial_gives_no_completion(
    task_id: int,
    expert_tasks: list[dict],
    backend: SimulatorStrategy,
    grader: TaskGrader,
) -> None:
    """Executing only the first solution command should not achieve a multi-step task."""
    entry = next((t for t in expert_tasks if t["task_id"] == task_id), None)
    assert entry is not None

    state_checks = entry.get("success_criteria", {}).get("state_checks", [])
    if len(state_checks) < 2:
        pytest.skip("Single state-check task — partial test not applicable")

    static_cmds = EXPERT_COMMANDS[task_id]
    if len(static_cmds) < 1:
        pytest.skip("No static commands — dynamic-only task")

    backend.reset_environment()
    _, state = _execute_setup(entry, backend)
    task = _build_task(entry, state)

    cmd = static_cmds[0]
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
