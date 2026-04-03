---
title: AWS RL Environment Server
emoji: 🥇
colorFrom: pink
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AWS RL Environment

An RL environment backed by a **simulated AWS cloud** powered by [MiniStack](https://github.com/Nahuel990/ministack). The agent sends AWS API calls as actions and receives API responses as observations. MiniStack runs inside the same Docker container, emulating 34 AWS services locally.

## Quick Start

```python
from aws_rl_env import AwsRlAction, AwsRlEnv

try:
    # Create environment from Docker image
    env = AwsRlEnv.from_docker_image("aws_rl_env-env:latest")

    # Reset
    result = env.reset()
    print(f"Episode: {result.observation.episode_id}")

    # Create an S3 bucket
    result = env.step(AwsRlAction(command="aws s3 mb s3://my-rl-bucket"))
    print(f"Create bucket success: {result.observation.command_success}")
    print(f"Output: {result.observation.command_output}")

    # Upload a file to the bucket
    result = env.step(AwsRlAction(command="aws s3 cp hello.txt s3://my-rl-bucket/"))
    print(f"Upload success: {result.observation.command_success}")

    # List buckets
    result = env.step(AwsRlAction(command="aws s3 ls"))
    print(f"Buckets: {result.observation.command_output}")

    # Describe EC2 instances
    result = env.step(AwsRlAction(command="aws ec2 describe-instances"))
    print(f"EC2 output: {result.observation.command_output}")

    # Check current task and resource state
    print(f"Task: {result.observation.task}")
    print(f"Task achieved: {result.observation.task_achieved}")
    print(f"Resources: {result.observation.resources}")

finally:
    env.close()
```

## Supported AWS Services

The environment supports **34 AWS services** via MiniStack:

| Category | Services |
|----------|----------|
| **Storage & DB** | S3, DynamoDB, RDS, ElastiCache, EFS |
| **Compute** | Lambda, ECS, EC2, Step Functions |
| **Messaging** | SQS, SNS, Kinesis, EventBridge, Firehose |
| **API** | API Gateway v1/v2, ALB/ELBv2 |
| **Security** | IAM, STS, Cognito, ACM, WAF v2, Secrets Manager |
| **Monitoring** | CloudWatch, CloudWatch Logs, SSM |
| **Infrastructure** | CloudFormation, Route53 |
| **Other** | SES, Athena, Glue, EMR |

## Building the Docker Image

```bash
docker build -t aws_rl_env-env:latest -f Dockerfile .
```

The Docker image bundles:
- The RL environment server (port 8000)
- MiniStack AWS emulator (port 4566)
- boto3 for AWS SDK access
- All MiniStack dependencies

## Environment Details

### Core Types

- `TaskID` — Unique task identifier (int)
- `EpisodeID` — Unique episode identifier (str)
- `StepCount` — Step counter within an episode (int)
- `AwsService` — Supported AWS services: `s3`, `ec2`, `dynamodb`, `lambda`

### Task

**Task**: Defines what the RL agent must accomplish

- `task_id` (TaskID) — Unique task identifier
- `difficulty` (TaskDifficulty) — One of: `warmup`, `beginner`, `intermediate`, `advanced`, `expert`
- `description` (str) — Human-readable task description
- `success_criteria` (dict) — Machine-readable criteria to evaluate task completion

### Action

**AwsRlAction**: An AWS CLI command to execute against MiniStack

- `command` (str) — AWS CLI command to execute, e.g. `"aws s3 ls"`, `"aws ec2 describe-instances"`

### Observation

**AwsRlObservation**: The result returned after each step

- `episode_id` (EpisodeID) — Unique identifier for the episode
- `step_count` (StepCount) — Current step count in the episode
- `command_success` (bool) — Whether the CLI command executed successfully
- `command_output` (str) — Stdout from the executed AWS CLI command
- `error` (str) — Stderr if the command failed
- `resources` (dict[AwsService, dict | list | str]) — Current resource state from MiniStack, keyed by service name
- `task` (Task | None) — The task the agent is trying to accomplish
- `task_achieved` (bool) — Whether the task has been achieved

## Architecture

```
┌─────────────────────────────────────────┐
│           Docker Container              │
│                                         │
│  ┌──────────────┐   ┌───────────────┐   │
│  │  RL Server   │   │  MiniStack    │   │
│  │  (port 8000) │──▶│  (port 4566)  │   │
│  │  FastAPI +   │   │  34 AWS       │   │
│  │  WebSocket   │   │  services     │   │
│  └──────────────┘   └───────────────┘   │
│         │                    │           │
│         │    boto3 calls     │           │
│         └────────────────────┘           │
└─────────────────────────────────────────┘
        ▲
        │ WebSocket / HTTP
        │
   RL Agent (client)
```

## Advanced Usage

### Connecting to an Existing Server

```python
from aws_rl_env import AwsRlAction, AwsRlEnv

env = AwsRlEnv(base_url="http://localhost:8000")
result = env.reset()

# Create a DynamoDB table
result = env.step(AwsRlAction(
    command="aws dynamodb create-table --table-name my-table --key-schema AttributeName=id,KeyType=HASH --attribute-definitions AttributeName=id,AttributeType=S --billing-mode PAY_PER_REQUEST"
))
print(f"Table created: {result.observation.command_success}")
print(f"Output: {result.observation.command_output}")
```

### Concurrent Sessions

```python
from aws_rl_env import AwsRlAction, AwsRlEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with AwsRlEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(AwsRlAction(
                command=f"aws s3api put-object --bucket client-{client_id} --key step-{i}.txt --body 'data from step {i}'"
            ))
        return client_id, result.observation.command_success

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

### Running Locally (without Docker)

Start MiniStack and the RL server separately:

```bash
# Terminal 1: Start MiniStack
pip install ministack
ministack  # Runs on port 4566

# Terminal 2: Start RL server
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
aws_rl_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── Dockerfile             # Container image (bundles RL server + MiniStack)
├── entrypoint.sh          # Starts MiniStack then RL server
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── client.py              # AwsRlEnv client
├── models.py              # AwsRlAction and AwsRlObservation models
├── ministack/             # MiniStack AWS emulator (bundled)
│   ├── app.py             # MiniStack ASGI application
│   ├── core/              # Routing, persistence, responses
│   └── services/          # 34 AWS service implementations
└── server/
    ├── __init__.py
    ├── aws_rl_env_environment.py  # Core RL environment (uses boto3 → MiniStack)
    └── app.py             # FastAPI application
```
