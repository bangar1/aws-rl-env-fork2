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
    print(f"Reset: {result.observation.success}")
    print(f"Supported services: {result.observation.metadata['supported_services']}")

    # Create an S3 bucket
    result = env.step(AwsRlAction(
        service="s3",
        operation="create_bucket",
        parameters={"Bucket": "my-rl-bucket"}
    ))
    print(f"Create bucket success: {result.observation.success}")

    # Put an object in the bucket
    result = env.step(AwsRlAction(
        service="s3",
        operation="put_object",
        parameters={"Bucket": "my-rl-bucket", "Key": "hello.txt", "Body": "world"}
    ))
    print(f"Put object success: {result.observation.success}")

    # List buckets
    result = env.step(AwsRlAction(
        service="s3",
        operation="list_buckets",
        parameters={}
    ))
    print(f"Buckets: {result.observation.response}")

    # Create an SQS queue
    result = env.step(AwsRlAction(
        service="sqs",
        operation="create_queue",
        parameters={"QueueName": "my-queue"}
    ))
    print(f"Queue URL: {result.observation.response.get('QueueUrl')}")

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

### Action

**AwsRlAction**: An AWS API call to execute
- `service` (str) - AWS service name (e.g. `"s3"`, `"sqs"`, `"dynamodb"`)
- `operation` (str) - boto3 client method (e.g. `"create_bucket"`, `"send_message"`)
- `parameters` (dict) - kwargs passed to the boto3 method

### Observation

**AwsRlObservation**: The result of the AWS API call
- `success` (bool) - Whether the call succeeded
- `response` (dict) - The AWS API response data
- `error` (str) - Error message if the call failed
- `service` (str) - Service that was called
- `operation` (str) - Operation that was executed
- `reward` (float) - +1.0 for success, -0.5 for client errors, -1.0 for invalid service/unexpected errors
- `done` (bool) - Always False (infinite episode)
- `metadata` (dict) - Step count and other info

### Reward Structure

| Outcome | Reward |
|---------|--------|
| Successful API call | +1.0 |
| AWS ClientError (e.g. bucket already exists) | -0.5 |
| Invalid parameter validation | -0.5 |
| Unsupported service | -1.0 |
| Unexpected error | -1.0 |

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
    service="dynamodb",
    operation="create_table",
    parameters={
        "TableName": "my-table",
        "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
        "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}],
        "BillingMode": "PAY_PER_REQUEST",
    }
))
print(f"Table created: {result.observation.success}")
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
                service="s3",
                operation="put_object",
                parameters={
                    "Bucket": f"client-{client_id}",
                    "Key": f"step-{i}.txt",
                    "Body": f"data from step {i}"
                }
            ))
        return client_id, result.observation.success

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
