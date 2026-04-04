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

A **Gymnasium-style RL environment** for training LLM agents on real-world AWS cloud operations. The agent sends AWS CLI commands as actions, receives structured observations, and progresses through a **curriculum of 21 tasks** across 5 difficulty tiers — from basic listing to SRE incident response.

The environment runs a **vendored MiniStack emulator** (34 AWS services, in-memory, zero-cost) inside the same Docker container, so no AWS account is needed.

## Key Innovations

- **Priority-queue curriculum** — Tasks are selected by weakness, novelty, and spaced-repetition schedules instead of random or round-robin sampling
- **Spaced repetition** — Graduated tasks resurface at exponentially increasing intervals (3 -> 6 -> 12 -> ... -> 48 episodes) to prevent catastrophic forgetting
- **Anti-reward-hacking** — Grading verifies ground-truth state in MiniStack, not agent output; partial credit is capped at 0.99; monotonic progress prevents manipulation
- **SRE incident tasks** — Expert-tier tasks provision broken infrastructure, then require the agent to diagnose and fix it
- **Shaped rewards** — Dense reward signals (progress bonuses, failure penalties) in [0.0, 1.0] guide exploration without enabling gaming

## Quick Start

```python
from aws_rl_env import AwsRlAction, AwsRlEnv

with AwsRlEnv.from_docker_image("aws-rl-env:latest") as env:
    result = env.reset()
    print(f"Task: {result.observation.task.description}")

    result = env.step(AwsRlAction(command="aws s3 mb s3://my-bucket"))
    print(f"Reward: {result.reward}, Done: {result.done}")
```

Or connect to a running server:

```python
env = AwsRlEnv(base_url="http://localhost:8000")
result = env.reset()
result = env.step(AwsRlAction(command="aws s3 ls"))
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Container                       │
│                                                          │
│  ┌─────────────────────┐      ┌────────────────────┐    │
│  │  FastAPI RL Server   │      │  MiniStack         │    │
│  │  (port 8000)         │─────>│  (port 4566)       │    │
│  │                      │      │  34 AWS services    │    │
│  │  - Environment       │      │  In-memory state    │    │
│  │  - Curriculum        │      │  Reset API          │    │
│  │  - Grading Engine    │      │                     │    │
│  │  - Episode Tracker   │      │                     │    │
│  └─────────────────────┘      └────────────────────┘    │
│          ^                             ^                  │
│          | OpenEnv HTTP/WS             | AWS CLI calls    │
└──────────┼─────────────────────────────┼─────────────────┘
           |                             |
      RL Agent (client)          (internal only)
```

### Episode Lifecycle

1. **`reset()`** -- Wipes MiniStack state, selects next task from curriculum, provisions setup commands (if any), returns initial observation
2. **`step(action)`** -- Validates command (`aws` prefix only), executes against MiniStack, records in tracker, grades with shaped reward, returns observation
3. **Terminates** when `task_achieved == True` or max steps reached

---

## Core Classes

### `AwsRlEnvironment`

[server/aws_rl_env_environment.py](server/aws_rl_env_environment.py) -- Implements the OpenEnv `Environment` interface. Orchestrates all services.

| Method | Description |
|--------|-------------|
| `reset()` | Wipe infra, select task, provision setup, return initial observation |
| `step(action)` | Execute command, grade, update curriculum, return observation |

### `Curriculum`

[server/services/curriculum.py](server/services/curriculum.py) -- Priority-queue-based task selection with progressive difficulty.

Selects the next task using a **max-heap scored by**:

```
score = (
    novelty_bonus          # +100 if never attempted (explore first)
    + weakness_weight      # +50 * (1 - task_success_rate) -- worse tasks get higher priority
    + spaced_rep_bonus     # +30 if graduated task is "due" for re-test
    - recency_penalty      # -20 if attempted in last 2 episodes (ensure variety)
)
```

| Feature | Detail |
|---------|--------|
| **Per-task mastery** | Sliding-window success rate with exponential decay (0.85^i weighting) |
| **Graduation** | Task is "graduated" when success rate >= mastery_threshold in window |
| **Spaced repetition** | Graduated tasks resurface at doubling intervals (3 -> 6 -> ... -> 48 episodes) |
| **Tier progression** | Advance when tier success rate >= advance_rate after min_episodes |
| **Fast-track** | Skip min_episodes wait after 3 consecutive episodes at >= 90% success |
| **Skill profile** | `get_stats()` returns per-task success rates, weak spots, and due re-tests |

### `TaskGrader`

[server/services/task_grader.py](server/services/task_grader.py) -- Evaluates task completion using a dispatcher pattern. Rewards are always in [0.0, 1.0].

**Grading strategies by tier:**

| Tier | Strategy | How it works |
|------|----------|--------------|
| Warmup | Command match | Checks command contains service string + correct operation |
| Beginner | Resource creation | Verifies resource actually exists in MiniStack via `ResourceVerifier` |
| Intermediate | Multi-step | Tracks ordered sequence of (operation, resource) pairs |
| Advanced | Multi-step + services | All steps completed AND all required services touched |
| Expert | State checks | Runs arbitrary AWS CLI commands to assert end-state (ground truth) |

**Reward shaping:**

```
if task_achieved:       reward = 1.0
else:
    reward = partial_progress * 0.8        # base: scaled to [0.0, 0.8]
    if progress_increased: reward += 0.1   # dense signal for advancing
    if command_failed:     reward *= 0.5   # penalty for errors
    reward = clamp(reward, 0.0, 0.99)      # never 1.0 without completion
```

### `EpisodeTracker`

[server/services/episode_tracker.py](server/services/episode_tracker.py) -- Maintains per-episode step history. Parses AWS CLI commands to extract (service, operation, resource) tuples. Tracks credited operations for deduplication and monotonic progress.

### `ResourceVerifier`

[server/services/resource_verifier.py](server/services/resource_verifier.py) -- Queries MiniStack directly to verify ground-truth resource state. Service-specific checks for S3, DynamoDB, Lambda, SQS, SNS, IAM, and API Gateway. Also evaluates `StateCheck` assertions (substring match, JSON path extraction).

### `EnvironmentDesigner`

[server/services/environment_designer.py](server/services/environment_designer.py) -- Provisions initial AWS state via setup commands before the agent acts. Used by SRE/expert tasks to create broken infrastructure the agent must fix.

### `AwsBackend`

[server/services/aws_backend.py](server/services/aws_backend.py) -- Executes AWS CLI commands against MiniStack (`AWS_ENDPOINT_URL=http://localhost:4566`). Provides `reset_environment()` via MiniStack's `/_ministack/reset` endpoint.

### `AwsRlEnv` (Client)

[client.py](client.py) -- OpenEnv HTTP/WebSocket client. Wraps `reset()` and `step()` calls to the server.

---

## Data Models

[models.py](models.py) -- All Pydantic models and type aliases.

### Action & Observation

```python
class AwsRlAction(Action):
    command: str   # AWS CLI command, e.g. "aws s3 ls"

class AwsRlObservation(Observation):
    episode_id: EpisodeID
    step_count: StepCount
    command_success: bool
    command_output: str          # stdout from AWS CLI
    error: str                   # stderr if failed
    resources: dict[AwsService, dict | list | str]
    task: Task | None            # current task definition
    task_achieved: bool
    done: bool
    reward: float                # shaped reward in [0.0, 1.0]
```

### Task Definitions

```python
class Task:
    task_id: TaskID              # 0-20
    difficulty: TaskDifficulty   # warmup | beginner | intermediate | advanced | expert
    description: str             # human-readable goal
    success_criteria: SuccessCriteria
    setup_commands: list[SetupCommand]  # pre-provision for SRE tasks

class SuccessCriteria:
    command_contains: str | None           # warmup/beginner
    operation: str | None                  # warmup/beginner
    resource_exists: ResourceExistsCheck | None  # beginner
    steps: list[StepCriteria]             # intermediate/advanced/expert
    services: list[AwsService]            # advanced/expert
    state_checks: list[StateCheck]        # expert (ground truth)
```

### Curriculum Configuration

```python
class TierConfig:
    min_episodes: int         # minimum episodes before promotion
    advance_rate: float       # tier success rate threshold (0.6 - 1.0)
    mastery_window: int       # sliding window size (default: 10)
    mastery_threshold: float  # per-task graduation threshold (default: 0.7)
    fast_track_rate: float    # early promotion threshold (default: 0.9)

class SpacedRepState:
    interval: int                  # episodes until next re-test (3 -> 48)
    last_graduated_episode: int    # when last graduated
```

---

## Task Catalog (21 Tasks)

### Warmup (6 tasks) -- Simple listing operations

| ID | Description | Service |
|----|-------------|---------|
| 0 | List all S3 buckets | S3 |
| 1 | Describe EC2 instances | EC2 |
| 2 | List DynamoDB tables | DynamoDB |
| 3 | List Lambda functions | Lambda |
| 4 | List SQS queues | SQS |
| 5 | List SNS topics | SNS |

### Beginner (5 tasks) -- Single-resource creation with verification

| ID | Description | Verified Resource |
|----|-------------|-------------------|
| 6 | Create an S3 bucket | Bucket exists in MiniStack |
| 7 | Create a DynamoDB table | Table exists |
| 8 | Create an SQS queue | Queue URL resolvable |
| 9 | Create an SNS topic | Topic ARN in list |
| 10 | Create a Lambda function | Function exists |

### Intermediate (4 tasks) -- Multi-step workflows

| ID | Description | Steps |
|----|-------------|-------|
| 11 | Create S3 bucket + upload file | create-bucket, put-object |
| 12 | Create DynamoDB table + insert item | create-table, put-item |
| 13 | Create SNS topic + SQS queue + subscribe | create-topic, create-queue, subscribe |
| 14 | Create IAM role + attach policy | create-role, attach-role-policy |

### Advanced (3 tasks) -- Cross-service architectures

| ID | Description | Services | Steps |
|----|-------------|----------|-------|
| 15 | Lambda + SQS event source pipeline | Lambda, SQS, IAM | 4-5 steps |
| 16 | Serverless API (DynamoDB + Lambda + API Gateway) | DynamoDB, Lambda, API Gateway, IAM | 7 steps |
| 17 | Fan-out notification system (SNS + SQS) | SNS, SQS | 5 steps |

### Expert (3 tasks) -- SRE incident response

| ID | Description | Setup | Fix Required |
|----|-------------|-------|-------------|
| 18 | Fix Lambda missing SQS permissions | Broken role + Lambda + queue | Attach SQS policy, create event source |
| 19 | Enable S3 versioning + lifecycle | Bucket + object | Enable versioning, add lifecycle rule |
| 20 | Fix DynamoDB throttling + alerting | Under-provisioned table + SNS | Scale to 50 RCU/WCU, subscribe SQS |

Expert tasks use **state checks** (ground-truth AWS CLI assertions) to verify the fix, not just command matching.

---

## Anti-Reward-Hacking Measures

| Defense | How it works |
|---------|-------------|
| **Ground-truth verification** | Grader queries MiniStack directly -- agent cannot fake resource state |
| **Deduplication** | `EpisodeTracker.has_executed_operation()` prevents re-earning credit for repeated commands |
| **Invisible grading** | Verification commands run server-side, invisible to the agent's observations |
| **Command allowlisting** | Only commands starting with `aws` are executed; pipes and shell escape are rejected |
| **No credit for read-only** | Running a `state_check` command earns no progress; only mutating `steps` earn credit |
| **Monotonic progress** | `partial_progress` can only increase within an episode |
| **Exact resource names** | `resource_exists` checks the exact name, not just any resource of that type |
| **State checks verify final state** | Expert tasks run actual CLI commands against MiniStack at grading time |

---

## Supported AWS Services (34)

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

---

## Project Structure

```
aws-rl-env/
├── __init__.py                    # Exports: AwsRlEnv, AwsRlAction, AwsRlObservation
├── models.py                      # Pydantic data models & type aliases
├── client.py                      # AwsRlEnv OpenEnv client
├── inference.py                   # LLM agent inference script
├── server/
│   ├── app.py                     # FastAPI application + web UI endpoints
│   ├── aws_rl_env_environment.py  # Core RL environment (reset/step)
│   └── services/
│       ├── aws_backend.py         # MiniStack command executor
│       ├── task_grader.py         # Grading engine with reward shaping
│       ├── curriculum.py          # Curriculum learning manager
│       ├── episode_tracker.py     # Per-episode step history
│       ├── resource_verifier.py   # Ground-truth state verification
│       ├── environment_designer.py # Setup provisioning for SRE tasks
│       └── tasks/
│           ├── warmup.yaml        # 6 listing tasks
│           ├── beginner.yaml      # 5 creation tasks
│           ├── intermediate.yaml  # 4 multi-step tasks
│           ├── advanced.yaml      # 3 architecture tasks
│           └── expert.yaml        # 3 SRE incident tasks
├── aws_infra/                     # Vendored MiniStack emulator
│   └── aws_infra/
│       ├── app.py                 # MiniStack ASGI router
│       ├── core/                  # Routing, persistence, responses
│       └── services/              # 34 AWS service implementations
├── Dockerfile                     # Multi-stage build (server + MiniStack)
├── Makefile                       # Dev tasks: run, format, lint, docker-*
├── openenv.yaml                   # OpenEnv manifest
└── pyproject.toml                 # Dependencies & build config
```

---

## Running

### Docker (recommended)

```bash
make docker-build          # Build image
make docker-run            # Run on port 8000
make docker-run-detach     # Run in background
make docker-health         # Health check
```

### Local (without Docker)

```bash
# Terminal 1: Start MiniStack
pip install ministack
ministack                  # port 4566

# Terminal 2: Start RL server
export AWS_ENDPOINT_URL=http://localhost:4566
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
uv run uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Or use the combined Makefile target:

```bash
make run                   # Starts MiniStack + server
```

### OpenEnv Deployment

```bash
make openenv-validate      # Validate config
make openenv-build         # Build environment
make openenv-push          # Push to HuggingFace Spaces
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MINISTACK_URL` | `http://localhost:4566` | MiniStack endpoint |
| `AWS_ACCESS_KEY_ID` | `test` | AWS credentials (any value works) |
| `AWS_SECRET_ACCESS_KEY` | `test` | AWS credentials (any value works) |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `MAX_STEPS` | `15` | Max steps per episode |
| `API_BASE_URL` | -- | LLM API endpoint (for inference.py) |
| `MODEL_NAME` | -- | LLM model name (for inference.py) |
| `HF_TOKEN` | -- | HuggingFace token (for inference.py) |
| `TEMPERATURE` | `0.7` | LLM sampling temperature |

---

## Curriculum Stats API

The curriculum exposes detailed training progress:

```python
curriculum.get_stats()
# {
#   "episode_count": 42,
#   "tier": "intermediate",
#   "tier_episodes": 12,
#   "tier_success_rate": 0.75,
#   "graduated_tasks": [0, 2, 4],
#   "weak_spots": [11, 12],
#   "skill_profile": {0: 0.95, 1: 0.8, ...},
#   "spaced_rep_due": [0, 2],
#   "avg_reward_last_10": 0.65
# }
```
