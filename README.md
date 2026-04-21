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

# AWS Cloud CLI and SRE Reinforcement Learning Environment

A **OpenEnv** RL environment** for training AI agents on real-world AWS cloud operations. The agent sends AWS CLI commands as actions, receives structured observations, and progresses through a **curriculum of 120+ tasks** across 5 difficulty tiers — from basic listing to SRE incident response and security posture auditing.

The agents interact with a **real-world AWS Shell simulator** — a vendored MiniStack emulator (34 AWS services, in-memory, zero-cost) inside the same Docker container. The response of every executed command is the same as production AWS. The grading system evaluates rewards and penalties based on the **actual AWS infrastructure state** instead of static metrics. No AWS account needed.

> **[Try the Playground](https://sizzing-aws-rl-env.hf.space/web)** | **[API Docs](https://sizzing-aws-rl-env.hf.space/docs)** | **[Hugging Face Space](https://huggingface.co/spaces/Sizzing/aws_rl_env)**


## Task Tiers (100+ Tasks)

### Warmup — 20 tasks
> List resources — single read-only commands

- Run one AWS CLI command to list or describe a resource type
- S3 buckets, EC2 instances, DynamoDB tables, Lambda functions, RDS, EBS volumes
- Graded by **command_match** — checks operation + service pair
- No setup required, no state mutations

### Beginner — 20 tasks
> Create single resources with verification

- Create an S3 bucket, DynamoDB table, SQS queue, or Lambda function
- Graded by **resource_creation** — verifies the exact resource exists in the AWS Infrastructure Simulator
- Introduces resource name validation — "my-bucket-2" won't satisfy a check for "my-bucket"
- First tier where idempotency bonus (+0.02) can be earned

### Intermediate — 20 tasks
> Multi-step workflows — create, configure, connect

- Ordered sequences: create a bucket then enable versioning, create a table then add an item
- Graded by **multi_step** — validates each step was completed in order
- Chaos injection begins at **10% probability** — resources may be silently mutated mid-episode
- Rollback penalty (-0.1) starts to matter with multi-step create/delete patterns

### Advanced — 20 tasks
> Cross-service architectures spanning multiple AWS services

- Wire Lambda to SQS, configure API Gateway with integrations, build event-driven pipelines
- Graded by **multi_step + services** — all required services must be configured
- Chaos injection escalates to **20% probability** — DynamoDB throughput, Lambda configs may change
- Hints cost more: 3 hints = only 61% of max reward (0.85³ decay)

### Expert — 20 tasks
> SRE incidents, drift detection & security posture audits

- Fix overly permissive S3 policies, replace broad IAM inline policies, repair broken infrastructure
- Graded by **state_checks** — actual CLI commands run against MiniStack at grading time
- Chaos injection at **30% probability** — maximum perturbation frequency
- **6 drift detection tasks** — correct infra is provisioned, then 2-3 random mutations applied from a pool
- Agent must audit environment, discover which resources drifted, and fix only those
- Drift is randomized per episode — prevents memorization of fix sequences

---

## Features

### 1. Curriculum & Training

Adaptive learning system that tracks mastery and selects optimal tasks.

#### Progressive Difficulty
- **What:** The environment organizes 120+ tasks across 5 tiers: Warmup, Beginner, Intermediate, Advanced, and Expert. Tasks progress from simple listing operations to complex SRE incident response and drift detection scenarios.
- **Why:** Prevents the agent from being overwhelmed by complex tasks early on. Scaffolded difficulty ensures the agent builds foundational skills before tackling multi-service architectures.
- **How:** The `CurriculumManager` maintains per-agent tier state. Promotion requires meeting a minimum episode count and success rate threshold. A fast-track mechanism allows agents scoring 90%+ on 3 consecutive episodes to skip the minimum wait.
- **Metrics:** 5 Difficulty Tiers | 120+ Total Tasks | 90% Fast-track Threshold

#### Mastery Tracking
- **What:** Each task independently tracks the agent's performance using a weighted success rate over a sliding window. Tasks "graduate" when performance exceeds the mastery threshold consistently.
- **Why:** Ensures the agent truly masters a skill before moving on. Prevents lucky single completions from being treated as mastery. Un-graduation catches skill decay.
- **How:** A `mastery_window` of 10 episodes and `mastery_threshold` of 0.7 (70% success). Minimum 3 attempts required before graduation. Recent results are weighted more heavily using exponential decay (factor 0.85). Graduated tasks can un-graduate if performance drops.
- **Metrics:** 70% Mastery Threshold | 10 Window Size | 0.85 Decay Factor

#### Spaced Repetition
- **What:** Graduated tasks don't disappear — they resurface at exponentially increasing intervals (3, 6, 12, 24, 48 episodes) for re-testing, earning a +30 priority bonus when due.
- **Why:** Prevents catastrophic forgetting. The agent must retain skills even as it learns new ones. Exponential spacing is the most efficient retention schedule, borrowed from cognitive science.
- **How:** Each task tracks a `spaced_rep_interval` starting at 3 episodes. When re-tested and passes, the interval doubles (up to 48). If it fails, the interval resets. `_is_spaced_rep_due()` checks elapsed episodes against the interval.
- **Metrics:** +30 Spaced Rep Bonus | 3→48 Interval Range | 2x Interval Growth

#### Priority Selection
- **What:** Tasks are ranked by a composite score combining novelty, weakness, spaced repetition due dates, and recency. The highest-scoring task is selected for each episode.
- **Why:** Optimizes the training curriculum by ensuring the agent explores new tasks, practices weak areas, revisits graduated skills, and maintains variety — all balanced automatically.
- **How:** `score = novelty_bonus (+100 if never attempted) + weakness_weight (+50 × (1 - success_rate)) + spaced_rep_bonus (+30 if due) - recency_penalty (-20 if attempted in last 2 episodes)`. Uses exponential decay (0.85) to emphasize recent performance.
- **Metrics:** +100 Novelty Bonus | +50 Max Weakness Weight | -20 Recency Penalty

#### Tier Progression
- **What:** Agents advance through tiers via standard promotion (minimum episodes + success rate) or fast-track (3 consecutive high-scoring episodes). Tiers gate access to increasingly complex task pools.
- **Why:** Provides structure to the learning process. Standard promotion ensures sufficient exposure; fast-track rewards agents that demonstrate immediate competence.
- **How:** Standard: complete `min_episodes` at current tier with `success_rate >= advance_rate`. Fast-track: 3 consecutive episodes at >= 90% success bypasses the minimum episode requirement. Un-promotion is not supported — agents cannot drop tiers.
- **Metrics:** 3 Fast-track Streak | 90% Fast-track Rate | 5 Total Tiers

### 2. Reward Shaping

Dense reward signals that encourage operational discipline and real progress.

```
if task_achieved:       reward = 1.0
else:
    reward = partial_progress * 0.8        # base: scaled to [0.0, 0.8]
    if progress_increased: reward += 0.1   # dense signal for advancing
    if command_failed:     reward *= 0.5   # penalty for errors
    reward = clamp(reward, 0.0, 0.99)      # never 1.0 without completion
    reward *= 0.85 ** hints_used           # hint decay
    if survived_chaos:    reward *= 1.05   # chaos survival bonus
```

#### Rollback Penalty & Idempotency Bonus
- **What:** Detects create→delete pairs on the same resource (rollbacks) and penalizes them (-0.1 each). Rewards graceful "already exists" handling (+0.02) where the agent retries idempotently.
- **Why:** First RL environment rewarding operational discipline. In production, create-then-delete cycles are wasteful. Handling "already exists" gracefully is a sign of robust automation.
- **How:** `EpisodeTracker.detect_rollbacks()` scans command history for paired create/delete operations on the same resource. Idempotency detection looks for commands that fail with "already exists" patterns (BucketAlreadyExists, ResourceInUseException, etc.) followed by successful continuation.
- **Metrics:** -0.1 Rollback Penalty | +0.02 Idempotency Bonus | Per-pair Detection

#### Shaped Reward System
- **What:** Rewards are carefully shaped: 1.0 for full completion, 0.0-0.8 for partial progress, +0.1 progress bonus for advancing, ×0.5 for failures, capped at 0.99 without completion. Chaos bonus (×1.05) and hint decay (×0.85^n) layer on top.
- **Why:** Dense reward signal prevents sparse-reward stagnation. The agent gets meaningful feedback on every step, not just at episode end. Capping at 0.99 ensures only real completion earns full credit.
- **How:** `TaskGrader` dispatches to 5 strategies by tier: `command_match` (warmup), `resource_creation` (beginner), `multi_step` (intermediate), `multi_step+services` (advanced), and `state_checks` (expert). Each returns `partial_progress` which is converted to reward with bonuses/penalties applied.
- **Metrics:** 1.0 Max Reward | 0.99 Progress Cap | ×1.05 Chaos Bonus

#### Multi-Strategy Grading
- **What:** Five distinct grading strategies, one per tier: `command_match` checks operation+service pairs, `resource_creation` verifies resources exist, `multi_step` validates ordered sequences, advanced adds service coverage, and expert runs `state_checks` against MiniStack.
- **Why:** Each tier tests fundamentally different skills. A single grading strategy would either be too lenient for beginners or miss the nuance needed for expert SRE tasks.
- **How:** `TaskGrader.grade()` dispatches based on the task's `grading_strategy` field. Each strategy returns a `GradeResult` with `partial_progress` (0.0-1.0), `completed` flag, and details. Grading is deterministic and fully automated.
- **Metrics:** 5 Grading Strategies | 100% Automated | Per-tier Selection

### 3. Resilience & Adaptability

Features that test agent robustness under unpredictable conditions.

#### Progressive Hint System
- **What:** A 3-level hint system where each level reveals progressively more detail: Level 1 names the AWS services, Level 2 describes the operations, Level 3 gives near-complete command structure. Each hint reduces the final reward by ×0.85.
- **Why:** Creates an information-reward tradeoff unique in RL. The agent learns to wean off hints over time — initially relying on them for unfamiliar tasks, then solving independently for maximum reward. From GRPO perspective, it creates a natural exploration/exploitation axis within a single episode.
- **How:** Agent issues special command `aws help --task-hint` as its action (intercepted before reaching MiniStack). Hints auto-generated from `SuccessCriteria` fields (services, steps, operations). Reward decay: `final_reward *= 0.85 ^ hints_used` — 0 hints: 1.0×, 1 hint: 0.85×, 2 hints: 0.72×, 3 hints: 0.61×. Curriculum naturally penalizes hint-dependent agents: lower rewards → slower graduation.
- **Metrics:** 3 Hint Levels | ×0.85 Decay Per Hint | ~61% Reward with 3 Hints

#### Chaos Injection Engine
- **What:** Silently mutates AWS resource state mid-episode to test agent resilience. Perturbations are scoped to services the current task uses. If the agent completes despite chaos, it earns a ×1.05 bonus.
- **Why:** Tests whether the agent can handle unexpected state changes — a critical SRE skill. Prevents brittle memorization of exact command sequences. Probability scales with tier difficulty.
- **How:** `ChaosEngine` selects perturbation templates specific to the services in use (S3 policy changes, DynamoDB throughput modifications, Lambda config alterations, etc.). Resource names are extracted from successful commands via regex. Chaos probability: 10% (Intermediate), 20% (Advanced), 30% (Expert).
- **Metrics:** ×1.05 Chaos Survival Bonus | 10-30% Probability by Tier | 5 Service Templates

#### Drift Detection Tasks
- **What:** 6 expert-tier tasks where infrastructure is provisioned correctly, then 2-3 random mutations are applied from a pool. The agent must audit, discover drifted resources, and fix only those — without knowing which drifted.
- **Why:** Randomized per episode, preventing memorization. Tests real SRE audit skills: the agent must reason about desired vs. actual state, not just follow a script.
- **How:** `DriftEngine` randomly selects 2-3 mutations from a task's `possible_drifts` pool and applies them after setup. Each task defines a `desired_state_spec` (natural language) and `state_checks` (ground truth CLI commands). Examples: S3 versioning/encryption drift, DynamoDB throughput changes, SNS subscription modifications.
- **Metrics:** 6 Drift Tasks | 2-3 Mutations Per Episode | Random Selection Per Run

### 4. Security Posture Audit

Tests *reasoning about configuration state* — the agent must READ and ANALYZE existing infrastructure, not just build things. Unlike SRE tasks (broken functionality), these have *working but insecure* infrastructure.

#### Public S3 Bucket Lockdown
- **What:** A pre-provisioned S3 bucket "public-assets" has an overly permissive bucket policy granting access to any principal (`Principal: *`). The agent must read the policy, identify the vulnerability, and replace it with a restrictive policy allowing only a specific IAM role.
- **Why:** Tests security reasoning — the infrastructure is functional but insecure. Unlike SRE tasks where things are broken, here the agent must understand what "correct" security posture looks like and make the right judgment call.
- **How:** Setup creates the bucket with a wide-open policy. State checks verify the new policy denies `Principal: *` and only allows the `app-role` principal to perform `s3:GetObject`.
- **Metrics:** S3 Target Service | Policy Attack Surface | Expert Tier

#### IAM Least Privilege
- **What:** An IAM role "app-role" has an inline policy with `Action: *` and `Resource: *` — full admin access. The agent must replace it with a least-privilege policy allowing only `dynamodb:GetItem` and `dynamodb:PutItem` on the users table.
- **Why:** IAM misconfiguration is the #1 cloud security risk. This task tests whether the agent understands permission scoping and can reason about what access an application actually needs vs. what it currently has.
- **How:** Setup creates the role with a wildcard policy. The agent must craft a replacement policy document with specific actions and resource ARN. State checks verify the policy document matches the expected least-privilege permissions.
- **Metrics:** IAM Target Service | 2 Allowed Actions | Expert Tier

#### Secrets in Lambda Environment
- **What:** A Lambda function "data-processor" has a database password stored as a plaintext environment variable (`DB_PASSWORD=hunter2`). The agent must create a secret in Secrets Manager, update the Lambda to reference the secret ARN, and remove the plaintext variable.
- **Why:** Plaintext secrets in environment variables is a critical security anti-pattern. This task combines multiple services (Lambda + Secrets Manager) and tests the agent's ability to perform a safe credential rotation without breaking the function.
- **How:** Setup creates the Lambda with the plaintext env var. The agent must: (1) create a secret in Secrets Manager, (2) add `SECRET_ARN` env var to Lambda, (3) remove `DB_PASSWORD`. State checks verify all three conditions.
- **Metrics:** 2 Services Involved | 3 Required Steps | Expert Tier

### 5. Anti-Reward-Hacking (8 Defense Layers)

8 defense layers that prevent the agent from gaming the reward system.

#### 1. Ground-Truth Verification via MiniStack
- **What:** The grader never trusts agent command output. It independently queries MiniStack (the simulated AWS backend) to verify resource state for 20+ services. Even if the agent crafts fake-looking stdout, the grader checks actual state.
- **Why:** Prevents reward hacking through output fabrication. The agent cannot game the system by producing convincing but fake CLI output — ground truth is always checked server-side.
- **How:** `ResourceVerifier` has per-service verification methods that query MiniStack directly. For expert tasks, `StateCheck` assertions run actual AWS CLI commands against MiniStack at grading time, checking either `output_contains` (substring) or `json_path` extraction with expected values.
- **Metrics:** 20+ Verified Services | 100% Server-side | 0 Agent Visibility

#### 2. Deduplication
- **What:** `EpisodeTracker.has_executed_operation()` tracks which (operation, resource) pairs have been credited. Running the same successful command twice does NOT increase `partial_progress`. Progress can only increase, never re-earn.
- **Why:** Prevents the agent from gaming the reward system by repeating the same command to accumulate credit. Each unique operation earns credit exactly once.
- **How:** `credit_operation()` records each (operation, resource) pair. Before granting credit, `is_operation_already_credited()` checks if this exact pair was already rewarded. The check is deterministic and happens at grading time.
- **Metrics:** 1x Credit Per Operation | Exact Match Type | (op, res) Tracking Granularity

#### 3. Grader Invisibility
- **What:** The verification commands run by `ResourceVerifier` are NOT returned in the observation's `command_output`. They happen server-side during grading. The agent cannot observe or mimic them.
- **Why:** If the agent could see which verification commands the grader runs, it could learn to craft fake outputs that match expected patterns. Keeping grader logic invisible forces the agent to actually perform the task.
- **How:** `ResourceVerifier` executes AWS CLI commands against MiniStack in a separate execution context. Results are consumed internally by the grading pipeline. The observation returned to the agent only contains output from the agent's own commands.
- **Metrics:** 0 Grader Cmds Exposed | Server Execution Context | 20+ Hidden Verifications

#### 4. Command Allowlisting
- **What:** Only commands starting with `aws` are executed. Any attempt to run shell commands, pipe to other tools, use redirects, or escape the sandbox is rejected with `success=False`.
- **Why:** Prevents the agent from escaping the AWS CLI sandbox. Without this, the agent could potentially execute arbitrary shell commands, access the filesystem, or interfere with the environment.
- **How:** The environment's `step()` method validates the command before execution. Commands not starting with `aws` are immediately rejected.
- **Metrics:** `aws *` Allowed Pattern | 0 Shell Access | Instant Rejection

#### 5. No Verification Reward
- **What:** If the agent runs a command that matches a `state_check` command exactly (e.g., `aws s3api get-bucket-versioning --bucket app-config-store`), it gets no progress credit. Progress is only earned through `steps` operations (mutating commands), not read-only queries.
- **Why:** Prevents the agent from gaming progress by running the same verification commands the grader uses. The agent can run read commands to understand state, but only mutation commands earn progress.
- **How:** During grading, the `TaskGrader` checks if the agent's command matches any `state_check` command. Matching commands are flagged as verification-only and excluded from credit. Only commands matching `steps` operations (create, put, update, delete) earn `partial_progress`.
- **Metrics:** 0 Credit for Reads | Mutate Rewarded Actions | Exact Match Detection

#### 6. Monotonic Progress
- **What:** `partial_progress` can only increase within an episode. It is clamped to [0.0, 0.99] — reaching 1.0 requires actual task completion. The agent cannot lose progress, but also cannot re-earn it.
- **Why:** Prevents cycling strategies where the agent creates and destroys resources repeatedly. Combined with deduplication, this ensures steady forward progress.
- **How:** In `TaskGrader`, `previous_progress` tracks the highest progress seen. New progress is always `max(previous, current)`. Reward is clamped at 0.99 for partial completion, reserving 1.0 exclusively for verified full completion.
- **Metrics:** 0.99 Max Without Completion | 1.0 Requires Full Completion | max() Progress Function

#### 7. Resource Name Validation
- **What:** For `resource_exists` checks, the verifier matches the exact resource name, not just any resource of that type. Creating "my-test-bucket-2" doesn't satisfy a check for "my-test-bucket".
- **Why:** Prevents the agent from creating arbitrarily named resources to game the verification system. Forces precise execution of the task requirements.
- **How:** `ResourceVerifier`'s per-service methods (`verify_s3_bucket`, `verify_dynamodb_table`, etc.) compare against the exact expected resource name from the task definition. Each of the 20+ supported services has its own verification logic.
- **Metrics:** Exact Name Matching | 20+ Verified Services | 0 Partial Matches

#### 8. State Checks Verify Final State
- **What:** For expert SRE tasks, `state_checks` run actual AWS CLI commands against MiniStack at grading time. The grader verifies the final infrastructure state — not the commands the agent ran.
- **Why:** The agent cannot fake the state. MiniStack is the ground truth. This decouples "what the agent did" from "what was actually achieved", making reward hacking extremely difficult.
- **How:** Each expert task defines `state_checks` with command + assertion pairs. Assertions support `output_contains` (substring match on CLI output) and `json_path + expected` (JSON extraction). The grader runs these checks against the live MiniStack state independently of the agent.
- **Metrics:** CLI Verification Method | 2 Assertion Types | Live State Source

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

WebSocket API:

```python
import websockets, json

async with websockets.connect("wss://sizzing-aws-rl-env.hf.space/ws") as ws:
    await ws.send(json.dumps({"type": "reset"}))
    obs = json.loads(await ws.recv())

    await ws.send(json.dumps({"type": "step", "data": {"command": "aws s3 ls"}}))
    obs = json.loads(await ws.recv())
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                     │
│                                                         │
│  ┌─────────────────────┐      ┌────────────────────┐    │
│  │  FastAPI RL Server  │      │  AWS Simulator     │    │
│  │  (port 8000)        │─────>│  (port 4566)       │    │
│  │                     │      │  34 AWS services   │    │
│  │  - Environment      │      │  In-memory state   │    │
│  │  - Curriculum       │      │  Reset API         │    │
│  │  - Grading Engine   │      │  (Ministack)       │    │
│  │  - Episode Tracker  │      │                    │    │
│  │  - Hint Provider    │      │                    │    │
│  └─────────────────────┘      └────────────────────┘    │
│          ^                             ^                │
│          | OpenEnv HTTP/WS             | AWS CLI calls  │
└──────────┼─────────────────────────────┼────────────────┘
           |                             |
      RL Agent (client, External)     (internal only)
```

### Episode Lifecycle

1. **`reset()`** — Wipes AWS Infracture state, selects next task from curriculum, provisions setup commands (if any), returns initial observation
2. **`step(action)`** — Validates command (`aws` prefix only), executes against MiniStack, records in tracker, grades with shaped reward, returns observation
3. **Hint request** — Agent sends `aws help --task-hint` to get a progressive hint (costs reward)
4. **Terminates** when `task_achieved == True` or max steps reached

---


## Core Classes

### `AwsRlEnvironment`

[server/aws_rl_env_environment.py](server/aws_rl_env_environment.py) — Implements the OpenEnv `Environment` interface. Orchestrates all services.

| Method | Description |
|--------|-------------|
| `reset()` | Wipe infra, select task, provision setup, return initial observation |
| `step(action)` | Execute command (or intercept hint request), grade, update curriculum, return observation |

### `Curriculum`

[server/services/curriculum.py](server/services/curriculum.py) — Priority-queue-based task selection with progressive difficulty.

Selects the next task using a **max-heap scored by**:

```
score = (
    novelty_bonus          # +100 if never attempted (explore first)
    + weakness_weight      # +50 * (1 - task_success_rate) — worse tasks get higher priority
    + spaced_rep_bonus     # +30 if graduated task is "due" for re-test
    - recency_penalty      # -20 if attempted in last 2 episodes (ensure variety)
)
```

### `TaskGrader`

[server/services/task_grader.py](server/services/task_grader.py) — Evaluates task completion using a dispatcher pattern. Rewards are always in [0.0, 1.0].

**Grading strategies by tier:**

| Tier | Strategy | How it works |
|------|----------|--------------|
| Warmup | Command match | Checks command contains service string + correct operation |
| Beginner | Resource creation | Verifies resource actually exists in MiniStack via `ResourceVerifier` |
| Intermediate | Multi-step | Tracks ordered sequence of (operation, resource) pairs |
| Advanced | Multi-step + services | All steps completed AND all required services touched |
| Expert | State checks | Runs arbitrary AWS CLI commands to assert end-state (ground truth) |

### `HintProvider`

[server/services/hint_provider.py](server/services/hint_provider.py) — Generates progressive hints from `SuccessCriteria` fields.

| Hint Level | What it reveals | Example |
|-----------|----------------|---------|
| Level 1 | Which AWS services to use | "You'll need IAM and Lambda" |
| Level 2 | Which operations | "Start with create-role, then put-role-policy" |
| Level 3 | Near-complete command structure | "Use: aws iam create-role --role-name ..." |

### `EpisodeTracker`

[server/services/episode_tracker.py](server/services/episode_tracker.py) — Maintains per-episode step history. Parses AWS CLI commands to extract (service, operation, resource) tuples. Tracks credited operations for deduplication, monotonic progress, and hint usage.

### `ResourceVerifier`

[server/services/resource_verifier.py](server/services/resource_verifier.py) — Queries MiniStack directly to verify ground-truth resource state. Service-specific checks for S3, DynamoDB, Lambda, SQS, SNS, IAM, Secrets Manager, and API Gateway. Also evaluates `StateCheck` assertions (substring match, JSON path extraction).

### `EnvironmentDesigner`

[server/services/environment_designer.py](server/services/environment_designer.py) — Provisions initial AWS state via setup commands before the agent acts. Used by SRE/expert tasks to create broken or insecure infrastructure the agent must fix.

### `AwsBackend`

[server/services/aws_backend.py](server/services/aws_backend.py) — Executes AWS CLI commands against MiniStack (`AWS_ENDPOINT_URL=http://localhost:4566`). Provides `reset_environment()` via MiniStack's `/_ministack/reset` endpoint.

### `AwsRlEnv` (Client)

[client.py](client.py) — OpenEnv HTTP/WebSocket client. Wraps `reset()` and `step()` calls to the server.

---

## Data Models

[models.py](models.py) — All Pydantic models and type aliases.

### Action

```python
class AwsRlAction(Action):
    command: str   # AWS CLI command, e.g. "aws s3 ls"
```

### Observation

```python
class AwsRlObservation(Observation):
    episode_id: EpisodeID
    step_count: StepCount
    command_success: bool
    command_output: str          # stdout from AWS CLI
    error: str                   # stderr if failed
    task: TaskInfo | None        # masked task definition (hides success criteria)
    task_achieved: bool
    partial_progress: float      # current task progress in [0.0, 1.0]
    hints_used: int              # number of hints requested this episode
    hint_text: str               # most recent hint text (if any)
```

### Environment State

```python
class AwsRlState(State):
    current_task: Task | None    # full task assigned for the episode
    tracker: TrackerState        # episode tracker snapshot
    infra_state: dict            # AWS infrastructure state keyed by service name
    chaos_occurred: bool         # whether chaos was injected this episode
    current_tier: str            # agent's current difficulty tier

class TrackerState:
    step_count: int              # steps taken this episode
    hints_used: int              # hints requested this episode
    progress: float              # current partial progress [0.0, 1.0]
    commands_executed: list[str] # commands executed this episode
    credited_operations: list[str]  # (operation, resource) pairs that earned credit
```

### Task Definitions

```python
class Task:
    task_id: TaskID
    difficulty: TaskDifficulty   # warmup | beginner | intermediate | advanced | expert
    description: str             # human-readable goal
    success_criteria: SuccessCriteria
    setup_commands: list[SetupCommand]       # pre-provision for SRE tasks
    desired_state_spec: str | None           # natural-language desired end state (drift tasks)
    possible_drifts: list[SetupCommand]      # pool of mutations for DriftEngine

class TaskInfo:
    """Agent-visible subset of Task — masks success_criteria, setup_commands, and possible_drifts."""
    task_id: TaskID
    difficulty: TaskDifficulty
    description: str
    desired_state_spec: str | None

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
    chaos_probability: float  # probability of chaos injection per step (default: 0.0)

class SpacedRepState:
    interval: int                  # episodes until next re-test (3 -> 48)
    last_graduated_episode: int    # when last graduated
```

---

## Project Structure

```
aws-rl-env/
├── __init__.py                    # Exports: AwsRlEnv, AwsRlAction, AwsRlObservation
├── models.py                      # Pydantic data models & type aliases
├── client.py                      # AwsRlEnv OpenEnv client
├── inference.py                   # LLM agent inference script
├── inference-complete.py          # Full inference pipeline with curriculum
├── server/
│   ├── app.py                     # FastAPI application + web UI endpoints
│   ├── aws_rl_env_environment.py  # Core RL environment (reset/step)
│   ├── templates/
│   │   └── index.html             # Web playground UI
│   ├── static/
│   │   ├── css/style.css          # Playground styles
│   │   └── js/app.js              # Playground frontend logic
│   └── services/
│       ├── aws_backend.py         # MiniStack command executor
│       ├── task_grader.py         # Grading engine with reward shaping
│       ├── curriculum.py          # Curriculum learning manager
│       ├── episode_tracker.py     # Per-episode step history & hints
│       ├── resource_verifier.py   # Ground-truth state verification
│       ├── environment_designer.py # Setup provisioning for SRE tasks
│       ├── hint_provider.py       # Progressive hint generator
│       ├── chaos_engine.py        # Chaos injection engine
│       ├── drift_engine.py        # Drift detection engine
│       ├── task_solutions.py      # Reference solutions for tasks
│       └── tasks/
│           ├── warmup.yaml        # 20 listing tasks
│           ├── beginner.yaml      # 20 creation tasks
│           ├── intermediate.yaml  # 20 multi-step tasks
│           ├── advanced.yaml      # 20 architecture tasks
│           ├── expert.yaml        # 20 SRE/security tasks
│           └── drift.yaml         # Drift detection tasks
├── tests/                         # Unit tests for core services
│   ├── test_aws_rl_env_environment.py
│   ├── test_drift_engine.py
│   ├── test_environment_designer.py
│   ├── test_episode_tracker.py
│   ├── test_hint_provider.py
│   ├── test_resource_verifier.py
│   └── test_task_grader.py
├── tests_tasks/                   # Integration tests per task tier
│   ├── test_warmup_tasks.py
│   ├── test_beginner_tasks.py
│   ├── test_intermediate_tasks.py
│   ├── test_advanced_tasks.py
│   ├── test_expert_tasks.py
│   └── test_drift_tasks.py
├── aws_infra/                     # MiniStack emulator (git subtree from ministackorg/ministack)
│   └── ministack/
│       ├── app.py                 # MiniStack ASGI router
│       ├── core/                  # Routing, persistence, responses
│       └── services/              # AWS service implementations
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

Use the combined Makefile target:

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
| `AWS_INFRA_URL` | `http://localhost:4566` | AWS Infra endpoint |
| `AWS_ACCESS_KEY_ID` | `test` | AWS credentials (any value works) |
| `AWS_SECRET_ACCESS_KEY` | `test` | AWS credentials (any value works) |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region |
| `MAX_STEPS` | `15` | Max steps per episode |
| `API_BASE_URL` | — | LLM API endpoint (for inference.py) |
| `MODEL_NAME` | — | LLM model name (for inference.py) |
| `HF_TOKEN` | — | HuggingFace token (for inference.py) |
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

---

## Links

- **GitHub**: [github.com/udaykiranpadhy/aws-rl-env](https://github.com/udaykiranpadhy/aws-rl-env)
- **Hugging Face Space**: [huggingface.co/spaces/Sizzing/aws_rl_env](https://huggingface.co/spaces/Sizzing/aws_rl_env)
- **API Reference**: [/docs](https://sizzing-aws-rl-env.hf.space/docs)
- **ReDoc**: [/redoc](https://sizzing-aws-rl-env.hf.space/redoc)
- **Portfolio**: [portfolio.udaykp.dev](https://portfolio.udaykp.dev)
