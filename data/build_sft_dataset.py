"""Generate SFT dataset for the AWS RL environment.

Produces command-only assistant targets in chat-messages JSONL format,
ready for trl.SFTTrainer + peft LoRA.

Composition (by source):
  55%  success_first_step       canonical command at step 1
  20%  multi_step_continuation  step N>1 with prior command history
  15%  failure_recovery         step 2 after a plausible typo/error
   5%  verification             read-only check after task completion
   5%  hint_usage               step 1 = aws help --task-hint

Tier sampling weights (warmup/beginner/intermediate/advanced/expert):
  0.50 / 0.30 / 0.15 / 0.05 / 0.00
  (expert skipped in SFT; GRPO will handle those with env reward)

Usage:
  python data/build_sft_dataset.py --train 1500 --val 150 --reserve 200 --seed 42
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` looking for the dir that contains server/services/tasks/.

    Makes the script location-independent: works whether it lives at repo root,
    in data/, scripts/, or anywhere else in the tree.
    """
    for p in [start, *start.parents]:
        if (p / "server" / "services" / "tasks").is_dir():
            return p
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
TASKS_DIR = REPO_ROOT / "server" / "services" / "tasks"
TESTS_DIR = REPO_ROOT / "tests_tasks"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "sft"

TIERS = ["warmup", "beginner", "intermediate", "advanced", "expert"]

TIER_WEIGHTS = {
    "warmup": 0.50,
    "beginner": 0.30,
    "intermediate": 0.15,
    "advanced": 0.05,
    "expert": 0.00,
}

SOURCE_MIX = {
    "success_first_step": 0.55,
    "multi_step_continuation": 0.20,
    "failure_recovery": 0.15,
    "verification": 0.05,
    "hint_usage": 0.05,
}

TESTS_FILES = {
    "warmup": ("test_warmup_tasks.py", "WARMUP_COMMANDS"),
    "beginner": ("test_beginner_tasks.py", "BEGINNER_COMMANDS"),
    "intermediate": ("test_intermediate_tasks.py", "INTERMEDIATE_COMMANDS"),
    "advanced": ("test_advanced_tasks.py", "ADVANCED_COMMANDS"),
    "expert": ("test_expert_tasks.py", "EXPERT_COMMANDS"),
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AWS cloud engineer interacting with a real AWS environment via CLI.
    Each turn you must send exactly ONE valid AWS CLI command (starting with 'aws').

    You will be given a task to accomplish. Read the task description carefully.
    Use the command output and error messages to guide your next action.

    Rules:
    - Only send AWS CLI commands (e.g. 'aws s3 ls', 'aws dynamodb create-table ...')
    - One command per turn — no pipes, no shell syntax, no chaining
    - Reply with ONLY the command, nothing else — no explanations, no quotes
    - If unsure, use 'aws help' to get unstuck, but try to be specific to the service if possible (e.g. 'aws s3 help')
    - When ever you need a hint, use 'aws help --task-hint' to get a task-specific hint (you can use this multiple times for more hints, but hints reduce your reward)
    """
).strip()

# Plausible reset-state outputs the env might return. Weighted toward "" since
# that's the most likely real value; others add mild prompt variance so the
# SFT trainer sees diverse surface forms of the same logical state.
_INITIAL_OUTPUTS: list[tuple[str, float]] = [
    ("", 0.7),
    ("Environment reset. Infra state wiped.", 0.2),
    ("Environment ready.", 0.1),
]


def _sample_initial_output(rng: random.Random) -> str:
    values, weights = zip(*_INITIAL_OUTPUTS)
    return rng.choices(values, weights=weights, k=1)[0]


def load_tasks() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for tier in TIERS:
        path = TASKS_DIR / f"{tier}.yaml"
        if not path.exists():
            out[tier] = []
            continue
        with open(path) as f:
            raw = yaml.safe_load(f) or []
        tasks = [t for t in raw if isinstance(t, dict) and "task_id" in t and "description" in t]
        out[tier] = tasks
    return out


def load_canonical_commands() -> dict[int, list[str]]:
    """Parse command dicts from tests_tasks/*.py without executing the files.

    Tests import pytest and set up fixtures, so importlib.exec fails outside
    the venv. AST-based literal extraction avoids that: we find the
    module-level `X_COMMANDS = {...}` assignment and literal-eval its value.
    Entries with non-literal values (f-strings etc.) are skipped silently.
    """
    out: dict[int, list[str]] = {}
    for _tier, (fname, var) in TESTS_FILES.items():
        path = TESTS_DIR / fname
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target_id, value = node.target.id, node.value
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target_id, value = node.targets[0].id, node.value
            else:
                continue
            if target_id != var or value is None:
                continue
            try:
                d = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(d, dict):
                continue
            for tid, cmd in d.items():
                if not isinstance(tid, int):
                    continue
                if isinstance(cmd, str):
                    out[tid] = [cmd]
                elif isinstance(cmd, (list, tuple)):
                    seq = [c for c in cmd if isinstance(c, str)]
                    if seq:
                        out[tid] = seq
    return out


def task_has_dynamic_ids(cmd_seq: list[str]) -> bool:
    """Detect commands that reference runtime-resolved IDs (sg-, subnet-, etc.)."""
    for cmd in cmd_seq:
        if re.search(r"\b(sg|subnet|vpc|ami|rtb|eni|igw|nat|eip|snap|vol)-[a-f0-9]{8,}\b", cmd):
            return True
        if re.search(r"\bi-[a-f0-9]{8,}\b", cmd):
            return True
    return False


OP_OUTPUTS: dict[str, str] = {
    "ls": "",
    "list-buckets": '{"Buckets":[]}',
    "list-tables": '{"TableNames":[]}',
    "list-functions": '{"Functions":[]}',
    "list-queues": "{}",
    "list-topics": '{"Topics":[]}',
    "list-users": '{"Users":[]}',
    "list-secrets": '{"SecretList":[]}',
    "list-clusters": '{"clusterArns":[]}',
    "list-named-queries": '{"NamedQueryIds":[]}',
    "describe-instances": '{"Reservations":[]}',
    "describe-db-instances": '{"DBInstances":[]}',
    "describe-cache-clusters": '{"CacheClusters":[]}',
    "get-databases": '{"DatabaseList":[]}',
    "create-bucket": '{"Location":"/<RESOURCE>"}',
    "put-object": '{"ETag":"\\"d41d8cd98f00b204e9800998ecf8427e\\""}',
    "put-bucket-versioning": "",
    "put-bucket-policy": "",
    "create-table": '{"TableDescription":{"TableName":"<RESOURCE>","TableStatus":"ACTIVE"}}',
    "put-item": "{}",
    "create-topic": '{"TopicArn":"arn:aws:sns:us-east-1:000000000000:<RESOURCE>"}',
    "create-queue": '{"QueueUrl":"https://sqs.us-east-1.amazonaws.com/000000000000/<RESOURCE>"}',
    "subscribe": '{"SubscriptionArn":"arn:aws:sns:us-east-1:000000000000:<RESOURCE>:abc123"}',
    "create-role": '{"Role":{"RoleName":"<RESOURCE>","Arn":"arn:aws:iam::000000000000:role/<RESOURCE>"}}',
    "attach-role-policy": "",
    "create-function": '{"FunctionName":"<RESOURCE>","FunctionArn":"arn:aws:lambda:us-east-1:000000000000:function:<RESOURCE>"}',
    "create-policy": '{"Policy":{"PolicyName":"<RESOURCE>","Arn":"arn:aws:iam::000000000000:policy/<RESOURCE>"}}',
    "create-secret": '{"ARN":"arn:aws:secretsmanager:us-east-1:000000000000:secret:<RESOURCE>"}',
    "get-bucket-policy": '{"Policy":"{\\"Version\\":\\"2012-10-17\\",\\"Statement\\":[{\\"Effect\\":\\"Allow\\",\\"Principal\\":\\"*\\",\\"Action\\":\\"s3:*\\",\\"Resource\\":\\"*\\"}]}"}',
}

_RESOURCE_FLAGS = (
    "--bucket",
    "--table-name",
    "--role-name",
    "--function-name",
    "--queue-name",
    "--topic-arn",
    "--policy-name",
    "--name",
    "--secret-id",
    "--cluster",
    "--key",
)


def simulate_output(command: str) -> str:
    tokens = command.split()
    if len(tokens) < 3:
        return ""
    op = tokens[2]
    resource = ""
    for i, tok in enumerate(tokens):
        if tok in _RESOURCE_FLAGS and i + 1 < len(tokens):
            resource = tokens[i + 1]
            break
    template = OP_OUTPUTS.get(op, "")
    return template.replace("<RESOURCE>", resource)


def _mistake_wrong_operation(cmd: str) -> tuple[str, str] | None:
    swaps = [
        ("list-tables", "ls"),
        ("list-buckets", "ls-all"),
        ("describe-instances", "list-instances"),
        ("list-functions", "list-lambdas"),
        ("create-bucket", "make-bucket"),
        ("put-item", "insert-item"),
        ("create-table", "make-table"),
        ("get-bucket-policy", "show-bucket-policy"),
        ("attach-role-policy", "attach-policy"),
        ("create-role", "new-role"),
    ]
    for good, bad in swaps:
        if good in cmd:
            return cmd.replace(good, bad, 1), (
                f"aws: error: argument operation: Invalid choice: '{bad}'"
            )
    return None


def _mistake_missing_arg(cmd: str) -> tuple[str, str] | None:
    m = re.search(r" (--[a-z-]+) (\S+)", cmd)
    if not m:
        return None
    flag, value = m.group(1), m.group(2)
    wrong = cmd.replace(f"{flag} {value}", "", 1).rstrip()
    return wrong, f"aws: error: the following arguments are required: {flag}"


def _mistake_wrong_service(cmd: str) -> tuple[str, str] | None:
    swaps = [
        ("aws dynamodb", "aws dynamo"),
        ("aws secretsmanager", "aws secrets"),
        ("aws cloudformation", "aws cfn"),
        ("aws elasticache", "aws elastic"),
        ("aws apigateway", "aws apigw"),
    ]
    for good, bad in swaps:
        if cmd.startswith(good):
            wrong = cmd.replace(good, bad, 1)
            return wrong, (
                f"aws: error: argument command: Invalid choice: '{bad.split()[-1]}'"
            )
    return None


def _mistake_s3_vs_s3api(cmd: str) -> tuple[str, str] | None:
    api_ops = (
        "create-bucket",
        "put-object",
        "put-bucket-versioning",
        "put-bucket-policy",
        "get-bucket-policy",
        "head-bucket",
    )
    for op in api_ops:
        if f"aws s3api {op}" in cmd:
            wrong = cmd.replace("aws s3api", "aws s3", 1)
            return wrong, (
                f"aws: error: argument operation: Invalid choice: '{op}'"
            )
    return None


def _mistake_typo_in_resource(cmd: str) -> tuple[str, str] | None:
    m = re.search(
        r"--(bucket|table-name|role-name|function-name|queue-name|name)\s+(\S+)", cmd
    )
    if not m:
        return None
    key, val = m.group(1), m.group(2)
    if len(val) < 3:
        return None
    typo = val[0] + val[2] + val[1] + val[3:]
    wrong = cmd.replace(f"--{key} {val}", f"--{key} {typo}", 1)
    return wrong, (
        f"An error occurred (NoSuchEntity): The resource '{typo}' was not found"
    )


_MISTAKES: list[Callable[[str], tuple[str, str] | None]] = [
    _mistake_wrong_operation,
    _mistake_missing_arg,
    _mistake_wrong_service,
    _mistake_s3_vs_s3api,
    _mistake_typo_in_resource,
]


def perturb_command(correct: str, rng: random.Random) -> tuple[str, str]:
    order = list(_MISTAKES)
    rng.shuffle(order)
    for m in order:
        out = m(correct)
        if out is not None:
            return out
    return correct + " --foo bar", "aws: error: unknown option: --foo"


def _jitter_reward(base: float, rng: random.Random) -> float:
    val = base + rng.uniform(-0.1, 0.1)
    return max(0.0, min(1.0, round(val, 2)))


def _trim_history(history: list[str], rng: random.Random) -> list[str]:
    if not history:
        return history
    options = [len(history), min(len(history), 4), min(len(history), 2)]
    k = rng.choice(options)
    return history[-k:]


def render_user_prompt(
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: list[str],
) -> str:
    history_block = "\n".join(history) if history else "None"
    return textwrap.dedent(
        f"""
        TASK: {task_description}

        Step: {step}
        Last command output: {last_output!r}
        Last error: {last_error!r}
        Last reward: {last_reward:.2f}

        Previous steps:
        {history_block}

        Send your next AWS CLI command.
        """
    ).strip()


def make_row(
    task: dict,
    tier: str,
    source: str,
    step_idx: int,
    user_prompt: str,
    assistant_command: str,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "difficulty": tier,
        "source": source,
        "step_idx": step_idx,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_command},
        ],
    }


def produce_success_first_step(
    task: dict, tier: str, commands: dict[int, list[str]], rng: random.Random
) -> dict | None:
    cmds = commands.get(task["task_id"])
    if not cmds:
        return None
    user = render_user_prompt(
        task["description"],
        step=0,
        last_output=_sample_initial_output(rng),
        last_error="",
        last_reward=_jitter_reward(0.0, rng),
        history=[],
    )
    return make_row(task, tier, "success_first_step", 0, user, cmds[0])


def produce_multi_step_continuation(
    task: dict, tier: str, commands: dict[int, list[str]], rng: random.Random
) -> dict | None:
    cmds = commands.get(task["task_id"])
    if not cmds or len(cmds) < 2:
        return None
    i = rng.randint(1, len(cmds) - 1)
    prior = cmds[:i]
    last_output = simulate_output(prior[-1])
    history = [f"{n + 1}. {c}" for n, c in enumerate(prior)]
    history = _trim_history(history, rng)
    base_reward = 0.2 + 0.6 * (i / len(cmds))
    user = render_user_prompt(
        task["description"],
        step=i,
        last_output=last_output,
        last_error="",
        last_reward=_jitter_reward(base_reward, rng),
        history=history,
    )
    return make_row(task, tier, "multi_step_continuation", i, user, cmds[i])


def produce_failure_recovery(
    task: dict, tier: str, commands: dict[int, list[str]], rng: random.Random
) -> dict | None:
    cmds = commands.get(task["task_id"])
    if not cmds:
        return None
    i = rng.randint(0, len(cmds) - 1)
    correct = cmds[i]
    wrong, err = perturb_command(correct, rng)
    history = [f"{n + 1}. {c}" for n, c in enumerate(cmds[:i])]
    history.append(f"{i + 1}. {wrong}")
    history = _trim_history(history, rng)
    step_now = i + 1
    user = render_user_prompt(
        task["description"],
        step=step_now,
        last_output="",
        last_error=err,
        last_reward=_jitter_reward(0.0 if i == 0 else 0.3, rng),
        history=history,
    )
    return make_row(task, tier, "failure_recovery", step_now, user, correct)


_VERIFY_MAP: list[tuple[str, Callable[[str], str | None]]] = [
    (
        "aws s3api create-bucket",
        lambda c: (
            f"aws s3api head-bucket{_flag_passthrough(c, '--bucket')}"
            if _flag_passthrough(c, "--bucket")
            else None
        ),
    ),
    (
        "aws s3api put-object",
        lambda c: (
            f"aws s3api list-objects-v2{_flag_passthrough(c, '--bucket')}"
            if _flag_passthrough(c, "--bucket")
            else None
        ),
    ),
    (
        "aws s3api put-bucket-versioning",
        lambda c: (
            f"aws s3api get-bucket-versioning{_flag_passthrough(c, '--bucket')}"
            if _flag_passthrough(c, "--bucket")
            else None
        ),
    ),
    (
        "aws s3api put-bucket-policy",
        lambda c: (
            f"aws s3api get-bucket-policy{_flag_passthrough(c, '--bucket')}"
            if _flag_passthrough(c, "--bucket")
            else None
        ),
    ),
    (
        "aws dynamodb create-table",
        lambda c: (
            f"aws dynamodb describe-table{_flag_passthrough(c, '--table-name')}"
            if _flag_passthrough(c, "--table-name")
            else None
        ),
    ),
    (
        "aws dynamodb put-item",
        lambda c: (
            f"aws dynamodb scan{_flag_passthrough(c, '--table-name')}"
            if _flag_passthrough(c, "--table-name")
            else None
        ),
    ),
    (
        "aws iam create-role",
        lambda c: (
            f"aws iam get-role{_flag_passthrough(c, '--role-name')}"
            if _flag_passthrough(c, "--role-name")
            else None
        ),
    ),
    (
        "aws iam attach-role-policy",
        lambda c: (
            f"aws iam list-attached-role-policies{_flag_passthrough(c, '--role-name')}"
            if _flag_passthrough(c, "--role-name")
            else None
        ),
    ),
    (
        "aws iam create-policy",
        lambda c: (
            f"aws iam list-policies --scope Local"
            if "--policy-name" in c
            else None
        ),
    ),
    (
        "aws lambda create-function",
        lambda c: (
            f"aws lambda get-function{_flag_passthrough(c, '--function-name')}"
            if _flag_passthrough(c, "--function-name")
            else None
        ),
    ),
    ("aws sns create-topic", lambda c: "aws sns list-topics"),
    ("aws sqs create-queue", lambda c: "aws sqs list-queues"),
    (
        "aws secretsmanager create-secret",
        lambda c: (
            f"aws secretsmanager describe-secret{_flag_passthrough(c, '--name', flag_out='--secret-id')}"
            if _flag_passthrough(c, "--name")
            else None
        ),
    ),
]


def _flag_passthrough(cmd: str, flag: str, flag_out: str | None = None) -> str:
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", cmd)
    if not m:
        return ""
    out_flag = flag_out or flag
    return f" {out_flag} {m.group(1)}"


def produce_verification(
    task: dict, tier: str, commands: dict[int, list[str]], rng: random.Random
) -> dict | None:
    cmds = commands.get(task["task_id"])
    if not cmds or len(cmds) < 2:
        return None
    last = cmds[-1]
    verify: str | None = None
    for prefix, fn in _VERIFY_MAP:
        if last.startswith(prefix):
            verify = fn(last)
            if verify:
                break
    if not verify:
        return None
    history = [f"{n + 1}. {c}" for n, c in enumerate(cmds)]
    history = _trim_history(history, rng)
    last_output = simulate_output(cmds[-1])
    step_now = len(cmds)
    user = render_user_prompt(
        task["description"],
        step=step_now,
        last_output=last_output,
        last_error="",
        last_reward=_jitter_reward(0.85, rng),
        history=history,
    )
    return make_row(task, tier, "verification", step_now, user, verify)


def produce_hint_usage(
    task: dict, tier: str, commands: dict[int, list[str]], rng: random.Random
) -> dict | None:
    user = render_user_prompt(
        task["description"],
        step=0,
        last_output=_sample_initial_output(rng),
        last_error="",
        last_reward=_jitter_reward(0.0, rng),
        history=[],
    )
    return make_row(task, tier, "hint_usage", 0, user, "aws help --task-hint")


PRODUCERS: dict[
    str, tuple[Callable[[dict, str, dict, random.Random], dict | None], list[str]]
] = {
    "success_first_step": (
        produce_success_first_step,
        ["warmup", "beginner", "intermediate", "advanced"],
    ),
    "multi_step_continuation": (
        produce_multi_step_continuation,
        ["intermediate", "advanced"],
    ),
    "failure_recovery": (
        produce_failure_recovery,
        ["warmup", "beginner", "intermediate", "advanced"],
    ),
    "verification": (produce_verification, ["intermediate", "advanced"]),
    "hint_usage": (produce_hint_usage, ["intermediate", "advanced"]),
}


def pick_tier(eligible_tiers: list[str], rng: random.Random) -> str | None:
    weights = [TIER_WEIGHTS[t] for t in eligible_tiers]
    if sum(weights) == 0:
        return rng.choice(eligible_tiers) if eligible_tiers else None
    return rng.choices(eligible_tiers, weights=weights, k=1)[0]


def build_dataset(
    n: int,
    tasks: dict[str, list[dict]],
    commands: dict[int, list[str]],
    rng: random.Random,
) -> list[dict]:
    counts = {src: round(n * pct) for src, pct in SOURCE_MIX.items()}
    diff = n - sum(counts.values())
    counts["success_first_step"] += diff

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for source, want in counts.items():
        producer, eligible_tiers = PRODUCERS[source]
        made = 0
        attempts = 0
        max_attempts = max(want * 20, 100)
        while made < want and attempts < max_attempts:
            attempts += 1
            tier = pick_tier(eligible_tiers, rng)
            if tier is None:
                break
            tier_tasks = [
                t
                for t in tasks.get(tier, [])
                if t["task_id"] in commands
                and not task_has_dynamic_ids(commands[t["task_id"]])
            ]
            if not tier_tasks:
                continue
            task = rng.choice(tier_tasks)
            row = producer(task, tier, commands, rng)
            if row is None:
                continue
            user_text = row["messages"][1]["content"]
            asst_text = row["messages"][2]["content"]
            key = (user_text, asst_text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            made += 1
        if made < want:
            print(
                f"  WARN: source '{source}' only produced {made}/{want} unique rows "
                f"after {attempts} attempts"
            )

    rng.shuffle(rows)
    return rows


def compute_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"total": 0}
    by_source = Counter(r["source"] for r in rows)
    by_tier = Counter(r["difficulty"] for r in rows)
    by_task = Counter(r["task_id"] for r in rows)
    return {
        "total": len(rows),
        "by_source": dict(by_source),
        "by_source_pct": {k: round(v / len(rows), 3) for k, v in by_source.items()},
        "by_tier": dict(by_tier),
        "by_tier_pct": {k: round(v / len(rows), 3) for k, v in by_tier.items()},
        "unique_tasks": len(by_task),
        "top_tasks": by_task.most_common(10),
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--reserve", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override auto-detected repo root (must contain server/services/tasks/)",
    )
    args = ap.parse_args()

    if args.repo_root is not None:
        global REPO_ROOT, TASKS_DIR, TESTS_DIR
        REPO_ROOT = args.repo_root.resolve()
        TASKS_DIR = REPO_ROOT / "server" / "services" / "tasks"
        TESTS_DIR = REPO_ROOT / "tests_tasks"

    if not TASKS_DIR.is_dir():
        raise SystemExit(
            f"ERROR: task dir not found at {TASKS_DIR}\n"
            f"  Auto-detected repo root: {REPO_ROOT}\n"
            f"  Pass --repo-root <path-to-aws-rl-env> to override."
        )

    rng = random.Random(args.seed)
    tasks = load_tasks()
    commands = load_canonical_commands()

    task_count = sum(len(v) for v in tasks.values())
    print(f"Loaded {task_count} tasks across {len(tasks)} tiers:")
    for tier in TIERS:
        tier_tasks = tasks.get(tier, [])
        with_cmd = sum(1 for t in tier_tasks if t["task_id"] in commands)
        with_cmd_no_dyn = sum(
            1
            for t in tier_tasks
            if t["task_id"] in commands
            and not task_has_dynamic_ids(commands[t["task_id"]])
        )
        print(
            f"  {tier:<13} {len(tier_tasks):3d} tasks  "
            f"({with_cmd} w/ canonical cmds, {with_cmd_no_dyn} after dynamic-id filter)"
        )
    print(f"Canonical commands loaded for {len(commands)} task IDs\n")

    total = args.train + args.val + args.reserve
    print(f"Building {total} rows (train={args.train} val={args.val} reserve={args.reserve})")
    all_rows = build_dataset(total, tasks, commands, rng)

    if len(all_rows) < total:
        print(f"\n  WARN: only built {len(all_rows)}/{total} unique rows")
        print("  Splits will be proportionally shrunk to preserve train/val/reserve ratio.\n")
        ratio = len(all_rows) / total
        train_n = int(args.train * ratio)
        val_n = int(args.val * ratio)
        reserve_n = len(all_rows) - train_n - val_n
    else:
        train_n, val_n, reserve_n = args.train, args.val, args.reserve

    train_rows = all_rows[:train_n]
    val_rows = all_rows[train_n : train_n + val_n]
    reserve_rows = all_rows[train_n + val_n : train_n + val_n + reserve_n]

    out_dir = args.out_dir
    write_jsonl(train_rows, out_dir / "aws_rl_sft.train.jsonl")
    write_jsonl(val_rows, out_dir / "aws_rl_sft.val.jsonl")
    write_jsonl(reserve_rows, out_dir / "aws_rl_sft.reserve.jsonl")

    stats = {
        "train": compute_stats(train_rows),
        "val": compute_stats(val_rows),
        "reserve": compute_stats(reserve_rows),
        "targets": {"source_mix": SOURCE_MIX, "tier_weights": TIER_WEIGHTS},
        "seed": args.seed,
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nWrote:")
    print(f"  {out_dir / 'aws_rl_sft.train.jsonl'}   ({len(train_rows)} rows)")
    print(f"  {out_dir / 'aws_rl_sft.val.jsonl'}     ({len(val_rows)} rows)")
    print(f"  {out_dir / 'aws_rl_sft.reserve.jsonl'} ({len(reserve_rows)} rows)")
    print(f"  {out_dir / 'dataset_stats.json'}")
    print(f"\nTrain source mix: {stats['train'].get('by_source_pct', {})}")
    print(f"Train tier mix:   {stats['train'].get('by_tier_pct', {})}")


if __name__ == "__main__":
    main()
