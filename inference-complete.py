"""
Complete Inference Loop — Sweeps All Difficulty Tiers
=====================================================

Runs the LLM agent through the full curriculum (warmup -> expert),
printing per-step rewards and per-tier summaries. The environment's
curriculum handles task selection and tier promotion automatically.

Prerequisites:
  - Environment accessible: Docker image (LOCAL_IMAGE_NAME) or running server (SERVER_URL)
  - LLM API accessible: API_BASE_URL + HF_TOKEN/API_KEY

Env vars:
  SERVER_URL       Server endpoint               (default: http://localhost:8000)
  LOCAL_IMAGE_NAME Docker image name             (uses Docker if set)
  API_BASE_URL     LLM API endpoint              (default: https://router.huggingface.co/v1)
  MODEL_NAME       Model identifier              (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN         HuggingFace API token
  API_KEY          Fallback API key
  MAX_STEPS        Max steps per episode          (default: 15)
  MAX_EPISODES     Max total episodes             (default: 200)
  TEMPERATURE      Generation temperature         (default: 0.7)
  MAX_TOKENS       Max generation tokens           (default: 512)
  CONVERGENCE_WINDOW   Episodes for plateau check (default: 10)
  CONVERGENCE_EPSILON  Min improvement threshold  (default: 0.01)
  CONVERGENCE_PATIENCE Consecutive plateau checks (default: 3)

Stopping: Runs until reward convergence is detected at expert tier,
  or MAX_EPISODES is reached. If the task is not complete the curriculum
  re-prioritises it (weakness gets higher priority), so the agent will
  retry weak tasks automatically.

Output format:
  ============================================================
  Episode 1 — Task 0: List all S3 buckets (tier: warmup)
    [Step 1] cmd="aws s3 ls" reward=1.00 success=True achieved=True
    Result: PASSED (steps=1, max_reward=1.00)
  ...
  FINAL RESULTS
  === TIER: warmup — 6/6 passed (100.0%) ===
  === OVERALL: 18/21 (85.7%) ===
"""

import asyncio
import os
import textwrap
from collections import defaultdict
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import AwsRlEnv
from models import AwsRlAction

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
MAX_EPISODES = int(os.getenv("MAX_EPISODES", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# Convergence at expert level
CONVERGENCE_WINDOW = int(os.getenv("CONVERGENCE_WINDOW", "10"))
CONVERGENCE_EPSILON = float(os.getenv("CONVERGENCE_EPSILON", "0.01"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))

ALL_TIERS = ["warmup", "beginner", "intermediate", "advanced", "expert"]

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
    """
).strip()


# ---------------------------------------------------------------------------
# Prompt building & model calling (reused from inference.py patterns)
# ---------------------------------------------------------------------------


def build_user_prompt(
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: List[str],
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


def get_model_command(
    client: OpenAI,
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(
        task_description, step, last_output, last_error, last_reward, history
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if the model wraps the command
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        return text if text.startswith("aws ") else "aws help"
    except Exception as exc:
        print(f"  [WARN] Model request failed: {exc}", flush=True)
        return "aws help"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(env: AwsRlEnv, llm_client: OpenAI) -> Optional[dict]:
    """Run a single episode: reset -> step loop -> return results."""
    result = await env.reset()
    obs = result.observation
    task = obs.task
    episode_num = obs.episode_id
    if task is None:
        print(f"Episode {episode_num} : No task assigned, skipping")
        return None

    tier = str(task.difficulty)
    task_desc = task.description
    task_id = int(task.task_id)

    print(f"\n{'=' * 60}")
    print(f"Episode {episode_num} -- Task {task_id}: {task_desc} (tier: {tier})")
    print(f"\n{'=' * 60}")

    history: List[str] = []
    last_output = obs.command_output
    last_error = ""
    last_reward = 0.0
    rewards: List[float] = []
    achieved = False

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        command = get_model_command(
            llm_client,
            task_desc,
            step,
            last_output,
            last_error,
            last_reward,
            history,
        )

        result = await env.step(AwsRlAction(command=command))
        obs = result.observation

        reward = result.reward or 0.0
        success = obs.command_success
        task_achieved = obs.task_achieved

        rewards.append(reward)

        print()
        print(f"\n{'-' * 60}")
        print(
            f'  [Step {step}] cmd="{command}"  command_output={obs.command_output!r} '
            f"reward={reward:.2f} command_success={success} achieved={task_achieved}"
        )
        print(f"\n{'-' * 60}")
        print()

        status = "OK" if success else "FAIL"
        history.append(
            f"Step {step} [{status}]: {command} [command_output]={obs.command_output!r} [error]={obs.error!r} -> reward={reward:.2f}"
        )
        last_output = obs.command_output
        last_error = obs.error
        last_reward = reward

        if task_achieved:
            achieved = True
            break

    max_reward = max(rewards) if rewards else 0.0
    result_str = "PASSED" if achieved else "FAILED"
    print(f"  Result: {result_str} (steps={len(rewards)}, max_reward={max_reward:.2f})")

    return {
        "task_id": task_id,
        "tier": tier,
        "achieved": achieved,
        "steps": len(rewards),
        "rewards": rewards,
        "max_reward": max_reward,
    }


# ---------------------------------------------------------------------------
# Convergence detector
# ---------------------------------------------------------------------------


class ConvergenceDetector:
    """Tracks reward history at expert tier and detects plateau."""

    def __init__(self) -> None:
        self.expert_rewards: List[float] = []
        self._plateau_count: int = 0

    def record(self, tier: str, max_reward: float) -> None:
        if tier == "expert":
            self.expert_rewards.append(max_reward)

    def is_converged(self) -> bool:
        if len(self.expert_rewards) < CONVERGENCE_WINDOW:
            return False

        half = CONVERGENCE_WINDOW // 2
        recent = self.expert_rewards[-half:]
        older = self.expert_rewards[-(half * 2) : -half]

        if not older:
            return False

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        improvement = abs(recent_avg - older_avg)

        if improvement < CONVERGENCE_EPSILON:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        return self._plateau_count >= CONVERGENCE_PATIENCE


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(tier_results: dict[str, list]) -> None:
    total_passed = 0
    total_tasks = 0

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")

    for tier in ALL_TIERS:
        results = tier_results.get(tier, [])
        if not results:
            print(f"\n=== TIER: {tier} -- not reached ===")
            continue

        passed = sum(1 for r in results if r["achieved"])
        total = len(results)
        pct = (passed / total * 100) if total > 0 else 0

        print(f"\n=== TIER: {tier} -- {passed}/{total} passed ({pct:.1f}%) ===")
        for r in results:
            status = "PASS" if r["achieved"] else "FAIL"
            print(
                f"  Task {r['task_id']}: {status} "
                f"(steps={r['steps']}, reward={r['max_reward']:.2f})"
            )

        total_passed += passed
        total_tasks += total

    overall_pct = (total_passed / total_tasks * 100) if total_tasks > 0 else 0
    print(f"\n=== OVERALL: {total_passed}/{total_tasks} ({overall_pct:.1f}%) ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    key = HF_TOKEN if HF_TOKEN else API_KEY
    if not key:
        print("ERROR: Set HF_TOKEN or API_KEY for the LLM API.")
        return

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=key)

    # Connect to environment
    if LOCAL_IMAGE_NAME:
        print(f"Starting environment from Docker image: {LOCAL_IMAGE_NAME}")
        env = await AwsRlEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        print(f"Connecting to server: {SERVER_URL}")
        env = AwsRlEnv(base_url=SERVER_URL)

    tier_results: dict[str, list] = defaultdict(list)
    convergence = ConvergenceDetector()

    try:
        for episode in range(1, MAX_EPISODES + 1):
            ep_result = await run_episode(env, llm_client)

            if ep_result is None:
                continue

            tier = ep_result["tier"]
            tier_results[tier].append(ep_result)

            # Track convergence
            convergence.record(tier, ep_result["max_reward"])

            if convergence.is_converged():
                print(
                    f"\nConvergence detected at expert level "
                    f"after {episode} episodes "
                    f"(plateau for {CONVERGENCE_PATIENCE} consecutive checks). "
                    f"Stopping."
                )
                break

        print_summary(tier_results)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[WARN] env.close() error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
