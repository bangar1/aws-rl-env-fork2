"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=create-s3-bucket env=aws_rl_env model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=aws s3api create-bucket --bucket my-test-bucket reward=1.00 done=false error=null
    [END] success=true steps=1 rewards=1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import AwsRlEnv
from models import AwsRlAction

load_dotenv()  # Load variables from .env file if present

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
if not API_BASE_URL:
    API_BASE_URL = "https://router.huggingface.co/v1"
if not MODEL_NAME:
    MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY")  # Optional if using HF_TOKEN

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("BENCHMARK", "aws_rl_env")
MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
TEMPERATURE = 0.7
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 1.0  # task_achieved yields reward=1.0

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


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    task_description: str,
    step: int,
    last_output: str,
    last_error: str,
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
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
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "aws help"


async def main() -> None:
    key = HF_TOKEN if HF_TOKEN else API_KEY
    client = OpenAI(base_url=API_BASE_URL, api_key=key)

    try:
        env = await AwsRlEnv.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception as e:
        pass

    # After
    try:
        env = AwsRlEnv(base_url="https://sizzing-aws-rl-env.hf.space")
    except Exception as e:
        return

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_name = "unknown"
    task_description = ""

    try:
        result = await env.reset()  # OpenENV.reset()
        obs = result.observation

        # Extract task info from the first observation
        if obs.task is not None:
            task_name = f"task-{obs.task.task_id}"
            task_description = obs.task.description
        else:
            task_description = "No task assigned."

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        last_output = obs.command_output
        last_error = ""
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            command = get_model_command(
                client,
                task_description,
                step,
                last_output,
                last_error,
                last_reward,
                history,
            )

            result = await env.step(AwsRlAction(command=command))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.error if obs.error else None

            rewards.append(reward)
            steps_taken = step
            last_output = obs.command_output
            last_error = obs.error
            last_reward = reward

            log_step(step=step, action=command, reward=reward, done=done, error=error)

            status = "OK" if obs.command_success else "FAIL"
            history.append(f"Step {step} [{status}]: {command} -> reward={reward:.2f}")

            # Task achieved — episode success
            if obs.task_achieved:
                success = True
                break

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        if not success:
            success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[DEBUG] Unhandled exception in main: {e}", flush=True)
        raise e