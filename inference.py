"""
Inference Script — Code Debug Environment
==========================================
Structured stdout format: [START], [STEP], [END].
"""

import os
import textwrap
from typing import List

from openai import OpenAI

from client import AwsRlEnv
from models import AwsRlAction, AwsRlObservation
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file if present

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "aws-rl-env"
MAX_STEPS = 15

client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
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
            max_tokens=800
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


def run_task(env_url: str) -> None:

    

    with AwsRlEnv(base_url=env_url).sync() as env:
        for _ in range(11):
            result = env.reset()
            obs: AwsRlObservation = result.observation
            last_output = obs.command_output
            last_error = ""
            last_reward = 0.0
            history: List[str] = []
            rewards: List[float] = []
            print(f"[START] task={obs.task.task_id} env={BENCHMARK} model={MODEL_NAME}")

            for step in range(1, MAX_STEPS + 1):
                command = get_model_command(
                    client_llm,
                    obs.task.description,
                    obs.step_count,
                    last_output,
                    last_error,
                    last_reward,
                    history,
                )

                result = env.step(
                    AwsRlAction(command=command)
                )
                obs: AwsRlObservation = result.observation

                reward = obs.reward or 0.0
                done = result.done
                last_error = obs.error
                last_output = obs.command_output
                last_reward = reward

                
                # Clamp reward to strictly (0, 1) for validator
                if reward <= 0.0:
                    reward = 0.01
                elif reward >= 1.0:
                    reward = 0.99
                
                rewards.append(reward)
                steps = step

                done_str = "true" if done else "false"
                print(f"[STEP] step={step} action={command!r} reward={reward:.2f} done={done_str} error={last_error!r}")

                # Task achieved — episode success
                if obs.task_achieved:
                    success = True
                    break

                if done:
                    break

            score = max(rewards) if rewards else 0.1
            score = min(max(score, 0.01), 0.99)  # clamp to (0, 1)


            success_str = "true" if obs.task_achieved else "false"
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}")


if __name__ == "__main__":
    ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

    run_task(ENV_URL)