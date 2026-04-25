"""GRPO training for the AWS RL environment — multi-turn rollouts + parallel envs.

Mirrors the kube-sre-gym training pattern (heavy logic in this module, thin
notebook on top):
  - Each "episode" runs up to MAX_TURNS steps.
  - Each step = one ``aws ...`` command; the command's stdout/stderr is fed
    back into the next turn's prompt as the user message.
  - Each GRPO step picks ONE curriculum task and runs G concurrent rollouts
    (one per env in MultiTurnEnvPool) sharing that task.
  - prompt_ids / completion_ids / logprobs are accumulated across turns so
    GRPO assigns episode-level reward to the full token sequence.

Usage (CLI)::

    # Single training pass with explicit hyperparams
    python train_grpo.py --mode train \\
        --env-url http://localhost:8000 \\
        --num-generations 8 --max-turns 6 --max-steps 200

    # Optuna search over hyperparams, then dump best_cfg.json
    python train_grpo.py --mode optuna --n-trials 6

    # Optuna search, then full-length retrain using the best config
    python train_grpo.py --mode full --n-trials 6 --max-steps 200
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from client import AwsRlEnv
from models import AwsRlAction, AwsRlObservation, Task, TaskDifficulty, TaskID
from server.services.curriculum import Curriculum

logger = logging.getLogger(__name__)


# ============================================================
# System prompt — multi-turn AWS CLI agent
# ============================================================

SYSTEM_PROMPT = """You are an expert AWS Operations agent. You operate a simulated AWS cloud by emitting ONE AWS CLI command per turn.

The user message contains:
  - The task description.
  - (Optional) A history of your previous commands and their outputs from earlier in this episode — use them to decide your next move.
  - The most recent observation (last command's stdout / stderr / progress).

Each turn:
  1. Optionally reason inside a single <think>...</think> block. Keep it concise.
  2. After </think>, on a NEW LINE, output EXACTLY ONE AWS CLI command starting with "aws ".

Hard rules:
  - The command line must contain ONLY the command — no markdown, no backticks, no quotes around it, no trailing commentary.
  - If a command failed last turn, try a DIFFERENT approach. Do not repeat the exact same command twice in a row.
  - When the task description names a specific resource (a bucket, table, queue, etc.), use that exact name.
"""


DEFAULT_CFG: dict[str, Any] = {
    "learning_rate": 5e-6,
    "beta": 0.04,
    "num_generations": 8,
    "temperature": 0.9,
    "top_p": 0.95,
    "lora_r": 16,
    "lora_alpha_mul": 2,
    "max_turns": 6,
}


# ============================================================
# Helpers — prompt formatting + command parsing
# ============================================================

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think\b[^>]*>.*", re.DOTALL | re.IGNORECASE)


def extract_aws_command(raw: str) -> str:
    """Strip <think> blocks + markdown fences, return the first ``aws ...`` line.

    Falls back to ``aws help`` so the env always gets a syntactically valid
    command (the env will just produce a help-text observation, which is a
    better RL signal than a parse error).
    """
    cleaned = _THINK_BLOCK.sub("", raw)
    cleaned = _OPEN_THINK.sub("", cleaned)
    for line in cleaned.splitlines():
        line = line.strip().strip("`").strip()
        if line.startswith("aws "):
            return line
    return "aws help"


def _truncate(text: str, n: int) -> str:
    if not text:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def format_observation(obs: AwsRlObservation) -> str:
    """Render the latest env observation as a compact text block."""
    parts: list[str] = []
    if obs.command_output:
        parts.append(f"Output:\n{_truncate(obs.command_output, 800)}")
    if obs.error:
        parts.append(f"Error:\n{_truncate(obs.error, 400)}")
    parts.append(
        f"Progress: {obs.partial_progress:.2f}  "
        f"Achieved: {obs.task_achieved}  Step: {obs.step_count}"
    )
    if obs.hint_text:
        parts.append(f"Hint: {_truncate(obs.hint_text, 200)}")
    return "\n".join(parts)


def format_history(history: list[dict], keep_last: int = 6) -> str:
    """Render the last ``keep_last`` (cmd, output, reward) tuples for context."""
    if not history:
        return ""
    recent = history[-keep_last:]
    rendered: list[str] = ["PREVIOUS COMMANDS:"]
    for i, h in enumerate(recent, start=max(1, len(history) - keep_last + 1)):
        rendered.append(
            f"[{i}] $ {h['command']}\n"
            f"    output: {_truncate(h['output'], 300)}\n"
            f"    reward: {h['reward']:.2f}"
        )
    return "\n".join(rendered)


def apply_chat_template(tokenizer: AutoTokenizer, messages: list[dict]) -> str:
    """Apply a chat template; fall back to a plain rendering if none is set."""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False)
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|{m['role']}|>\n{m['content']}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def build_user_prompt(task: Task, obs: AwsRlObservation, history: list[dict]) -> str:
    desc = task.description
    if task.desired_state_spec:
        desc = f"{desc}\n\nDesired end state:\n{task.desired_state_spec}"
    history_text = format_history(history)
    obs_text = format_observation(obs)
    if history_text:
        return f"TASK: {desc}\n\n{history_text}\n\n---\n\nCURRENT OBSERVATION:\n{obs_text}"
    return f"TASK: {desc}\n\nCURRENT OBSERVATION:\n{obs_text}"


# ============================================================
# Policy loading — Unsloth 4-bit base + LoRA-from-SFT-adapter
# ============================================================


@dataclass
class PolicySpec:
    base_model: str = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
    sft_adapter: str = "Sizzing/aws-rl-sft-qwen25coder3b-adapter"
    max_seq_length: int = 3072


def load_policy(
    base_model: str,
    sft_adapter: Optional[str] = None,
    max_seq_length: int = 3072,
    trainable: bool = True,
):
    """Load Unsloth 4-bit base + (optional) LoRA adapter from the SFT run.

    ``trainable=True`` returns a PeftModel ready for GRPO training (Unsloth's
    training kernels enabled, input require-grads hook installed).
    ``trainable=False`` returns the same stack in inference mode for eval.
    """
    from unsloth import FastLanguageModel

    base, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    if sft_adapter:
        model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=trainable)
    else:
        # No adapter: GRPOTrainer can attach a fresh LoRA via peft_config later.
        model = base

    if trainable:
        FastLanguageModel.for_training(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def free_model(model) -> None:
    """Release VRAM held by ``model`` and any captured optimizer state."""
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Multi-turn rollout — one episode in one env
# ============================================================


@dataclass
class SamplingCfg:
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 256
    max_prompt_length: int = 2048


_GENERATE_LOCK = threading.Lock()
"""Serialise model.generate() calls across the asyncio.gather rollout group.

The model lives on a single GPU; concurrent generate() calls would collide.
We let the env step run concurrently (the slow part — WebSocket round-trip +
MiniStack execution); only the generation is serialised.
"""


def _generate_with_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    sampling: SamplingCfg,
) -> tuple[list[int], list[int], list[float]]:
    """Generate one completion + return per-token logprobs.

    Returns: (prompt_ids, completion_ids, completion_logprobs).
    """
    with _GENERATE_LOCK:
        prompt_input = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=sampling.max_prompt_length,
        ).to(model.device)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                gen_out = model.generate(
                    **prompt_input,
                    max_new_tokens=sampling.max_new_tokens,
                    do_sample=True,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            if was_training:
                model.train()

        prompt_ids = prompt_input.input_ids[0].tolist()
        prompt_len = len(prompt_ids)
        completion_seq = gen_out.sequences[0, prompt_len:].tolist()

        # Per-token logprobs from raw logits.
        logprobs: list[float] = []
        for i, scores_t in enumerate(gen_out.scores):
            if i >= len(completion_seq):
                break
            lp = torch.log_softmax(scores_t[0].float(), dim=-1)
            logprobs.append(float(lp[completion_seq[i]].item()))

    return prompt_ids, completion_seq, logprobs


async def rollout_one_episode(
    env: AwsRlEnv,
    task: Task,
    model,
    tokenizer,
    system_prompt: str,
    max_turns: int,
    max_total_tokens: int,
    sampling: SamplingCfg,
) -> dict:
    """Run one multi-turn episode in one env, accumulating tokens across turns."""
    try:
        res = await env.reset(task=task)
    except Exception as e:
        logger.warning("reset() failed for task=%s: %s", task.task_id, e)
        return {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "task_reward": -1.0,
            "task_achieved": False,
            "final_progress": 0.0,
            "num_steps": 0,
            "transcript": [{"error": f"reset failed: {e!r}"}],
        }
    obs: AwsRlObservation = res.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    history: list[dict] = []
    final_progress = float(getattr(obs, "partial_progress", 0.0) or 0.0)
    final_achieved = bool(getattr(obs, "task_achieved", False))

    for _turn in range(max_turns):
        if res.done:
            break
        if len(completion_ids) >= max_total_tokens:
            break

        user_text = build_user_prompt(task, obs, history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        # Generation runs on the calling thread (blocking) but env.step calls
        # for other rollouts in this group can overlap because they're all
        # awaiting in the same loop.
        loop = asyncio.get_running_loop()
        turn_prompt_ids, turn_completion_ids, turn_logprobs = await loop.run_in_executor(
            None, _generate_with_logprobs, model, tokenizer, prompt_text, sampling
        )
        completion_text = tokenizer.decode(turn_completion_ids, skip_special_tokens=True)
        cmd = extract_aws_command(completion_text)

        try:
            res = await env.step(AwsRlAction(command=cmd))
            step_reward = float(res.reward or 0.0)
        except Exception as e:
            logger.warning("step() error on cmd=%r: %s", cmd[:80], e)
            step_reward = -0.1
            history.append(
                {
                    "command": cmd,
                    "output": f"ERROR: {e!r}",
                    "reward": step_reward,
                }
            )
            prompt_ids.extend(turn_prompt_ids)
            completion_ids.extend(turn_completion_ids)
            logprobs.extend(turn_logprobs)
            step_rewards.append(step_reward)
            break

        prompt_ids.extend(turn_prompt_ids)
        completion_ids.extend(turn_completion_ids)
        logprobs.extend(turn_logprobs)
        step_rewards.append(step_reward)
        obs = res.observation
        final_progress = float(getattr(obs, "partial_progress", 0.0) or 0.0)
        final_achieved = bool(getattr(obs, "task_achieved", False))
        history.append(
            {
                "command": cmd,
                "output": _truncate(getattr(obs, "command_output", "") or "", 500),
                "reward": step_reward,
            }
        )

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "task_reward": float(sum(step_rewards)) if step_rewards else -1.0,
        "task_achieved": final_achieved,
        "final_progress": final_progress,
        "num_steps": len(history),
        "transcript": history,
        "task_id": int(task.task_id),
        "difficulty": task.difficulty.value,
    }


# ============================================================
# MultiTurnEnvPool — sync wrapper around N async env sessions
# ============================================================


class MultiTurnEnvPool:
    """N persistent WebSocket env sessions, exposed via a sync ``run_group`` API.

    Owns a background thread running an asyncio loop. Connect / close happens
    once for the lifetime of training. Submitted coroutines run in the
    background loop via ``asyncio.run_coroutine_threadsafe`` and the calling
    thread blocks on the resulting concurrent.futures.Future.
    """

    def __init__(self, base_url: str, size: int, timeout_s: float = 120.0) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        self.base_url = base_url
        self.size = size
        self.timeout_s = timeout_s
        self._envs: list[AwsRlEnv] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._setup_error: Optional[BaseException] = None

    def start(self) -> None:
        """Open N WebSocket sessions on the background loop."""
        if self._thread is not None:
            return

        def run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._connect_all())
            except BaseException as e:
                self._setup_error = e
                self._ready.set()
                return
            self._ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True, name="env-pool")
        self._thread.start()
        self._ready.wait()
        if self._setup_error is not None:
            raise RuntimeError(
                f"MultiTurnEnvPool failed to connect {self.size} sessions to "
                f"{self.base_url}: {self._setup_error!r}"
            )
        logger.info("MultiTurnEnvPool: %d sessions on %s", self.size, self.base_url)

    async def _connect_all(self) -> None:
        envs = [AwsRlEnv(base_url=self.base_url) for _ in range(self.size)]
        try:
            await asyncio.gather(*(e.connect() for e in envs))
        except BaseException:
            await asyncio.gather(*(e.close() for e in envs), return_exceptions=True)
            raise
        self._envs = envs

    def close(self) -> None:
        if self._thread is None or self._loop is None:
            return
        loop = self._loop

        async def _shutdown() -> None:
            await asyncio.gather(
                *(e.close() for e in self._envs), return_exceptions=True
            )

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            fut.result(timeout=10.0)
        except Exception as e:
            logger.warning("Pool shutdown error (ignored): %s", e)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=5.0)
            self._thread = None
            self._loop = None
            self._envs = []

    def run_group(
        self,
        task: Task,
        model,
        tokenizer,
        system_prompt: str,
        max_turns: int,
        max_total_tokens: int,
        sampling: SamplingCfg,
    ) -> list[dict]:
        """Run N concurrent multi-turn rollouts on the same task. Sync; blocks."""
        assert self._loop is not None and self._envs, "call start() first"

        async def _gather() -> list[dict]:
            return list(
                await asyncio.gather(
                    *(
                        rollout_one_episode(
                            env,
                            task,
                            model,
                            tokenizer,
                            system_prompt,
                            max_turns,
                            max_total_tokens,
                            sampling,
                        )
                        for env in self._envs
                    )
                )
            )

        fut = asyncio.run_coroutine_threadsafe(_gather(), self._loop)
        return fut.result(timeout=self.timeout_s * max(1, max_turns))

    def __enter__(self) -> "MultiTurnEnvPool":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ============================================================
# Reward functions (TRL convention) + rollout_func factory
# ============================================================


def reward_task(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("task_reward")
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_achieved(completions: list[str], **kwargs) -> list[float]:
    flags = kwargs.get("task_achieved")
    if flags is None:
        return [0.0 for _ in completions]
    return [float(f) for f in flags]


def reward_progress(completions: list[str], **kwargs) -> list[float]:
    progress = kwargs.get("final_progress")
    if progress is None:
        return [0.0 for _ in completions]
    return [float(p) for p in progress]


def make_rollout_func(
    curriculum: Curriculum,
    pool: MultiTurnEnvPool,
    model,
    tokenizer,
    system_prompt: str,
    max_turns: int,
    max_total_tokens: int,
    sampling: SamplingCfg,
    log_episode: Callable[[Task, list[dict]], None],
) -> Callable:
    """Build the closure GRPO calls each step.

    ``prompts`` length equals ``num_generations``. We ignore the prompt strings
    because the curriculum drives task selection — every rollout in the group
    runs the same task forced through ``env.reset(task=...)``.
    """

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        task = curriculum.next_task()
        results = pool.run_group(
            task,
            model,
            tokenizer,
            system_prompt,
            max_turns,
            max_total_tokens,
            sampling,
        )
        # Pad / truncate to len(prompts) — defence in depth, group size should match.
        if len(results) < len(prompts):
            results.extend(results[-1:] * (len(prompts) - len(results)))
        results = results[: len(prompts)]

        group_rewards = [r["task_reward"] for r in results]
        group_achieved = [r["task_achieved"] for r in results]
        group_progress = [r["final_progress"] for r in results]

        curriculum.record_result(
            task,
            achieved=any(group_achieved),
            reward=float(sum(group_rewards) / len(group_rewards)) if group_rewards else 0.0,
        )
        log_episode(task, results)

        return {
            "prompt_ids": [r["prompt_ids"] for r in results],
            "completion_ids": [r["completion_ids"] for r in results],
            "logprobs": [r["logprobs"] for r in results],
            "task_reward": group_rewards,
            "task_achieved": [float(a) for a in group_achieved],
            "final_progress": group_progress,
        }

    return rollout_func


# ============================================================
# CSV / JSONL logging + reward plotter
# ============================================================


class EpisodeLogger:
    """Append-only CSV + JSONL writer for per-rollout episode rows."""

    HEADER = [
        "step",
        "rollout_idx",
        "task_id",
        "difficulty",
        "task_reward",
        "task_achieved",
        "final_progress",
        "num_steps",
        "tier",
        "tier_success_rate",
        "timestamp",
    ]

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = output_dir / "reward_log.csv"
        self.jsonl_path = output_dir / "transcripts.jsonl"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)
        self._step_counter = 0

    def log(self, task: Task, results: list[dict], curriculum: Curriculum) -> None:
        self._step_counter += 1
        stats = curriculum.get_stats()
        ts = datetime.now().isoformat()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for i, r in enumerate(results):
                writer.writerow(
                    [
                        self._step_counter,
                        i,
                        int(task.task_id),
                        task.difficulty.value,
                        f"{r['task_reward']:.4f}",
                        int(bool(r["task_achieved"])),
                        f"{r['final_progress']:.4f}",
                        r["num_steps"],
                        stats["tier"],
                        stats["tier_success_rate"],
                        ts,
                    ]
                )
        with open(self.jsonl_path, "a") as f:
            for i, r in enumerate(results):
                f.write(
                    json.dumps(
                        {
                            "step": self._step_counter,
                            "rollout_idx": i,
                            "task_id": int(task.task_id),
                            "difficulty": task.difficulty.value,
                            "task_reward": r["task_reward"],
                            "task_achieved": bool(r["task_achieved"]),
                            "final_progress": r["final_progress"],
                            "num_steps": r["num_steps"],
                            "tier": stats["tier"],
                            "transcript": r["transcript"],
                        }
                    )
                    + "\n"
                )

        rewards = [r["task_reward"] for r in results]
        achieved = [bool(r["task_achieved"]) for r in results]
        logger.info(
            "Step %d task=%d (%s) rewards=%s achieved=%d/%d tier=%s tier_rate=%.2f",
            self._step_counter,
            int(task.task_id),
            task.difficulty.value,
            [round(r, 2) for r in rewards],
            sum(achieved),
            len(achieved),
            stats["tier"],
            stats["tier_success_rate"],
        )


def plot_rewards(csv_path: Path, out_path: Path) -> None:
    """Per-step mean group reward + 10-step rolling avg + per-tier curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not csv_path.exists():
        logger.warning("No CSV at %s — skipping plot.", csv_path)
        return

    steps_data: dict[int, list[float]] = {}
    tier_data: dict[str, list[tuple[int, float]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            r = float(row["task_reward"])
            tier = row["tier"]
            steps_data.setdefault(step, []).append(r)
            tier_data.setdefault(tier, []).append((step, r))

    if not steps_data:
        logger.warning("CSV at %s has no rows — skipping plot.", csv_path)
        return

    steps = sorted(steps_data.keys())
    means = [sum(steps_data[s]) / len(steps_data[s]) for s in steps]

    rolling = []
    window = 10
    for i in range(len(means)):
        lo = max(0, i - window + 1)
        rolling.append(sum(means[lo : i + 1]) / (i - lo + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(steps, means, label="mean group reward", alpha=0.5)
    ax1.plot(steps, rolling, label=f"rolling avg (k={window})", linewidth=2)
    ax1.set_xlabel("GRPO step")
    ax1.set_ylabel("reward")
    ax1.set_title("Group mean reward over training")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for tier, points in tier_data.items():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax2.scatter(xs, ys, s=10, alpha=0.5, label=tier)
    ax2.set_xlabel("GRPO step")
    ax2.set_ylabel("reward")
    ax2.set_title("Per-rollout reward by curriculum tier")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Reward plot written to %s", out_path)


# ============================================================
# Validation eval + Optuna search
# ============================================================


def pick_validation_task_ids(
    curriculum: Optional[Curriculum] = None,
    k_per_tier: int = 2,
    seed: int = 42,
) -> list[int]:
    """Pick a frozen list of task ids — k per tier — for held-out validation."""
    import random

    rng = random.Random(seed)
    cur = curriculum or Curriculum()
    chosen: list[int] = []
    for tier in TaskDifficulty:
        try:
            from server.services.curriculum import load_tier

            tier_tasks = load_tier(tier, cur._tasks_dir)
        except Exception as e:
            logger.warning("Could not load tier %s for val: %s", tier.value, e)
            continue
        if not tier_tasks:
            continue
        sample = rng.sample(tier_tasks, k=min(k_per_tier, len(tier_tasks)))
        chosen.extend(int(t.task_id) for t in sample)
    return chosen


def evaluate_on_validation(
    model,
    tokenizer,
    pool: MultiTurnEnvPool,
    val_task_ids: list[int],
    system_prompt: str,
    max_turns: int,
    max_total_tokens: int,
    sampling: SamplingCfg,
    curriculum: Optional[Curriculum] = None,
) -> dict[str, float]:
    """Run ONE rollout per val task on env[0] of the pool. Return aggregate metrics."""
    cur = curriculum or Curriculum()
    achieved_flags: list[float] = []
    progresses: list[float] = []
    rewards: list[float] = []

    async def _eval_one(task: Task) -> dict:
        env = pool._envs[0]
        return await rollout_one_episode(
            env,
            task,
            model,
            tokenizer,
            system_prompt,
            max_turns,
            max_total_tokens,
            sampling,
        )

    for tid in val_task_ids:
        try:
            task = cur.get_task_by_id(TaskID(int(tid)))
        except KeyError:
            logger.warning("val task_id=%d not found — skipping", tid)
            continue
        fut = asyncio.run_coroutine_threadsafe(_eval_one(task), pool._loop)
        try:
            res = fut.result(timeout=pool.timeout_s * max(1, max_turns))
        except Exception as e:
            logger.warning("val rollout failed for task=%d: %s", tid, e)
            continue
        achieved_flags.append(float(res["task_achieved"]))
        progresses.append(float(res["final_progress"]))
        rewards.append(float(res["task_reward"]))

    n = max(1, len(achieved_flags))
    return {
        "achieved_rate": sum(achieved_flags) / n,
        "mean_progress": sum(progresses) / n,
        "mean_reward": sum(rewards) / n,
        "n_evaluated": float(len(achieved_flags)),
    }


def _build_grpo_config(
    output_dir: Path,
    cfg: dict[str, Any],
    max_steps: int,
    max_completion_length: int,
    max_prompt_length: int,
    save_steps: int = 25,
    save_strategy: str = "steps",
    report_to: str = "none",
) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        learning_rate=float(cfg["learning_rate"]),
        beta=float(cfg["beta"]),
        num_generations=int(cfg["num_generations"]),
        generation_batch_size=int(cfg["num_generations"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        report_to=report_to,
        loss_type="dapo",
        mask_truncated_completions=True,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        use_vllm=False,
        remove_unused_columns=False,
    )


def _build_dummy_dataset(num_rows: int) -> Dataset:
    """A length-only dataset; the prompts are ignored by ``rollout_func``."""
    return Dataset.from_dict({"prompt": ["solve"] * max(1, num_rows)})


def optuna_search(
    n_trials: int,
    trial_max_steps: int,
    val_task_ids: list[int],
    base_model: str,
    sft_adapter: Optional[str],
    env_url: str,
    output_dir: Path,
    max_total_tokens: int = 2048,
    max_completion_length: int = 256,
    max_prompt_length: int = 2048,
    seed: int = 42,
):
    """TPE-sampled hyperparam search. Persists to ``output_dir/optuna.db``."""
    import optuna

    output_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        study_name="aws-rl-grpo",
        storage=f"sqlite:///{output_dir / 'optuna.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def _objective(trial: optuna.Trial) -> float:
        cfg = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            "beta": trial.suggest_float("beta", 0.0, 0.1),
            "num_generations": trial.suggest_categorical("num_generations", [4, 8]),
            "temperature": trial.suggest_float("temperature", 0.7, 1.0),
            "top_p": trial.suggest_float("top_p", 0.85, 0.98),
            "lora_r": trial.suggest_categorical("lora_r", [8, 16, 32]),
            "lora_alpha_mul": trial.suggest_categorical("lora_alpha_mul", [1, 2, 4]),
            "max_turns": trial.suggest_categorical("max_turns", [4, 6, 8]),
        }
        trial_dir = output_dir / f"trial_{trial.number:03d}"
        return _run_one_trial(
            cfg=cfg,
            trial_max_steps=trial_max_steps,
            val_task_ids=val_task_ids,
            base_model=base_model,
            sft_adapter=sft_adapter,
            env_url=env_url,
            output_dir=trial_dir,
            max_total_tokens=max_total_tokens,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
        )

    study.optimize(_objective, n_trials=n_trials, gc_after_trial=True)

    best_path = output_dir / "best_cfg.json"
    payload = {"best_value": study.best_value, "best_params": dict(study.best_params)}
    with open(best_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "Optuna study finished. best_value=%.4f best_params=%s -> %s",
        study.best_value,
        study.best_params,
        best_path,
    )
    return study


def _run_one_trial(
    cfg: dict[str, Any],
    trial_max_steps: int,
    val_task_ids: list[int],
    base_model: str,
    sft_adapter: Optional[str],
    env_url: str,
    output_dir: Path,
    max_total_tokens: int,
    max_completion_length: int,
    max_prompt_length: int,
) -> float:
    """One Optuna trial: load → train → eval on val tasks → tear down → return objective."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Optuna trial cfg=%s -> %s", cfg, output_dir)

    model = tokenizer = None
    pool: Optional[MultiTurnEnvPool] = None
    trainer: Optional[GRPOTrainer] = None
    try:
        model, tokenizer = load_policy(base_model, sft_adapter, trainable=True)

        pool = MultiTurnEnvPool(env_url, size=int(cfg["num_generations"]))
        pool.start()

        curriculum = Curriculum()
        sampling = SamplingCfg(
            temperature=float(cfg["temperature"]),
            top_p=float(cfg["top_p"]),
            max_new_tokens=max_completion_length,
            max_prompt_length=max_prompt_length,
        )
        ep_logger = EpisodeLogger(output_dir)
        rollout_func = make_rollout_func(
            curriculum=curriculum,
            pool=pool,
            model=model,
            tokenizer=tokenizer,
            system_prompt=SYSTEM_PROMPT,
            max_turns=int(cfg["max_turns"]),
            max_total_tokens=max_total_tokens,
            sampling=sampling,
            log_episode=lambda task, results: ep_logger.log(task, results, curriculum),
        )

        dataset = _build_dummy_dataset(trial_max_steps * int(cfg["num_generations"]))
        grpo_cfg = _build_grpo_config(
            output_dir=output_dir,
            cfg=cfg,
            max_steps=trial_max_steps,
            max_completion_length=max_completion_length,
            max_prompt_length=max_prompt_length,
            save_strategy="no",
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_task, reward_achieved, reward_progress],
            train_dataset=dataset,
            args=grpo_cfg,
            rollout_func=rollout_func,
            peft_config=None if sft_adapter else _lora_config(cfg),
        )
        trainer.train()

        metrics = evaluate_on_validation(
            model=trainer.model,
            tokenizer=tokenizer,
            pool=pool,
            val_task_ids=val_task_ids,
            system_prompt=SYSTEM_PROMPT,
            max_turns=int(cfg["max_turns"]),
            max_total_tokens=max_total_tokens,
            sampling=sampling,
            curriculum=curriculum,
        )
        objective = 0.7 * metrics["achieved_rate"] + 0.3 * metrics["mean_progress"]
        with open(output_dir / "trial_metrics.json", "w") as f:
            json.dump({"cfg": cfg, "metrics": metrics, "objective": objective}, f, indent=2)
        logger.info("Trial done: metrics=%s objective=%.4f", metrics, objective)
        return float(objective)
    finally:
        if trainer is not None:
            try:
                del trainer
            except Exception:
                pass
        if model is not None:
            free_model(model)
        if pool is not None:
            try:
                pool.close()
            except Exception:
                logger.exception("Pool close error during trial cleanup")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _lora_config(cfg: dict[str, Any]) -> LoraConfig:
    r = int(cfg["lora_r"])
    alpha_mul = int(cfg["lora_alpha_mul"])
    return LoraConfig(
        r=r,
        lora_alpha=r * alpha_mul,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


# ============================================================
# Main training entrypoint (single training pass)
# ============================================================


def run_training(
    cfg: dict[str, Any],
    *,
    base_model: str,
    sft_adapter: Optional[str],
    env_url: str,
    output_dir: Path,
    max_steps: int,
    max_total_tokens: int = 4096,
    max_completion_length: int = 256,
    max_prompt_length: int = 2048,
    push_to_hub: bool = False,
    hub_repo: Optional[str] = None,
) -> Path:
    """Run a full GRPO training pass with the supplied config dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run_training cfg=%s -> %s", cfg, output_dir)

    model, tokenizer = load_policy(base_model, sft_adapter, trainable=True)
    pool = MultiTurnEnvPool(env_url, size=int(cfg["num_generations"]))
    pool.start()

    curriculum = Curriculum()
    sampling = SamplingCfg(
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        max_new_tokens=max_completion_length,
        max_prompt_length=max_prompt_length,
    )
    ep_logger = EpisodeLogger(output_dir)
    rollout_func = make_rollout_func(
        curriculum=curriculum,
        pool=pool,
        model=model,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_turns=int(cfg["max_turns"]),
        max_total_tokens=max_total_tokens,
        sampling=sampling,
        log_episode=lambda task, results: ep_logger.log(task, results, curriculum),
    )

    dataset = _build_dummy_dataset(max_steps * int(cfg["num_generations"]))
    grpo_cfg = _build_grpo_config(
        output_dir=output_dir,
        cfg=cfg,
        max_steps=max_steps,
        max_completion_length=max_completion_length,
        max_prompt_length=max_prompt_length,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_task, reward_achieved, reward_progress],
        train_dataset=dataset,
        args=grpo_cfg,
        rollout_func=rollout_func,
        peft_config=None if sft_adapter else _lora_config(cfg),
    )

    try:
        trainer.train()
    finally:
        try:
            pool.close()
        except Exception:
            logger.exception("Pool close error after training")
        try:
            plot_rewards(ep_logger.csv_path, output_dir / "reward_plot.png")
        except Exception as e:
            logger.warning("plot_rewards failed: %s", e)

    trainer.save_model(str(output_dir))
    logger.info("Adapter saved to %s", output_dir)

    if push_to_hub and hub_repo:
        trainer.push_to_hub(repo_id=hub_repo)
        logger.info("Adapter pushed to https://huggingface.co/%s", hub_repo)

    return output_dir


# ============================================================
# CLI
# ============================================================


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "optuna", "full"], default="train")
    p.add_argument("--base-model", default=PolicySpec.base_model)
    p.add_argument("--sft-adapter", default=PolicySpec.sft_adapter,
                   help="HF repo id of the SFT adapter (use empty string to disable)")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--output-dir", default=None)

    # Train-mode hyperparams (mirror DEFAULT_CFG keys)
    p.add_argument("--num-generations", type=int, default=DEFAULT_CFG["num_generations"])
    p.add_argument("--max-turns", type=int, default=DEFAULT_CFG["max_turns"])
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_CFG["learning_rate"])
    p.add_argument("--beta", type=float, default=DEFAULT_CFG["beta"])
    p.add_argument("--temperature", type=float, default=DEFAULT_CFG["temperature"])
    p.add_argument("--top-p", type=float, default=DEFAULT_CFG["top_p"])
    p.add_argument("--lora-r", type=int, default=DEFAULT_CFG["lora_r"])
    p.add_argument("--lora-alpha-mul", type=int, default=DEFAULT_CFG["lora_alpha_mul"])
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--max-total-tokens", type=int, default=4096)

    # Optuna-specific
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--trial-max-steps", type=int, default=30)
    p.add_argument("--val-tasks-per-tier", type=int, default=2)

    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo", default=None)
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs") / f"aws-rl-grpo-{ts}"


def _cli_cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "lora_r": args.lora_r,
        "lora_alpha_mul": args.lora_alpha_mul,
        "max_turns": args.max_turns,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_adapter = args.sft_adapter or None

    if args.mode in ("optuna", "full"):
        val_ids = pick_validation_task_ids(k_per_tier=args.val_tasks_per_tier)
        with open(output_dir / "val_task_ids.json", "w") as f:
            json.dump(val_ids, f)
        study = optuna_search(
            n_trials=args.n_trials,
            trial_max_steps=args.trial_max_steps,
            val_task_ids=val_ids,
            base_model=args.base_model,
            sft_adapter=sft_adapter,
            env_url=args.env_url,
            output_dir=output_dir,
            max_total_tokens=args.max_total_tokens,
            max_completion_length=args.max_completion_length,
            max_prompt_length=args.max_prompt_length,
        )
        if args.mode == "optuna":
            return
        cfg = {**DEFAULT_CFG, **dict(study.best_params)}
    else:
        cfg = _cli_cfg(args)

    run_training(
        cfg,
        base_model=args.base_model,
        sft_adapter=sft_adapter,
        env_url=args.env_url,
        output_dir=output_dir,
        max_steps=args.max_steps,
        max_total_tokens=args.max_total_tokens,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()
