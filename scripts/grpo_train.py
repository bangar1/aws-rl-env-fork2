"""End-to-end GRPO training on AWS RL Env, driven from Google Colab.

This script is pedagogical — it shows how the moving pieces connect:

    [Central Curriculum] --picks task_id--> [G parallel rollouts via GrpoPool]
                                                     |
                                                     v
                          [Per-rollout trajectory of (prompt, action, reward)]
                                                     |
                                                     v
                          [Group-normalized advantages: A_i = (R_i - mean) / std]
                                                     |
                                                     v
                          [PPO-style policy-gradient loss on logprobs]
                                                     |
                                                     v
                                             [Optimizer step]

Why this is "GRPO" and not vanilla REINFORCE:
    GRPO (Group Relative Policy Optimization, DeepSeek) replaces the value
    baseline with the **group mean** of rewards. For each task we sample G
    trajectories; the advantage for rollout i is A_i = (R_i - mean(R)) / std(R).
    This is variance-reduced and critic-free — perfect for our env where G=8.

Requirements (install in the Colab cell before running):
    !pip install unsloth trl torch transformers accelerate bitsandbytes httpx websockets

Prerequisites:
    The RL env server must be running somewhere Colab can reach, with
    AWS_RL_ENV_POOL_SIZE=8 set. Easiest:
        docker run -p 8000:8000 -e AWS_RL_ENV_POOL_SIZE=8 aws-rl-env:latest
    And expose port 8000 via `cloudflared tunnel` or `ngrok http 8000`.

    Set BASE_URL below to the public URL of that tunnel.

Run:
    python scripts/grpo_train.py
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from client import AwsRlEnv
from models import AwsRlAction, Task
from scripts.grpo_pool import GrpoPool
from server.services.curriculum import Curriculum

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — tune for your setup
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("AWS_RL_ENV_BASE_URL", "http://localhost:8000")
GROUP_SIZE = int(os.getenv("GRPO_GROUP_SIZE", "8"))  # G in GRPO
NUM_GRPO_STEPS = int(os.getenv("GRPO_NUM_STEPS", "100"))  # outer training steps
MAX_EPISODE_STEPS = int(
    os.getenv("GRPO_MAX_STEPS", "15")
)  # per-rollout step cap (matches MAX_STEPS in env)
LEARNING_RATE = float(os.getenv("GRPO_LR", "5e-6"))
KL_COEFF = float(os.getenv("GRPO_KL", "0.04"))  # KL penalty vs reference model
CLIP_EPS = float(os.getenv("GRPO_CLIP", "0.2"))  # PPO clip for stability
TEMPERATURE = float(os.getenv("GRPO_TEMP", "0.9"))
MAX_NEW_TOKENS = int(os.getenv("GRPO_MAX_NEW", "96"))  # per model generation
MODEL_NAME = os.getenv("GRPO_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


# ---------------------------------------------------------------------------
# Model loading (Unsloth — 4-bit LoRA, Colab-friendly)
# ---------------------------------------------------------------------------


def load_model_and_tokenizer():
    """Load a 4-bit LoRA-wrapped model via Unsloth.

    Unsloth is a drop-in replacement for transformers that ~2x speeds up
    fine-tuning on a single GPU and fits a 1.5B model on a free Colab T4.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=4096,
        load_in_4bit=True,
    )
    # LoRA wrapping — only these params receive gradients
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    FastLanguageModel.for_training(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt construction & action extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AWS SRE agent. You operate a simulated AWS cloud by \
emitting one AWS CLI command at a time. You will see the task description and the most \
recent command output, then reply with EXACTLY ONE AWS CLI command on a single line \
starting with 'aws '. No explanation, no markdown, no quotes — just the command."""


def build_prompt(tokenizer, task: Task, history: List[Tuple[str, str]]) -> str:
    """Build a chat prompt from the task + command/output history."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TASK: {task.description}"},
    ]
    for cmd, out in history[-4:]:  # keep last 4 turns to fit context
        messages.append({"role": "assistant", "content": cmd})
        messages.append({"role": "user", "content": f"OUTPUT:\n{out[:400]}"})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_command(raw: str) -> str:
    """Pull the first `aws …` line out of the model's raw decoded output."""
    for line in raw.splitlines():
        line = line.strip().strip("`").strip()
        if line.startswith("aws "):
            return line
    return "aws help"  # safe fallback so env always accepts the command


# ---------------------------------------------------------------------------
# Rollout — one trajectory, one env, one task
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One step of a trajectory. `prompt_ids` + `action_ids` are what we backprop on."""

    prompt_ids: torch.Tensor  # shape [prompt_len]
    action_ids: torch.Tensor  # shape [action_len]
    logprob_sum: (
        torch.Tensor
    )  # scalar — sum of model logprobs over action_ids at sample time
    reward: float


@dataclass
class Trajectory:
    steps: List[Step]
    total_reward: float


@torch.no_grad()
def generate_action(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one command from the model; return (text, prompt_ids, action_ids, logprob_sum)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"][0]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    full_ids = out.sequences[0]
    action_ids = full_ids[prompt_ids.size(0) :]
    # Gather per-token logprobs of the sampled tokens (at generation time)
    if out.scores:
        logits = torch.stack(out.scores, dim=0)  # [T, 1, V]
        logprobs = torch.log_softmax(logits, dim=-1)[:, 0, :]
        token_lp = logprobs.gather(1, action_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sum = token_lp.sum()
    else:
        logprob_sum = torch.tensor(0.0, device=device)
    text = tokenizer.decode(action_ids, skip_special_tokens=True)
    return text, prompt_ids, action_ids, logprob_sum


async def run_single_rollout(
    env: AwsRlEnv,
    task: Task,
    model,
    tokenizer,
    device: torch.device,
) -> Trajectory:
    """Drive one env through up to MAX_EPISODE_STEPS, recording every step."""
    result = await env.reset(task_id=task.task_id)
    history: List[Tuple[str, str]] = []
    steps: List[Step] = []
    total_reward = 0.0

    for _ in range(MAX_EPISODE_STEPS):
        prompt = build_prompt(tokenizer, task, history)
        raw, prompt_ids, action_ids, logprob_sum = generate_action(
            model, tokenizer, prompt, device
        )
        command = extract_command(raw)
        result = await env.step(AwsRlAction(command=command))
        reward = float(result.reward)
        total_reward += reward
        steps.append(
            Step(
                prompt_ids=prompt_ids.cpu(),
                action_ids=action_ids.cpu(),
                logprob_sum=logprob_sum.detach().cpu(),
                reward=reward,
            )
        )
        history.append((command, result.observation.command_output or ""))
        if result.done:
            break

    return Trajectory(steps=steps, total_reward=total_reward)


# ---------------------------------------------------------------------------
# GRPO loss — group-normalized advantages, PPO-style clipped ratio
# ---------------------------------------------------------------------------


def compute_group_advantages(rewards: List[float]) -> List[float]:
    """Core GRPO step: subtract the group mean and divide by group std.

    A_i = (R_i - mean(R_1..G)) / (std(R_1..G) + eps)

    This makes the "baseline" the group's own performance — no value network
    needed. If all G rollouts tied, advantages are zero (no signal, correct).
    """
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var) + 1e-8
    return [(r - mean) / std for r in rewards]


def logprob_under_current_model(
    model, tokenizer, step: Step, device: torch.device
) -> torch.Tensor:
    """Re-score the sampled action under the CURRENT policy (for gradient).

    At rollout time we recorded the old policy's logprob_sum. To get a
    differentiable ratio we have to recompute it now with the current weights.
    """
    full = torch.cat([step.prompt_ids, step.action_ids]).unsqueeze(0).to(device)
    attn = torch.ones_like(full)
    outputs = model(input_ids=full, attention_mask=attn)
    logits = outputs.logits[0, :-1, :]  # predict next token
    targets = full[0, 1:]
    prompt_len = step.prompt_ids.size(0)
    # Only the action tokens contribute to the loss
    action_logits = logits[prompt_len - 1 : prompt_len - 1 + step.action_ids.size(0)]
    action_targets = targets[prompt_len - 1 : prompt_len - 1 + step.action_ids.size(0)]
    logp = F.log_softmax(action_logits, dim=-1)
    token_logp = logp.gather(1, action_targets.unsqueeze(-1)).squeeze(-1)
    return token_logp.sum()


def grpo_loss(
    model,
    tokenizer,
    trajectories: List[Trajectory],
    device: torch.device,
) -> torch.Tensor:
    """GRPO objective: maximize clipped advantage-weighted logprob ratio.

    loss = -mean_i [ min(ratio_i * A_i, clip(ratio_i, 1-eps, 1+eps) * A_i) ]
    """
    rewards = [t.total_reward for t in trajectories]
    advantages = compute_group_advantages(rewards)

    losses: List[torch.Tensor] = []
    for traj, adv in zip(trajectories, advantages):
        if not traj.steps:
            continue
        adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
        for step in traj.steps:
            new_logp = logprob_under_current_model(model, tokenizer, step, device)
            old_logp = step.logprob_sum.to(device)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            losses.append(-torch.min(unclipped, clipped))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


async def train() -> None:
    model, tokenizer = load_model_and_tokenizer()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    curriculum = Curriculum()
    async with GrpoPool(
        base_url=BASE_URL, size=GROUP_SIZE, curriculum=curriculum
    ) as pool:
        logger.info("Connected pool of %d envs against %s", GROUP_SIZE, BASE_URL)

        for step_idx in range(NUM_GRPO_STEPS):
            # 1) central curriculum picks ONE task for the whole group
            task = curriculum.next_task()
            logger.info(
                "[step %d/%d] task_id=%d tier=%s",
                step_idx + 1,
                NUM_GRPO_STEPS,
                task.task_id,
                task.difficulty.value,
            )

            # 2) launch G parallel rollouts, all on the same task_id
            rollout_coros = [
                run_single_rollout(e, task, model, tokenizer, device) for e in pool.envs
            ]
            trajectories = await asyncio.gather(*rollout_coros)
            rewards = [t.total_reward for t in trajectories]
            logger.info(
                "  rewards: min=%.3f mean=%.3f max=%.3f",
                min(rewards),
                sum(rewards) / len(rewards),
                max(rewards),
            )

            # 3) GRPO loss + update
            model.train()
            loss = grpo_loss(model, tokenizer, trajectories, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            logger.info("  loss=%.4f", loss.item())

            # 4) feed result back to curriculum (one record per group, not per rollout)
            pool.record_group_result(task, rewards)

    # Save LoRA adapter
    output_dir = os.getenv("GRPO_OUTPUT_DIR", "./grpo_lora_out")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Saved LoRA adapter to %s", output_dir)


if __name__ == "__main__":
    asyncio.run(train())
