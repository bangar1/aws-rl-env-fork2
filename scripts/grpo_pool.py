"""GRPO rollout pool helper — designed to run from a Google Colab notebook.

Opens N persistent WebSocket sessions against a single server deployed with
AWS_RL_ENV_POOL_SIZE=N. All rollouts in a group share the same task (picked by
one central Curriculum) and run concurrently via asyncio.gather.

Usage (Colab cell):
    from scripts.grpo_pool import GrpoPool

    async def rollout(env, task):
        res = await env.reset(task=task)
        done = False
        total = 0.0
        while not done:
            action = AwsRlAction(command=policy(res.observation))
            res = await env.step(action)
            total += res.reward
            done = res.done
        return total

    async with GrpoPool(base_url="https://tunnel.example.com", size=8) as pool:
        for _ in range(num_grpo_steps):
            task = pool.curriculum.next_task()
            rewards = await pool.run_group(lambda e: rollout(e, task))
            pool.record_group_result(task, rewards)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, List, Optional, Sequence

from client import AwsRlEnv
from models import Task
from server.services.curriculum import Curriculum

logger = logging.getLogger(__name__)


class GrpoPool:
    """Manages N AwsRlEnv clients against a pooled server for GRPO rollouts."""

    def __init__(
        self,
        base_url: str,
        size: int = 8,
        curriculum: Optional[Curriculum] = None,
    ) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        self.base_url = base_url
        self.size = size
        self.curriculum = curriculum or Curriculum()
        self.envs: List[AwsRlEnv] = []

    async def connect(self) -> None:
        """Open N persistent WebSocket sessions. Each binds to its own MiniStack.

        All-or-nothing: if any single session fails to connect, every already
        opened session is closed before re-raising, so the server's pool does
        not leak slots and callers never see a half-initialised pool.
        """
        if self.envs:
            return
        envs = [AwsRlEnv(base_url=self.base_url) for _ in range(self.size)]
        try:
            await asyncio.gather(*(e.connect() for e in envs))
        except BaseException:
            # Roll back: close every env (successful or not). return_exceptions
            # so a close() failure doesn't mask the original connect error.
            await asyncio.gather(
                *(e.close() for e in envs),
                return_exceptions=True,
            )
            raise
        # Only publish the pool after the entire group connected successfully.
        self.envs = envs
        logger.info(
            "GrpoPool connected: %d sessions against %s", self.size, self.base_url
        )

    async def close(self) -> None:
        """Close all WebSocket sessions. Server releases MiniStacks back to pool."""
        if not self.envs:
            return
        await asyncio.gather(*(e.close() for e in self.envs), return_exceptions=True)
        self.envs = []

    async def reset_group(self, task: Task) -> None:
        """Reset all N envs onto the same task. Runs concurrently.

        The full Task is serialised to the server, so envs do not have to
        look the task up through their own curriculum.
        """
        await asyncio.gather(*(e.reset(task=task) for e in self.envs))

    async def run_group(
        self,
        rollout_fn: Callable[[AwsRlEnv], Awaitable[float]],
    ) -> List[float]:
        """Run `rollout_fn` on each of the N envs concurrently, return rewards.

        The caller is responsible for calling reset_group() beforehand (or
        doing the reset inside rollout_fn with the same task_id).
        """
        return list(await asyncio.gather(*(rollout_fn(e) for e in self.envs)))

    def record_group_result(
        self,
        task: Task,
        rewards: Sequence[float],
        success_threshold: float = 0.99,
    ) -> None:
        """Feed one group-level result back to the central curriculum.

        A group is considered "achieved" if at least one rollout scored above
        the success threshold. The recorded reward is the group mean.
        """
        achieved = any(r >= success_threshold for r in rewards)
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        self.curriculum.record_result(task, achieved=achieved, reward=mean_reward)

    @asynccontextmanager
    async def session(self):
        try:
            await self.connect()
            yield self
        finally:
            await self.close()

    async def __aenter__(self) -> "GrpoPool":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
