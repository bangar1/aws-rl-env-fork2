"""
Configuration Drift Engine.

Randomly applies a subset of a task's possible mutations after the correct
state has been provisioned. This forces the agent to audit and discover
which resources drifted rather than memorising a fixed solution path.
"""

from __future__ import annotations

import logging
import random

from models import Task
from server.services.aws_backend import AwsBackend

logger = logging.getLogger(__name__)

# Default range for how many drifts to apply (inclusive).
_MIN_DRIFTS = 2
_MAX_DRIFTS = 3


class DriftEngine:
    """Selects and applies random configuration drifts for a task."""

    def __init__(self, backend: AwsBackend) -> None:
        self._backend = backend

    def apply_drift(self, task: Task) -> list[str]:
        """Randomly select and execute K of N possible drifts.

        Args:
            task: A task whose ``possible_drifts`` list defines the
                candidate mutations.

        Returns:
            Human-readable descriptions of the drifts that were applied
            (empty list if none).
        """
        if not task.possible_drifts:
            return []

        pool = task.possible_drifts
        k = self._pick_count(len(pool))
        selected = random.sample(pool, k)

        applied: list[str] = []
        for drift in selected:
            success, _stdout, stderr = self._backend.execute_command(drift.command)
            label = drift.description or drift.command
            if success:
                logger.info("Drift applied: %s", label)
                applied.append(label)
            else:
                logger.warning("Drift command failed: %s — %s", drift.command, stderr)

        return applied

    @staticmethod
    def _pick_count(pool_size: int) -> int:
        """Determine how many drifts to apply given the pool size."""
        if pool_size <= 1:
            return pool_size
        lo = min(_MIN_DRIFTS, pool_size)
        hi = min(_MAX_DRIFTS, pool_size)
        return random.randint(lo, hi)
