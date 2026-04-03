"""Curriculum manager for progressive LLM training in the AWS RL environment."""

import logging
from pathlib import Path

import yaml

from models import Task, TaskDifficulty, TaskID

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).parent / "tasks"

# Map YAML filenames to difficulty tiers
_TIER_FILES: dict[TaskDifficulty, str] = {
    TaskDifficulty.WARMUP: "warmup.yaml",
    TaskDifficulty.BEGINNER: "beginner.yaml",
    TaskDifficulty.INTERMEDIATE: "intermediate.yaml",
    TaskDifficulty.ADVANCED: "advanced.yaml",
    TaskDifficulty.EXPERT: "expert.yaml",
}


def load_tier(difficulty: TaskDifficulty, tasks_dir: Path = TASKS_DIR) -> list[Task]:
    """Load tasks for a single difficulty tier from its YAML file.

    Args:
        difficulty: The tier to load.
        tasks_dir: Directory containing per-tier YAML files.

    Returns:
        List of Task objects for the requested tier.
    """
    filename = _TIER_FILES.get(difficulty)
    if filename is None:
        logger.warning("No file mapping for difficulty: %s", difficulty.value)
        return []

    filepath = tasks_dir / filename
    if not filepath.exists():
        logger.warning("Task file not found: %s", filepath)
        return []

    with open(filepath) as f:
        entries = yaml.safe_load(f) or []

    tasks = [
        Task(
            task_id=TaskID(entry["task_id"]),
            difficulty=difficulty,
            description=entry["description"],
            success_criteria=entry.get("success_criteria", {}),
        )
        for entry in entries
    ]
    logger.info("Loaded %d %s tasks from %s", len(tasks), difficulty.value, filepath.name)
    return tasks


class Curriculum:
    """Manages progressive task assignment for LLM training.

    The curriculum tracks the agent's performance across difficulty tiers
    and promotes to harder tasks once the agent demonstrates competence
    at the current level. Only the current tier's tasks are held in memory;
    previous tiers are unloaded on promotion.
    """

    def __init__(
        self,
        promotion_threshold: float = 0.7,
        warmup_episodes: int = 5,
        tasks_dir: Path = TASKS_DIR,
    ) -> None:
        """Initialise the curriculum starting at the warmup stage.

        Args:
            promotion_threshold: Success rate (0-1) needed to advance to
                the next difficulty tier.
            warmup_episodes: Minimum number of warmup episodes before the
                agent can be promoted.
            tasks_dir: Directory containing per-tier YAML task files.
        """
        self._promotion_threshold = promotion_threshold
        self._warmup_episodes = warmup_episodes
        self._tasks_dir = tasks_dir

        # Ordered difficulty progression
        self._levels = list(TaskDifficulty)

        # Tracking state
        self._current_level_idx: int = 0
        self._task_idx: int = 0
        self._episode_results: dict[TaskDifficulty, list[bool]] = {
            level: [] for level in self._levels
        }

        # Load only the starting tier
        self._current_tasks: list[Task] = load_tier(self.current_difficulty, self._tasks_dir)

        logger.info(
            "Curriculum initialised — starting at %s with %d tasks",
            self.current_difficulty.value,
            len(self._current_tasks),
        )

    # -- public API ----------------------------------------------------------

    @property
    def current_difficulty(self) -> TaskDifficulty:
        return self._levels[self._current_level_idx]

    @property
    def current_level_success_rate(self) -> float:
        results = self._episode_results[self.current_difficulty]
        if not results:
            return 0.0
        return sum(results) / len(results)

    @property
    def is_warmup(self) -> bool:
        return self.current_difficulty == TaskDifficulty.WARMUP

    def next_task(self) -> Task:
        """Return the next task the agent should attempt.

        Cycles through tasks at the current difficulty level, round-robin.
        """
        if not self._current_tasks:
            self._current_tasks = load_tier(self.current_difficulty, self._tasks_dir)

        task = self._current_tasks[self._task_idx % len(self._current_tasks)]
        self._task_idx += 1
        return task

    def record_result(self, task: Task, achieved: bool) -> None:
        """Record the outcome of an episode and check for promotion.

        Args:
            task: The task that was attempted.
            achieved: Whether the agent completed the task successfully.
        """
        self._episode_results[task.difficulty].append(achieved)
        logger.info(
            "Episode result: task=%d difficulty=%s achieved=%s rate=%.2f",
            task.task_id,
            task.difficulty.value,
            achieved,
            self.current_level_success_rate,
        )
        self._maybe_promote()

    def reset(self) -> None:
        """Reset curriculum back to warmup (useful for a full training restart)."""
        self._current_level_idx = 0
        self._task_idx = 0
        self._episode_results = {level: [] for level in self._levels}
        self._current_tasks = load_tier(self.current_difficulty, self._tasks_dir)
        logger.info("Curriculum reset to %s", self.current_difficulty.value)

    # -- internals -----------------------------------------------------------

    def _maybe_promote(self) -> None:
        """Promote to the next difficulty tier if the threshold is met."""
        results = self._episode_results[self.current_difficulty]

        min_episodes = (
            self._warmup_episodes if self.is_warmup else 3
        )
        if len(results) < min_episodes:
            return

        if self.current_level_success_rate < self._promotion_threshold:
            return

        if self._current_level_idx >= len(self._levels) - 1:
            logger.info("Agent has reached the highest difficulty tier")
            return

        prev_rate = self.current_level_success_rate
        self._current_level_idx += 1
        self._task_idx = 0
        self._current_tasks = load_tier(self.current_difficulty, self._tasks_dir)
        logger.info(
            "Promoted to %s (previous rate: %.2f)",
            self.current_difficulty.value,
            prev_rate,
        )
