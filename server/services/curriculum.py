"""Curriculum manager for progressive LLM training in the AWS RL environment.

Training flow:
  1. Agent starts at the warmup tier with simple listing tasks.
  2. A priority queue selects the next task based on weakness, novelty,
     spaced repetition, and recency — replacing blind round-robin.
  3. Per-task mastery tracking graduates individual tasks once the agent
     demonstrates sustained competence.
  4. Graduated tasks resurface via spaced repetition at exponentially
     increasing intervals to prevent catastrophic forgetting.
  5. Fast-track promotion lets strong agents skip minimum episode waits.
  6. Exponential decay on history ensures recent results matter more.
"""

import heapq
import logging
import random
from collections import defaultdict
from pathlib import Path

import yaml

from models import (
    SetupCommand,
    SpacedRepState,
    SuccessCriteria,
    Task,
    TaskDifficulty,
    TaskID,
    TierConfig,
)

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).parent / "tasks"

# ---------------------------------------------------------------------------
# Per-tier configuration
# ---------------------------------------------------------------------------

TIER_CONFIGS: dict[TaskDifficulty, TierConfig] = {
    TaskDifficulty.WARMUP: TierConfig(
        min_episodes=5,
        advance_rate=0.6,
        mastery_window=10,
        mastery_threshold=0.7,
        fast_track_rate=0.9,
    ),
    TaskDifficulty.BEGINNER: TierConfig(
        min_episodes=5,
        advance_rate=0.6,
        mastery_window=10,
        mastery_threshold=0.7,
        fast_track_rate=0.9,
    ),
    TaskDifficulty.INTERMEDIATE: TierConfig(
        min_episodes=8,
        advance_rate=0.65,
        mastery_window=10,
        mastery_threshold=0.7,
        fast_track_rate=0.9,
    ),
    TaskDifficulty.ADVANCED: TierConfig(
        min_episodes=10,
        advance_rate=0.7,
        mastery_window=10,
        mastery_threshold=0.7,
        fast_track_rate=0.9,
    ),
    TaskDifficulty.EXPERT: TierConfig(
        min_episodes=0,
        advance_rate=1.0,
        mastery_window=10,
        mastery_threshold=0.7,
        fast_track_rate=1.0,
    ),
}

# Map YAML filenames to difficulty tiers
_TIER_FILES: dict[TaskDifficulty, str] = {
    TaskDifficulty.WARMUP: "warmup.yaml",
    TaskDifficulty.BEGINNER: "beginner.yaml",
    TaskDifficulty.INTERMEDIATE: "intermediate.yaml",
    TaskDifficulty.ADVANCED: "advanced.yaml",
    TaskDifficulty.EXPERT: "expert.yaml",
}

# ---------------------------------------------------------------------------
# Priority score tuning constants
# ---------------------------------------------------------------------------

_NOVELTY_BONUS = 100  # untried tasks — explore first
_WEAKNESS_WEIGHT = 50  # multiplied by (1 - success_rate)
_SPACED_REP_BONUS = 30  # graduated task due for re-test
_RECENCY_PENALTY = 20  # attempted in last 2 episodes

# Exponential decay factor for weighted success rate
_DECAY_FACTOR = 0.85

# Minimum attempts before a task can be graduated
_MIN_ATTEMPTS_FOR_MASTERY = 3

# Fast-track requires at least this many episodes in the tier
_FAST_TRACK_MIN_EPISODES = 3


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_tier(difficulty: TaskDifficulty, tasks_dir: Path = TASKS_DIR) -> list[Task]:
    """Load tasks for a single difficulty tier from its YAML file."""
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
            success_criteria=SuccessCriteria(**entry.get("success_criteria", {})),
            setup_commands=[
                SetupCommand(command=cmd)
                if isinstance(cmd, str)
                else SetupCommand(**cmd)
                for cmd in entry.get("setup_commands", [])
            ],
        )
        for entry in entries
    ]
    logger.info(
        "Loaded %d %s tasks from %s", len(tasks), difficulty.value, filepath.name
    )
    return tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weighted_success_rate(results: list[bool], decay: float = _DECAY_FACTOR) -> float:
    """Compute success rate with exponential decay — recent results matter more."""
    if not results:
        return 0.0
    weights = [decay**i for i in range(len(results) - 1, -1, -1)]
    total_weight = sum(weights)
    return sum(w * float(r) for w, r in zip(weights, results)) / total_weight


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


class Curriculum:
    """Manages progressive task assignment with priority-queue-based selection.

    Features:
      - Priority queue task selection (novelty, weakness, spaced rep, recency)
      - Per-task mastery tracking with graduation
      - Spaced repetition for graduated tasks (prevents catastrophic forgetting)
      - Fast-track tier promotion for strong agents
      - Exponential decay on success history
      - Rich observability via get_stats()
    """

    def __init__(
        self,
        tier_configs: dict[TaskDifficulty, TierConfig] | None = None,
        tasks_dir: Path = TASKS_DIR,
    ) -> None:
        self._tier_configs = tier_configs or TIER_CONFIGS
        self._tasks_dir = tasks_dir

        # Ordered difficulty progression
        self._levels = list(TaskDifficulty)

        # Tier tracking
        self._current_level_idx: int = 0
        self._tier_episodes: int = 0
        self._tier_results: list[bool] = []  # results within current tier

        # Per-task tracking
        self._task_history: dict[TaskID, list[bool]] = defaultdict(list)
        self._task_attempt_count: dict[TaskID, int] = defaultdict(int)
        self._last_attempted_episode: dict[TaskID, int] = {}
        self._graduated_tasks: set[TaskID] = set()
        self._spaced_rep: dict[TaskID, SpacedRepState] = {}

        # Global counters
        self._episode_count: int = 0
        self._episode_rewards: list[float] = []

        # Load starting tier
        self._current_tasks: list[Task] = load_tier(
            self.current_difficulty, self._tasks_dir
        )
        self._task_map: dict[TaskID, Task] = {t.task_id: t for t in self._current_tasks}

        # Priority queue: list of (-score, random_tiebreaker, task_id)
        self._priority_queue: list[tuple[float, float, TaskID]] = []
        self._rebuild_priority_queue()

        logger.info(
            "Curriculum initialised — starting at %s with %d tasks",
            self.current_difficulty.value,
            len(self._current_tasks),
        )

    # -- Properties -----------------------------------------------------------

    @property
    def current_difficulty(self) -> TaskDifficulty:
        return self._levels[self._current_level_idx]

    @property
    def tier_config(self) -> TierConfig:
        return self._tier_configs[self.current_difficulty]

    @property
    def current_level_success_rate(self) -> float:
        return _weighted_success_rate(self._tier_results)

    @property
    def is_warmup(self) -> bool:
        return self.current_difficulty == TaskDifficulty.WARMUP

    # -- Public API -----------------------------------------------------------

    def next_task(self) -> Task:
        """Select the highest-priority task from the current tier."""
        if not self._current_tasks:
            self._current_tasks = load_tier(self.current_difficulty, self._tasks_dir)
            self._task_map = {t.task_id: t for t in self._current_tasks}
            self._rebuild_priority_queue()

        if not self._priority_queue:
            self._rebuild_priority_queue()

        # Pop highest priority (most negative = highest score)
        _, _, task_id = heapq.heappop(self._priority_queue)
        task = self._task_map[task_id]

        # If queue is now empty, rebuild for next call
        if not self._priority_queue:
            self._rebuild_priority_queue()

        return task

    def record_result(self, task: Task, achieved: bool, reward: float = 0.0) -> None:
        """Record episode outcome, update mastery, check promotion."""
        self._episode_count += 1
        self._tier_episodes += 1
        self._episode_rewards.append(reward)

        # Per-tier results
        self._tier_results.append(achieved)

        # Per-task results
        self._task_history[task.task_id].append(achieved)
        self._task_attempt_count[task.task_id] += 1
        self._last_attempted_episode[task.task_id] = self._episode_count

        # Check mastery
        self._check_mastery(task.task_id)

        # Check tier promotion
        self._maybe_promote()

        # Rebuild priority queue with updated scores
        self._rebuild_priority_queue()

        logger.info(
            "Episode %d: task=%d difficulty=%s achieved=%s tier_rate=%.2f",
            self._episode_count,
            task.task_id,
            task.difficulty.value,
            achieved,
            self.current_level_success_rate,
        )

    def reset(self) -> None:
        """Reset curriculum back to warmup (full training restart)."""
        self._current_level_idx = 0
        self._tier_episodes = 0
        self._tier_results.clear()
        self._task_history.clear()
        self._task_attempt_count.clear()
        self._last_attempted_episode.clear()
        self._graduated_tasks.clear()
        self._spaced_rep.clear()
        self._episode_count = 0
        self._episode_rewards.clear()
        self._current_tasks = load_tier(self.current_difficulty, self._tasks_dir)
        self._task_map = {t.task_id: t for t in self._current_tasks}
        self._rebuild_priority_queue()
        logger.info("Curriculum reset to %s", self.current_difficulty.value)

    # -- Observability --------------------------------------------------------

    def get_skill_profile(self) -> dict[TaskID, float]:
        """Weighted success rate per task over recent history."""
        config = self.tier_config
        return {
            task_id: round(_weighted_success_rate(results[-config.mastery_window :]), 2)
            for task_id, results in self._task_history.items()
            if results
        }

    def get_weak_spots(self) -> list[TaskID]:
        """Tasks in the current tier below mastery threshold."""
        config = self.tier_config
        profile = self.get_skill_profile()
        return [
            task_id
            for task_id in self._task_map
            if profile.get(task_id, 0.0) < config.mastery_threshold
            and task_id not in self._graduated_tasks
        ]

    def get_stats(self) -> dict:
        """Full curriculum state for logging/debugging."""
        return {
            "episode_count": self._episode_count,
            "tier": self.current_difficulty.value,
            "tier_episodes": self._tier_episodes,
            "tier_success_rate": round(self.current_level_success_rate, 3),
            "graduated_tasks": sorted(self._graduated_tasks),
            "weak_spots": self.get_weak_spots(),
            "skill_profile": self.get_skill_profile(),
            "spaced_rep_due": [
                int(tid) for tid in self._task_map if self._is_spaced_rep_due(tid)
            ],
            "avg_reward_last_10": round(
                sum(self._episode_rewards[-10:])
                / max(1, len(self._episode_rewards[-10:])),
                3,
            ),
        }

    # -- Priority queue -------------------------------------------------------

    def _compute_priority(self, task_id: TaskID) -> float:
        """Compute composite priority score for a task. Higher = selected sooner."""
        config = self.tier_config
        score = 0.0

        attempts = self._task_attempt_count.get(task_id, 0)

        # Novelty: never attempted → explore first
        if attempts == 0:
            score += _NOVELTY_BONUS
            return score  # no other signals available yet

        # Weakness: worse tasks get higher priority
        results = self._task_history.get(task_id, [])
        task_rate = _weighted_success_rate(results[-config.mastery_window :])
        score += _WEAKNESS_WEIGHT * (1.0 - task_rate)

        # Spaced repetition: graduated task due for re-test
        if task_id in self._graduated_tasks and self._is_spaced_rep_due(task_id):
            score += _SPACED_REP_BONUS

        # Recency penalty: attempted in last 2 episodes
        last_ep = self._last_attempted_episode.get(task_id, -100)
        if self._episode_count - last_ep <= 2:
            score -= _RECENCY_PENALTY

        return score

    def _rebuild_priority_queue(self) -> None:
        """Recompute priorities for all current-tier tasks and rebuild the heap."""
        self._priority_queue.clear()
        for task in self._current_tasks:
            score = self._compute_priority(task.task_id)
            # heapq is a min-heap, so negate score for max-priority-first
            # random tiebreaker prevents deterministic ordering among equal scores
            heapq.heappush(
                self._priority_queue,
                (-score, random.random(), task.task_id),
            )

    # -- Mastery & spaced repetition ------------------------------------------

    def _check_mastery(self, task_id: TaskID) -> None:
        """Check if a task should be graduated or un-graduated."""
        config = self.tier_config
        results = self._task_history.get(task_id, [])
        recent = results[-config.mastery_window :]

        if len(recent) < _MIN_ATTEMPTS_FOR_MASTERY:
            return

        rate = _weighted_success_rate(recent)

        if rate >= config.mastery_threshold:
            if task_id not in self._graduated_tasks:
                self._graduated_tasks.add(task_id)
                self._spaced_rep[task_id] = SpacedRepState(
                    interval=3,
                    last_graduated_episode=self._episode_count,
                )
                logger.info(
                    "Task %d GRADUATED (rate=%.2f) — scheduling spaced repetition",
                    task_id,
                    rate,
                )
        else:
            # Un-graduate if performance dropped
            if task_id in self._graduated_tasks:
                self._graduated_tasks.discard(task_id)
                self._spaced_rep.pop(task_id, None)
                logger.info(
                    "Task %d UN-GRADUATED (rate=%.2f) — resetting to active",
                    task_id,
                    rate,
                )

    def _is_spaced_rep_due(self, task_id: TaskID) -> bool:
        """Check if a graduated task is due for a re-test."""
        state = self._spaced_rep.get(task_id)
        if state is None:
            return False
        episodes_since = self._episode_count - state.last_graduated_episode
        return episodes_since >= state.interval

    def _advance_spaced_rep(self, task_id: TaskID) -> None:
        """Double the interval after a successful re-test."""
        state = self._spaced_rep.get(task_id)
        if state is not None:
            state.interval = min(state.interval * 2, 48)  # cap at 48 episodes
            state.last_graduated_episode = self._episode_count

    # -- Tier promotion -------------------------------------------------------

    def _maybe_promote(self) -> None:
        """Advance to the next difficulty tier if the agent is ready."""
        if self._current_level_idx >= len(self._levels) - 1:
            return  # already at max tier

        config = self.tier_config
        rate = self.current_level_success_rate

        # Fast-track: high success rate after minimum 3 episodes
        fast_track = (
            self._tier_episodes >= _FAST_TRACK_MIN_EPISODES
            and rate >= config.fast_track_rate
        )

        if not fast_track and self._tier_episodes < config.min_episodes:
            return

        if rate < config.advance_rate:
            return

        prev_tier = self.current_difficulty.value
        prev_rate = rate
        self._current_level_idx += 1
        self._tier_episodes = 0
        self._tier_results.clear()
        self._current_tasks = load_tier(self.current_difficulty, self._tasks_dir)
        self._task_map = {t.task_id: t for t in self._current_tasks}
        self._rebuild_priority_queue()
        logger.info(
            "PROMOTED from %s to %s (rate=%.2f%s)",
            prev_tier,
            self.current_difficulty.value,
            prev_rate,
            ", FAST-TRACK" if fast_track else "",
        )
