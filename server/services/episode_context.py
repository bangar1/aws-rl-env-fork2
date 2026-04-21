"""EpisodeContext — the per-episode truth for tier-dependent runtime behavior.

An `EpisodeContext` is the single source for everything about *this one episode*
that the env needs to know at runtime: which task is running, which tier's
dynamics apply (chaos probability, reported tier), and where — if anywhere —
a terminal result should be recorded.

Two construction sites encode the two episode-planning modes:

    # Local mode: env picks the task from its own curriculum. Terminal results
    # flow back to that same curriculum so local mastery/promotion tracking
    # continues to work.
    ctx = EpisodeContext.for_local(task=task, curriculum=self._curriculum)

    # Trainer mode: the trainer hands in a Task it picked from its own
    # (central) curriculum and owns result recording. The env must NOT mutate
    # any local tier-progression state for this episode.
    ctx = EpisodeContext.for_external(task=task)

With this split, `_sync_state`, chaos injection, and result recording all read
from `ctx` and no longer consult `self._curriculum.current_difficulty` — which
was the coupling that let external task injection corrupt local tier stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from models import Task, TaskDifficulty
from server.services.curriculum import TIER_CONFIGS

if TYPE_CHECKING:
    from server.services.curriculum import Curriculum


RecordResultFn = Callable[[Task, bool, float], None]


@dataclass(frozen=True)
class EpisodeContext:
    """Immutable per-episode context. `tier` and `chaos_probability` are
    derived from `task.difficulty` so they can never drift out of sync.
    """

    task: Task
    record_result: Optional[RecordResultFn]

    @property
    def tier(self) -> TaskDifficulty:
        return self.task.difficulty

    @property
    def chaos_probability(self) -> float:
        return TIER_CONFIGS[self.task.difficulty].chaos_probability

    @classmethod
    def for_local(cls, task: Task, curriculum: "Curriculum") -> "EpisodeContext":
        """Local mode — results flow back to the env's own curriculum."""
        return cls(task=task, record_result=curriculum.record_result)

    @classmethod
    def for_external(cls, task: Task) -> "EpisodeContext":
        """Trainer mode — terminal result recording is handled by the caller."""
        return cls(task=task, record_result=None)
