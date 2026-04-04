"""Task grading engine — evaluates task completion and computes shaped rewards.

All rewards are in the [0.0, 1.0] range. Only full task completion yields 1.0.
Includes anti-reward-hacking defenses.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from models import SuccessCriteria, Task
from server.services.aws_backend import AwsBackend
from server.services.episode_tracker import EpisodeTracker, StepRecord
from server.services.resource_verifier import ResourceVerifier

logger = logging.getLogger(__name__)


class GradeResult(BaseModel):
    """Outcome of grading a single step."""

    task_achieved: bool = False
    partial_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    reward: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class TaskGrader:
    """Evaluates task completion and computes shaped rewards.

    Dispatches to different grading strategies based on which fields
    are populated on the task's ``SuccessCriteria``.
    """

    def __init__(self, backend: AwsBackend) -> None:
        self._verifier = ResourceVerifier(backend)

    def grade(
        self,
        task: Task,
        tracker: EpisodeTracker,
        latest_step: StepRecord,
        chaos_occurred: bool = False,
    ) -> GradeResult:
        criteria = task.success_criteria

        # Dispatch based on populated criteria fields
        if criteria.state_checks:
            result = self._grade_state_checks(criteria, tracker)
        elif criteria.steps:
            result = self._grade_multi_step(criteria, tracker)
        elif criteria.resource_exists is not None:
            result = self._grade_resource_creation(criteria, latest_step)
        elif criteria.command_contains is not None:
            result = self._grade_command_match(criteria, latest_step)
        else:
            result = GradeResult(reason="no recognised success_criteria fields")

        # Compute shaped reward
        result.reward = self._compute_reward(
            result, latest_step, tracker, chaos_occurred
        )

        # Update tracker's previous progress (monotonic — never decrease)
        if result.partial_progress > tracker.previous_progress:
            tracker.previous_progress = result.partial_progress

        return result

    # -- Grading strategies ---------------------------------------------------

    def _grade_command_match(
        self, criteria: SuccessCriteria, latest_step: StepRecord
    ) -> GradeResult:
        """Warmup: check the latest command matches expected service + operation."""
        cmd = latest_step.command.lower()
        contains = (criteria.command_contains or "").lower()
        operation = (criteria.operation or "").lower()

        contains_ok = contains != "" and contains in cmd
        operation_ok = operation != "" and operation in cmd
        succeeded = latest_step.success
        achieved = contains_ok and operation_ok and succeeded

        return GradeResult(
            task_achieved=achieved,
            partial_progress=1.0 if achieved else 0.0,
            reason=(
                f"command_match: contains={contains_ok}, "
                f"op={operation_ok}, success={succeeded}"
            ),
        )

    def _grade_resource_creation(
        self,
        criteria: SuccessCriteria,
        latest_step: StepRecord,
    ) -> GradeResult:
        """Beginner: verify the resource actually exists in MiniStack."""
        re_spec = criteria.resource_exists
        assert re_spec is not None
        service = re_spec.service
        name = re_spec.name

        exists = self._verifier.resource_exists(service, name)

        # Command matching gives partial credit (0.5)
        contains = (criteria.command_contains or "").lower()
        operation = (criteria.operation or "").lower()
        cmd = latest_step.command.lower()
        cmd_ok = contains in cmd and operation in cmd and latest_step.success

        if exists:
            progress = 1.0
        elif cmd_ok:
            progress = 0.5
        else:
            progress = 0.0

        return GradeResult(
            task_achieved=exists,
            partial_progress=progress,
            reason=(
                f"resource_creation: exists={exists}, "
                f"cmd_ok={cmd_ok}, service={service}, name={name}"
            ),
        )

    def _grade_multi_step(
        self, criteria: SuccessCriteria, tracker: EpisodeTracker
    ) -> GradeResult:
        """Intermediate/Advanced: check ordered step completion."""
        steps = criteria.steps
        if not steps:
            return GradeResult(reason="empty steps list")

        completed = 0
        for step in steps:
            if tracker.has_executed_operation(step.operation, step.resource):
                completed += 1
            else:
                break  # ordered — stop at first incomplete step

        total = len(steps)
        progress = completed / total if total > 0 else 0.0

        # For advanced tasks with services requirement, also check services
        services_required = criteria.services
        services_met = all(tracker.has_used_service(svc) for svc in services_required)

        achieved = completed == total and (not services_required or services_met)

        return GradeResult(
            task_achieved=achieved,
            partial_progress=progress,
            reason=(
                f"multi_step: {completed}/{total} steps, "
                f"services_met={services_met if services_required else 'n/a'}"
            ),
        )

    def _grade_state_checks(
        self, criteria: SuccessCriteria, tracker: EpisodeTracker
    ) -> GradeResult:
        """Expert/SRE: verify end-state via arbitrary commands.

        state_checks are the source of truth for task completion.
        steps (if present) provide partial progress signals only.
        """
        state_checks = criteria.state_checks
        steps = criteria.steps

        # Evaluate state checks (ground truth)
        checks_passed = 0
        for check in state_checks:
            check_dict = check.model_dump(exclude_none=True)
            if self._verifier.check_state(check_dict):
                checks_passed += 1

        total_checks = len(state_checks)
        all_checks_pass = checks_passed == total_checks and total_checks > 0

        # Evaluate steps for partial progress signal
        steps_completed = 0
        for step in steps:
            if tracker.has_executed_operation(step.operation, step.resource):
                steps_completed += 1
            else:
                break

        # Progress combines steps (for dense signal) and state checks
        total_steps = len(steps)
        if total_steps > 0:
            step_progress = steps_completed / total_steps
        else:
            step_progress = 0.0

        # Weight: steps give up to 0.7, state checks give the remaining 0.3
        if total_checks > 0:
            check_progress = checks_passed / total_checks
            progress = step_progress * 0.7 + check_progress * 0.3
        else:
            progress = step_progress

        # Check services requirement
        services_required = criteria.services
        services_met = all(tracker.has_used_service(svc) for svc in services_required)

        # Task achieved only when ALL state checks pass
        achieved = all_checks_pass and (not services_required or services_met)

        return GradeResult(
            task_achieved=achieved,
            partial_progress=min(progress, 1.0),
            reason=(
                f"state_checks: {checks_passed}/{total_checks} passed, "
                f"steps: {steps_completed}/{total_steps}, "
                f"services_met={services_met if services_required else 'n/a'}"
            ),
        )

    # -- Reward shaping -------------------------------------------------------

    def _compute_reward(
        self,
        result: GradeResult,
        latest_step: StepRecord,
        tracker: EpisodeTracker,
        chaos_occurred: bool = False,
    ) -> float:
        """Compute a shaped reward in [0.0, 1.05]."""
        if result.task_achieved:
            return 1.05 if chaos_occurred else 1.0

        # Base: partial progress scaled to 0.0–0.8 range
        progress_reward = result.partial_progress * 0.8

        # Bonus for advancing progress (dense signal)
        progress_delta = result.partial_progress - tracker.previous_progress
        if progress_delta > 0:
            progress_reward += 0.1

        # Penalty for failed commands
        if not latest_step.success:
            progress_reward *= 0.5

        # Rollback penalty: wasteful create→delete pairs
        progress_reward -= 0.1 * tracker.detect_rollbacks()

        # Idempotency bonus: graceful "already exists" handling
        progress_reward += 0.02 * tracker.detect_idempotent_retries()

        # Clamp to [0.0, 0.99] — never reach 1.0 without achieving
        return min(max(progress_reward, 0.0), 0.99)
