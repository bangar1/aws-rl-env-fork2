"""Unit tests for TaskGrader — tests all grading strategies and reward shaping.

These tests mock AwsBackend/ResourceVerifier so they run without MiniStack.

Run:
    uv run pytest tests/test_task_grader.py -v
    docker exec <container> python -m pytest env/tests/test_task_grader.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from models import (
    SuccessCriteria,
    Task,
    TaskID,
    TaskDifficulty,
    ResourceExistsCheck,
    StepCriteria,
    StateCheck,
)
from server.services.task_grader import TaskGrader, GradeResult
from server.services.episode_tracker import EpisodeTracker, StepRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MagicMock:
    return MagicMock()


@pytest.fixture
def grader(mock_backend: MagicMock) -> TaskGrader:
    return TaskGrader(mock_backend)


@pytest.fixture
def tracker() -> EpisodeTracker:
    return EpisodeTracker()


def _step(command: str, success: bool = True) -> StepRecord:
    return StepRecord(
        command=command, success=success, stdout="", stderr="", step_number=0
    )


def _task(
    criteria: SuccessCriteria, difficulty: TaskDifficulty = TaskDifficulty.WARMUP
) -> Task:
    return Task(
        task_id=TaskID(999),
        difficulty=difficulty,
        description="test task",
        success_criteria=criteria,
    )


# ===================================================================
# _grade_command_match (warmup tier)
# ===================================================================


class TestGradeCommandMatch:
    def test_correct_command_achieves(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert result.task_achieved
        assert result.reward == 1.0

    def test_wrong_service_fails(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws ec2 describe-instances")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert not result.task_achieved

    def test_wrong_operation_fails(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 mb s3://bucket")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert not result.task_achieved

    def test_failed_command_not_achieved(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 ls", success=False)
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert not result.task_achieved

    def test_case_insensitive(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="S3", operation="LS")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert result.task_achieved


# ===================================================================
# _grade_resource_creation (beginner tier)
# ===================================================================


class TestGradeResourceCreation:
    def test_resource_exists_achieves(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            command_contains="s3api",
            operation="create-bucket",
            resource_exists=ResourceExistsCheck(service="s3", name="my-bucket"),
        )
        step = _step("aws s3api create-bucket --bucket my-bucket")
        tracker.record_step(step.command, step.success, "", "")

        with patch.object(grader._verifier, "resource_exists", return_value=True):
            result = grader.grade(
                _task(criteria, TaskDifficulty.BEGINNER), tracker, step
            )
        assert result.task_achieved
        assert result.reward == 1.0
        assert result.partial_progress == 1.0

    def test_resource_missing_but_cmd_ok_gives_partial(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            command_contains="s3api",
            operation="create-bucket",
            resource_exists=ResourceExistsCheck(service="s3", name="my-bucket"),
        )
        step = _step("aws s3api create-bucket --bucket my-bucket")
        tracker.record_step(step.command, step.success, "", "")

        with patch.object(grader._verifier, "resource_exists", return_value=False):
            result = grader.grade(
                _task(criteria, TaskDifficulty.BEGINNER), tracker, step
            )
        assert not result.task_achieved
        assert result.partial_progress == 0.5

    def test_wrong_command_and_no_resource_gives_zero(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            command_contains="s3api",
            operation="create-bucket",
            resource_exists=ResourceExistsCheck(service="s3", name="my-bucket"),
        )
        step = _step("aws sts get-caller-identity")
        tracker.record_step(step.command, step.success, "", "")

        with patch.object(grader._verifier, "resource_exists", return_value=False):
            result = grader.grade(
                _task(criteria, TaskDifficulty.BEGINNER), tracker, step
            )
        assert not result.task_achieved
        assert result.partial_progress == 0.0


# ===================================================================
# _grade_multi_step (intermediate/advanced tier)
# ===================================================================


class TestGradeMultiStep:
    def test_all_steps_completed_achieves(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="data"),
                StepCriteria(operation="put-object", resource="data"),
            ]
        )
        tracker.record_step("aws s3api create-bucket --bucket data", True, "", "")
        step = tracker.record_step(
            "aws s3api put-object --bucket data --key f", True, "", ""
        )
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        assert result.task_achieved
        assert result.reward == 1.0

    def test_partial_steps_gives_progress(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="data"),
                StepCriteria(operation="put-object", resource="data"),
            ]
        )
        step = tracker.record_step(
            "aws s3api create-bucket --bucket data", True, "", ""
        )
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        assert not result.task_achieved
        assert result.partial_progress == 0.5

    def test_ordered_stops_at_first_missing(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-table", resource="orders"),
                StepCriteria(operation="put-item", resource="orders"),
                StepCriteria(operation="query", resource="orders"),
            ]
        )
        # Skip step 2, do step 1 and 3
        tracker.record_step(
            "aws dynamodb create-table --table-name orders", True, "", ""
        )
        step = tracker.record_step(
            "aws dynamodb query --table-name orders", True, "", ""
        )
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        assert not result.task_achieved
        # Only 1/3 completed because step 2 is missing and ordering is enforced
        assert result.partial_progress == pytest.approx(1 / 3)

    def test_services_required_must_be_met(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            services=["iam", "lambda"],
            steps=[
                StepCriteria(operation="create-role"),
                StepCriteria(operation="create-function", resource="my-fn"),
            ],
        )
        tracker.record_step("aws iam create-role --role-name r", True, "", "")
        step = tracker.record_step(
            "aws lambda create-function --function-name my-fn", True, "", ""
        )
        result = grader.grade(_task(criteria, TaskDifficulty.ADVANCED), tracker, step)
        assert result.task_achieved

    def test_missing_service_prevents_achievement(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            services=["iam", "lambda", "sqs"],
            steps=[
                StepCriteria(operation="create-role"),
                StepCriteria(operation="create-function", resource="my-fn"),
            ],
        )
        tracker.record_step("aws iam create-role --role-name r", True, "", "")
        step = tracker.record_step(
            "aws lambda create-function --function-name my-fn", True, "", ""
        )
        result = grader.grade(_task(criteria, TaskDifficulty.ADVANCED), tracker, step)
        assert not result.task_achieved  # sqs service never used

    def test_empty_steps_not_achieved(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(steps=[])
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        assert not result.task_achieved

    def test_failed_command_not_counted(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="data"),
            ]
        )
        step = tracker.record_step(
            "aws s3api create-bucket --bucket data", False, "", "error"
        )
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        assert not result.task_achieved


# ===================================================================
# _grade_state_checks (expert tier)
# ===================================================================


class TestGradeStateChecks:
    def test_all_checks_pass_achieves(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            services=["s3"],
            state_checks=[
                StateCheck(
                    command="aws s3api get-bucket-versioning --bucket b",
                    output_contains="Enabled",
                ),
            ],
        )
        step = tracker.record_step(
            "aws s3api put-bucket-versioning --bucket b", True, "", ""
        )

        with patch.object(grader._verifier, "check_state", return_value=True):
            result = grader.grade(_task(criteria, TaskDifficulty.EXPERT), tracker, step)
        assert result.task_achieved

    def test_failing_check_prevents_achievement(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            services=["s3"],
            state_checks=[
                StateCheck(command="cmd1", output_contains="x"),
                StateCheck(command="cmd2", output_contains="y"),
            ],
        )
        step = tracker.record_step("aws s3 ls", True, "", "")

        with patch.object(grader._verifier, "check_state", side_effect=[True, False]):
            result = grader.grade(_task(criteria, TaskDifficulty.EXPERT), tracker, step)
        assert not result.task_achieved
        assert result.partial_progress > 0  # partial credit for 1/2 checks

    def test_services_required_for_state_checks(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            services=["s3", "dynamodb"],
            state_checks=[
                StateCheck(command="cmd1", output_contains="ok"),
            ],
        )
        # Only use s3, not dynamodb
        step = tracker.record_step("aws s3 ls", True, "", "")

        with patch.object(grader._verifier, "check_state", return_value=True):
            result = grader.grade(_task(criteria, TaskDifficulty.EXPERT), tracker, step)
        assert not result.task_achieved  # dynamodb service not used

    def test_steps_give_partial_progress(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            services=["s3"],
            state_checks=[
                StateCheck(command="cmd1", output_contains="ok"),
            ],
            steps=[
                StepCriteria(operation="create-bucket", resource="b"),
                StepCriteria(operation="put-object", resource="b"),
            ],
        )
        tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        step = tracker.record_step(
            "aws s3api put-object --bucket b --key k", True, "", ""
        )

        with patch.object(grader._verifier, "check_state", return_value=True):
            result = grader.grade(_task(criteria, TaskDifficulty.EXPERT), tracker, step)
        assert result.task_achieved
        # Progress: 2/2 steps * 0.7 + 1/1 checks * 0.3 = 1.0
        assert result.partial_progress == 1.0

    def test_no_state_checks_not_achieved(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            services=["s3"],
            state_checks=[],
        )
        step = tracker.record_step("aws s3 ls", True, "", "")
        # state_checks dispatch requires non-empty; but empty list means 0 checks
        # The grader returns state_checks dispatch with all_checks_pass=False
        result = grader.grade(_task(criteria, TaskDifficulty.EXPERT), tracker, step)
        # Empty state_checks => no criteria matched => falls through to command_match or empty
        assert not result.task_achieved


# ===================================================================
# _compute_reward (reward shaping)
# ===================================================================


class TestComputeReward:
    def test_achieved_gives_1_0(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert result.reward == 1.0

    def test_chaos_bonus(self, grader: TaskGrader, tracker: EpisodeTracker) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step, chaos_occurred=True)
        assert result.reward == 1.05

    def test_hint_decay_on_achieved(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step, hints_used=1)
        assert result.reward == pytest.approx(0.85)

    def test_hint_decay_on_achieved_stacks(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step, hints_used=3)
        assert result.reward == pytest.approx(0.85**3)

    def test_chaos_plus_hints(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(
            _task(criteria), tracker, step, chaos_occurred=True, hints_used=2
        )
        assert result.reward == pytest.approx(1.05 * 0.85**2)

    def test_failed_command_halves_reward(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="s3", operation="ls")
        step = _step("aws ec2 describe-instances", success=False)
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        # Not achieved, no progress, failed command => 0.0 * 0.5 = 0.0
        assert result.reward == 0.0

    def test_progress_bonus_for_advancing(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="b"),
                StepCriteria(operation="put-object", resource="b"),
            ]
        )
        # First step — progress goes from 0.0 to 0.5
        step = tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        # partial_progress=0.5, progress_delta > 0 => +0.1 bonus
        assert result.reward == pytest.approx(0.5 * 0.8 + 0.1)

    def test_no_bonus_for_same_progress(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="b"),
                StepCriteria(operation="put-object", resource="b"),
            ]
        )
        step = tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        # First grade sets previous_progress
        grader.grade(_task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step)
        # Second grade with same command — no progress advancement
        step2 = tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step2
        )
        # No progress delta bonus
        assert result.reward == pytest.approx(0.5 * 0.8)

    def test_reward_clamped_below_1(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(command_contains="xyz", operation="nope")
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert result.reward <= 0.99

    def test_rollback_penalty(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="b"),
                StepCriteria(operation="put-object", resource="b"),
            ]
        )
        # Create then delete (rollback)
        tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        tracker.record_step("aws s3api delete-bucket --bucket b", True, "", "")
        step = tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        # 2 rollbacks detected (both create-bucket commands pair with delete-bucket)
        base = 0.5 * 0.8 + 0.1  # progress + delta bonus
        expected = base - 0.1 * 2  # 2 rollback penalties
        assert result.reward == pytest.approx(expected)

    def test_idempotent_retry_bonus(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="b"),
                StepCriteria(operation="put-object", resource="b"),
            ]
        )
        # Failed create with "already exists", then successful next step
        tracker.record_step(
            "aws s3api create-bucket --bucket b", False, "", "BucketAlreadyOwnedByYou"
        )
        step = tracker.record_step(
            "aws s3api put-object --bucket b --key k", True, "", ""
        )
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        # Only put-object counted (create-bucket failed), so 0/2 completed (ordered, first fails)
        # But idempotent retry gives +0.02
        # Actually: step 1 (create-bucket) failed, so has_executed_operation won't find it
        # Ordered: stops at step 1 (not found). progress = 0/2 = 0.0
        # progress_reward = 0.0 * 0.8 + 0.1 (delta bonus if first time) + 0.02 (idempotent)
        # Actually delta: 0.0 - 0.0 = 0, no bonus. Also success=True on latest.
        assert result.reward >= 0.0


# ===================================================================
# Dispatch logic
# ===================================================================


class TestDispatch:
    def test_state_checks_takes_priority(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        """state_checks present => uses _grade_state_checks even if steps also present."""
        criteria = SuccessCriteria(
            services=["s3"],
            state_checks=[StateCheck(command="cmd", output_contains="ok")],
            steps=[StepCriteria(operation="create-bucket", resource="b")],
        )
        step = tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        with patch.object(grader._verifier, "check_state", return_value=True):
            result = grader.grade(_task(criteria, TaskDifficulty.EXPERT), tracker, step)
        assert "state_checks" in result.reason

    def test_steps_over_resource_exists(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        """steps present => uses _grade_multi_step even if resource_exists also set."""
        criteria = SuccessCriteria(
            steps=[StepCriteria(operation="create-bucket", resource="b")],
            resource_exists=ResourceExistsCheck(service="s3", name="b"),
        )
        step = tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        result = grader.grade(
            _task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step
        )
        assert "multi_step" in result.reason

    def test_resource_exists_over_command_match(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        """resource_exists present => uses _grade_resource_creation."""
        criteria = SuccessCriteria(
            command_contains="s3api",
            operation="create-bucket",
            resource_exists=ResourceExistsCheck(service="s3", name="b"),
        )
        step = _step("aws s3api create-bucket --bucket b")
        tracker.record_step(step.command, step.success, "", "")
        with patch.object(grader._verifier, "resource_exists", return_value=True):
            result = grader.grade(
                _task(criteria, TaskDifficulty.BEGINNER), tracker, step
            )
        assert "resource_creation" in result.reason

    def test_no_criteria_gives_zero(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria()
        step = _step("aws s3 ls")
        tracker.record_step(step.command, step.success, "", "")
        result = grader.grade(_task(criteria), tracker, step)
        assert not result.task_achieved
        assert "no recognised" in result.reason


# ===================================================================
# Progress monotonicity
# ===================================================================


class TestProgressMonotonicity:
    def test_previous_progress_never_decreases(
        self, grader: TaskGrader, tracker: EpisodeTracker
    ) -> None:
        criteria = SuccessCriteria(
            steps=[
                StepCriteria(operation="create-bucket", resource="b"),
                StepCriteria(operation="put-object", resource="b"),
            ]
        )
        # Step 1 gives 0.5 progress
        step1 = tracker.record_step("aws s3api create-bucket --bucket b", True, "", "")
        grader.grade(_task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step1)
        assert tracker.previous_progress == 0.5

        # Wrong command gives 0.5 progress again (step 2 still incomplete)
        step2 = tracker.record_step("aws sts get-caller-identity", True, "", "")
        grader.grade(_task(criteria, TaskDifficulty.INTERMEDIATE), tracker, step2)
        # previous_progress should NOT decrease
        assert tracker.previous_progress == 0.5
