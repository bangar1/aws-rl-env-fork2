"""Unit tests for the EpisodeTracker — command history, rollback detection, and grading helpers.

These are pure unit tests that do not require MiniStack or Docker.

Run:
    python -m pytest tests/test_episode_tracker.py -v
"""

import pytest

from server.services.episode_tracker import (
    EpisodeTracker,
    StepRecord,
    _command_mentions_resource,
    _extract_resource_name,
    _parse_aws_command,
)


# ---------------------------------------------------------------------------
# _parse_aws_command
# ---------------------------------------------------------------------------


class TestParseAwsCommand:
    def test_standard_command(self) -> None:
        assert _parse_aws_command("aws s3api create-bucket --bucket foo") == (
            "s3api",
            "create-bucket",
        )

    def test_simple_service(self) -> None:
        assert _parse_aws_command("aws iam list-roles") == ("iam", "list-roles")

    def test_too_few_parts(self) -> None:
        assert _parse_aws_command("aws s3") == (None, None)

    def test_not_aws(self) -> None:
        assert _parse_aws_command("gcloud compute instances list") == (None, None)

    def test_empty_string(self) -> None:
        assert _parse_aws_command("") == (None, None)

    def test_leading_whitespace(self) -> None:
        assert _parse_aws_command("  aws lambda list-functions") == (
            "lambda",
            "list-functions",
        )


# ---------------------------------------------------------------------------
# _command_mentions_resource
# ---------------------------------------------------------------------------


class TestCommandMentionsResource:
    def test_flag_match(self) -> None:
        assert _command_mentions_resource(
            "aws s3api create-bucket --bucket my-bucket", "my-bucket"
        )

    def test_flag_value_syntax(self) -> None:
        assert _command_mentions_resource(
            "aws dynamodb describe-table --table-name=orders", "orders"
        )

    def test_function_name_flag(self) -> None:
        assert _command_mentions_resource(
            "aws lambda invoke --function-name processor /dev/null", "processor"
        )

    def test_arn_word_boundary(self) -> None:
        assert _command_mentions_resource(
            "aws lambda create-event-source-mapping "
            "--event-source-arn arn:aws:sqs:us-east-1:000000000000:my-queue",
            "my-queue",
        )

    def test_no_match(self) -> None:
        assert not _command_mentions_resource(
            "aws s3api create-bucket --bucket other-bucket", "my-bucket"
        )

    def test_different_resource_no_match(self) -> None:
        assert not _command_mentions_resource(
            "aws s3api create-bucket --bucket test-bucket", "prod-bucket"
        )

    def test_role_name(self) -> None:
        assert _command_mentions_resource(
            "aws iam attach-role-policy --role-name my-role "
            "--policy-arn arn:aws:iam::aws:policy/ReadOnly",
            "my-role",
        )


# ---------------------------------------------------------------------------
# _extract_resource_name
# ---------------------------------------------------------------------------


class TestExtractResourceName:
    def test_bucket(self) -> None:
        assert (
            _extract_resource_name("aws s3api create-bucket --bucket demo")
            == "demo"
        )

    def test_table_name_equals(self) -> None:
        assert (
            _extract_resource_name("aws dynamodb describe-table --table-name=users")
            == "users"
        )

    def test_no_resource_flag(self) -> None:
        assert _extract_resource_name("aws sts get-caller-identity") is None

    def test_first_flag_wins(self) -> None:
        cmd = "aws s3api put-object --bucket first --name second"
        assert _extract_resource_name(cmd) == "first"


# ---------------------------------------------------------------------------
# EpisodeTracker — record_step & basic properties
# ---------------------------------------------------------------------------


class TestRecordStep:
    def test_returns_step_record(self) -> None:
        t = EpisodeTracker()
        step = t.record_step("aws s3 ls", True, "buckets...", "")
        assert isinstance(step, StepRecord)
        assert step.command == "aws s3 ls"
        assert step.success is True
        assert step.step_number == 0

    def test_increments_step_counter(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3 ls", True, "", "")
        t.record_step("aws ec2 describe-instances", True, "", "")
        assert t.step_count == 2

    def test_command_history(self) -> None:
        t = EpisodeTracker()
        t.record_step("cmd1", True, "", "")
        t.record_step("cmd2", False, "", "err")
        assert len(t.command_history) == 2
        assert t.command_history[0].command == "cmd1"
        assert t.command_history[1].success is False

    def test_history_is_copy(self) -> None:
        t = EpisodeTracker()
        t.record_step("cmd", True, "", "")
        history = t.command_history
        history.clear()
        assert t.step_count == 1  # internal state not affected


# ---------------------------------------------------------------------------
# EpisodeTracker — reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_clears_all_state(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3 ls", True, "", "")
        t.credit_operation("ls", None)
        t.record_hint()
        t.previous_progress = 0.5

        t.reset()

        assert t.step_count == 0
        assert t.command_history == []
        assert t.hints_used == 0
        assert t.previous_progress == 0.0
        assert not t.is_operation_already_credited("ls", None)


# ---------------------------------------------------------------------------
# EpisodeTracker — has_executed_operation
# ---------------------------------------------------------------------------


class TestHasExecutedOperation:
    def test_matches_successful_command(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        assert t.has_executed_operation("create-bucket")

    def test_ignores_failed_command(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", False, "", "err")
        assert not t.has_executed_operation("create-bucket")

    def test_matches_with_resource(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        assert t.has_executed_operation("create-bucket", "demo")

    def test_wrong_resource(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        assert not t.has_executed_operation("create-bucket", "other")

    def test_wrong_operation(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        assert not t.has_executed_operation("delete-bucket")

    def test_resource_none_matches_any(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws dynamodb create-table --table-name orders", True, "", "")
        assert t.has_executed_operation("create-table")
        assert t.has_executed_operation("create-table", "orders")

    def test_empty_history(self) -> None:
        assert not EpisodeTracker().has_executed_operation("anything")


# ---------------------------------------------------------------------------
# EpisodeTracker — has_used_service
# ---------------------------------------------------------------------------


class TestHasUsedService:
    def test_exact_service(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws sqs create-queue --queue-name q1", True, "", "")
        assert t.has_used_service("sqs")

    def test_substring_match(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket b", True, "", "")
        assert t.has_used_service("s3")  # "s3" in "s3api"

    def test_ignores_failed(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws iam list-roles", False, "", "err")
        assert not t.has_used_service("iam")

    def test_no_match(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3 ls", True, "", "")
        assert not t.has_used_service("lambda")

    def test_non_aws_command(self) -> None:
        t = EpisodeTracker()
        t.record_step("echo hello", True, "hello", "")
        assert not t.has_used_service("echo")


# ---------------------------------------------------------------------------
# EpisodeTracker — credit_operation / is_operation_already_credited
# ---------------------------------------------------------------------------


class TestCreditedOperations:
    def test_not_credited_by_default(self) -> None:
        t = EpisodeTracker()
        assert not t.is_operation_already_credited("create-bucket", "demo")

    def test_credit_and_check(self) -> None:
        t = EpisodeTracker()
        t.credit_operation("create-bucket", "demo")
        assert t.is_operation_already_credited("create-bucket", "demo")

    def test_different_resource_not_credited(self) -> None:
        t = EpisodeTracker()
        t.credit_operation("create-bucket", "demo")
        assert not t.is_operation_already_credited("create-bucket", "other")

    def test_none_resource(self) -> None:
        t = EpisodeTracker()
        t.credit_operation("list-buckets", None)
        assert t.is_operation_already_credited("list-buckets", None)
        assert not t.is_operation_already_credited("list-buckets", "demo")


# ---------------------------------------------------------------------------
# EpisodeTracker — hints
# ---------------------------------------------------------------------------


class TestHints:
    def test_initial_zero(self) -> None:
        assert EpisodeTracker().hints_used == 0

    def test_record_hint_increments(self) -> None:
        t = EpisodeTracker()
        assert t.record_hint() == 1
        assert t.record_hint() == 2
        assert t.hints_used == 2

    def test_reset_clears_hints(self) -> None:
        t = EpisodeTracker()
        t.record_hint()
        t.reset()
        assert t.hints_used == 0


# ---------------------------------------------------------------------------
# EpisodeTracker — previous_progress
# ---------------------------------------------------------------------------


class TestPreviousProgress:
    def test_default_zero(self) -> None:
        assert EpisodeTracker().previous_progress == 0.0

    def test_setter(self) -> None:
        t = EpisodeTracker()
        t.previous_progress = 0.75
        assert t.previous_progress == 0.75


# ---------------------------------------------------------------------------
# EpisodeTracker — detect_rollbacks
# ---------------------------------------------------------------------------


class TestDetectRollbacks:
    def test_no_rollbacks(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        assert t.detect_rollbacks() == 0

    def test_create_then_delete(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        t.record_step("aws s3api delete-bucket --bucket demo", True, "", "")
        assert t.detect_rollbacks() == 1

    def test_failed_delete_not_counted(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        t.record_step("aws s3api delete-bucket --bucket demo", False, "", "err")
        assert t.detect_rollbacks() == 0

    def test_different_resource_not_counted(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket a", True, "", "")
        t.record_step("aws s3api delete-bucket --bucket b", True, "", "")
        assert t.detect_rollbacks() == 0

    def test_multiple_rollbacks(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket a", True, "", "")
        t.record_step("aws s3api delete-bucket --bucket a", True, "", "")
        t.record_step(
            "aws dynamodb create-table --table-name t1", True, "", ""
        )
        t.record_step(
            "aws dynamodb delete-table --table-name t1", True, "", ""
        )
        assert t.detect_rollbacks() == 2

    def test_attach_detach_role_policy(self) -> None:
        t = EpisodeTracker()
        t.record_step(
            "aws iam attach-role-policy --role-name r1 "
            "--policy-arn arn:aws:iam::aws:policy/ReadOnly",
            True, "", "",
        )
        t.record_step(
            "aws iam detach-role-policy --role-name r1 "
            "--policy-arn arn:aws:iam::aws:policy/ReadOnly",
            True, "", "",
        )
        assert t.detect_rollbacks() == 1

    def test_failed_create_not_tracked(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", False, "", "err")
        t.record_step("aws s3api delete-bucket --bucket demo", True, "", "")
        assert t.detect_rollbacks() == 0


# ---------------------------------------------------------------------------
# EpisodeTracker — detect_idempotent_retries
# ---------------------------------------------------------------------------


class TestDetectIdempotentRetries:
    def test_no_retries(self) -> None:
        t = EpisodeTracker()
        t.record_step("aws s3api create-bucket --bucket demo", True, "", "")
        assert t.detect_idempotent_retries() == 0

    def test_already_exists_then_success(self) -> None:
        t = EpisodeTracker()
        t.record_step(
            "aws s3api create-bucket --bucket demo",
            False, "", "BucketAlreadyOwnedByYou",
        )
        t.record_step("aws s3api put-object --bucket demo --key f", True, "", "")
        assert t.detect_idempotent_retries() == 1

    def test_already_exists_no_followup(self) -> None:
        t = EpisodeTracker()
        t.record_step(
            "aws s3api create-bucket --bucket demo",
            False, "", "BucketAlreadyExists",
        )
        # No next step
        assert t.detect_idempotent_retries() == 0

    def test_already_exists_followed_by_failure(self) -> None:
        t = EpisodeTracker()
        t.record_step(
            "aws sqs create-queue --queue-name q",
            False, "", "QueueNameExists",
        )
        t.record_step("aws sqs send-message --queue-url q", False, "", "err")
        assert t.detect_idempotent_retries() == 0

    def test_generic_already_exists(self) -> None:
        t = EpisodeTracker()
        t.record_step(
            "aws lambda create-function --function-name fn",
            False, "", "Resource already exists",
        )
        t.record_step("aws lambda invoke --function-name fn", True, "", "")
        assert t.detect_idempotent_retries() == 1

    def test_non_create_failure_ignored(self) -> None:
        t = EpisodeTracker()
        t.record_step(
            "aws s3api delete-bucket --bucket demo",
            False, "", "BucketAlreadyExists",  # nonsensical but tests the guard
        )
        t.record_step("aws s3 ls", True, "", "")
        assert t.detect_idempotent_retries() == 0

    def test_multiple_retries(self) -> None:
        t = EpisodeTracker()
        t.record_step(
            "aws s3api create-bucket --bucket a",
            False, "", "BucketAlreadyExists",
        )
        t.record_step("aws s3api put-object --bucket a --key f", True, "", "")
        t.record_step(
            "aws sqs create-queue --queue-name q",
            False, "", "QueueNameExists",
        )
        t.record_step("aws sqs send-message --queue-url q", True, "", "")
        assert t.detect_idempotent_retries() == 2
