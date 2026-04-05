"""Unit tests for HintProvider — tests progressive hint generation.

Run:
    docker exec <container> python -m pytest env/tests/test_hint_provider.py -v
"""

import pytest

from models import (
    Task,
    TaskID,
    TaskDifficulty,
    SuccessCriteria,
    StepCriteria,
    ResourceExistsCheck,
)
from server.services.hint_provider import HintProvider, MAX_HINT_LEVEL, _infer_service


@pytest.fixture
def provider() -> HintProvider:
    return HintProvider()


def _task(criteria: SuccessCriteria) -> Task:
    return Task(
        task_id=TaskID(1),
        difficulty=TaskDifficulty.WARMUP,
        description="test",
        success_criteria=criteria,
    )


# ===================================================================
# Level 1: Service hints
# ===================================================================


class TestHintServices:
    def test_explicit_services(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria(services=["s3", "iam"]))
        hint = provider.get_hint(task, 1)
        assert "s3" in hint
        assert "iam" in hint

    def test_inferred_from_steps(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(
                steps=[
                    StepCriteria(operation="create-bucket", resource="b"),
                    StepCriteria(operation="create-function", resource="fn"),
                ]
            )
        )
        hint = provider.get_hint(task, 1)
        assert "s3api" in hint
        assert "lambda" in hint

    def test_inferred_from_operation(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(command_contains="dynamodb", operation="create-table")
        )
        hint = provider.get_hint(task, 1)
        assert "dynamodb" in hint

    def test_no_services_fallback(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria())
        hint = provider.get_hint(task, 1)
        assert "Review" in hint

    def test_no_duplicate_services(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(
                steps=[
                    StepCriteria(operation="create-bucket"),
                    StepCriteria(operation="put-object"),  # both map to s3api
                ]
            )
        )
        hint = provider.get_hint(task, 1)
        assert hint.count("s3api") == 1


# ===================================================================
# Level 2: Operation hints
# ===================================================================


class TestHintOperations:
    def test_from_steps(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(
                steps=[
                    StepCriteria(operation="create-table", resource="t"),
                    StepCriteria(operation="put-item", resource="t"),
                ]
            )
        )
        hint = provider.get_hint(task, 2)
        assert "create-table" in hint
        assert "put-item" in hint
        assert "in order" in hint.lower()

    def test_from_single_operation(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria(operation="list-buckets"))
        hint = provider.get_hint(task, 2)
        assert "list-buckets" in hint

    def test_no_operations_fallback(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria())
        hint = provider.get_hint(task, 2)
        assert "documentation" in hint.lower()


# ===================================================================
# Level 3: Command structure hints
# ===================================================================


class TestHintCommands:
    def test_from_steps_with_resource(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(
                steps=[
                    StepCriteria(operation="create-bucket", resource="my-bucket"),
                ]
            )
        )
        hint = provider.get_hint(task, 3)
        assert "create-bucket" in hint
        assert "my-bucket" in hint
        assert "aws" in hint

    def test_from_steps_without_resource(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(
                steps=[
                    StepCriteria(operation="create-role"),
                ]
            )
        )
        hint = provider.get_hint(task, 3)
        assert "create-role" in hint
        assert "..." in hint

    def test_from_operation_with_resource_exists(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(
                operation="create-bucket",
                resource_exists=ResourceExistsCheck(service="s3", name="data-bucket"),
            )
        )
        hint = provider.get_hint(task, 3)
        assert "create-bucket" in hint
        assert "data-bucket" in hint

    def test_multi_step_uses_arrow_separator(self, provider: HintProvider) -> None:
        task = _task(
            SuccessCriteria(
                steps=[
                    StepCriteria(operation="create-bucket", resource="b"),
                    StepCriteria(operation="put-object", resource="b"),
                ]
            )
        )
        hint = provider.get_hint(task, 3)
        assert "→" in hint

    def test_no_commands_fallback(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria())
        hint = provider.get_hint(task, 3)
        assert "help" in hint.lower()


# ===================================================================
# Level clamping
# ===================================================================


class TestLevelClamping:
    def test_level_zero_clamped_to_one(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria(services=["s3"]))
        hint = provider.get_hint(task, 0)
        assert "s3" in hint  # level 1 output

    def test_negative_level_clamped(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria(services=["s3"]))
        hint = provider.get_hint(task, -5)
        assert "s3" in hint

    def test_level_above_max_clamped(self, provider: HintProvider) -> None:
        task = _task(SuccessCriteria(operation="create-bucket"))
        hint = provider.get_hint(task, 99)
        # Should return level 3 (command structure)
        assert "create-bucket" in hint

    def test_max_hint_level_is_three(self) -> None:
        assert MAX_HINT_LEVEL == 3


# ===================================================================
# _infer_service helper
# ===================================================================


class TestInferService:
    @pytest.mark.parametrize(
        "operation,expected",
        [
            ("create-bucket", "s3api"),
            ("put-object", "s3api"),
            ("create-table", "dynamodb"),
            ("create-function", "lambda"),
            ("create-queue", "sqs"),
            ("create-topic", "sns"),
            ("create-role", "iam"),
            ("create-policy", "iam"),
            ("create-user", "iam"),
            ("create-rest-api", "apigateway"),
            ("create-secret", "secretsmanager"),
            ("describe-instances", "ec2"),
            ("create-security-group", "iam"),  # "group" keyword matches iam before ec2
        ],
    )
    def test_known_operations(self, operation: str, expected: str) -> None:
        assert _infer_service(operation) == expected

    def test_unknown_operation_returns_none(self) -> None:
        assert _infer_service("unknown-operation") is None

    def test_empty_operation_returns_none(self) -> None:
        assert _infer_service("") is None
