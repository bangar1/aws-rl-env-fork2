"""Per-episode command history tracker for multi-step task evaluation."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Maps common AWS CLI flag names to resource identifiers
_RESOURCE_FLAGS: list[str] = [
    "--bucket",
    "--table-name",
    "--function-name",
    "--queue-name",
    "--topic-name",
    "--role-name",
    "--rest-api-id",
    "--name",
    "--resource",
]


class StepRecord(BaseModel):
    """A single command executed within an episode."""

    command: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    step_number: int = Field(ge=0)


def _parse_aws_command(command: str) -> tuple[str | None, str | None]:
    """Extract (service, operation) from an AWS CLI command.

    Example: 'aws s3api create-bucket --bucket foo' -> ('s3api', 'create-bucket')
    """
    parts = command.strip().split()
    if len(parts) < 3 or parts[0] != "aws":
        return None, None
    return parts[1], parts[2]


def _command_mentions_resource(command: str, resource: str) -> bool:
    """Check if the command references a specific resource name."""
    parts = command.strip().split()
    for i, part in enumerate(parts):
        if part in _RESOURCE_FLAGS and i + 1 < len(parts):
            if parts[i + 1] == resource:
                return True
    # Also match if the resource appears as a value in key=value flags
    # e.g. --table-name=orders
    for part in parts:
        for flag in _RESOURCE_FLAGS:
            if part.startswith(f"{flag}=") and part.split("=", 1)[1] == resource:
                return True
    # Match resource in ARN-like patterns or bare arguments
    if re.search(rf"\b{re.escape(resource)}\b", command):
        return True
    return False


# Maps create operations to their corresponding delete operations.
_CREATE_DELETE_PAIRS: dict[str, str] = {
    "create-bucket": "delete-bucket",
    "create-table": "delete-table",
    "create-function": "delete-function",
    "create-queue": "delete-queue",
    "create-topic": "delete-topic",
    "create-role": "delete-role",
    "create-rest-api": "delete-rest-api",
    "create-secret": "delete-secret",
    "put-bucket-policy": "delete-bucket-policy",
    "attach-role-policy": "detach-role-policy",
}

_ALREADY_EXISTS_PATTERNS: list[str] = [
    "already exists",
    "BucketAlreadyExists",
    "BucketAlreadyOwnedByYou",
    "ResourceInUseException",
    "ResourceConflictException",
    "EntityAlreadyExists",
    "QueueNameExists",
    "TopicAlreadyExists",
]


def _extract_resource_name(command: str) -> str | None:
    """Extract the primary resource name from an AWS CLI command."""
    parts = command.strip().split()
    for i, part in enumerate(parts):
        if part in _RESOURCE_FLAGS and i + 1 < len(parts):
            return parts[i + 1]
        for flag in _RESOURCE_FLAGS:
            if part.startswith(f"{flag}="):
                return part.split("=", 1)[1]
    return None


class EpisodeTracker:
    """Tracks command history within a single episode for grading."""

    def __init__(self) -> None:
        self._history: list[StepRecord] = []
        self._step_counter: int = 0
        self._previous_progress: float = 0.0
        # Track which (operation, resource) pairs have been credited
        self._credited_operations: set[tuple[str, str | None]] = set()
        self._hints_used: int = 0

    def reset(self) -> None:
        self._history.clear()
        self._step_counter = 0
        self._previous_progress = 0.0
        self._credited_operations.clear()
        self._hints_used = 0

    def record_step(
        self, command: str, success: bool, stdout: str, stderr: str
    ) -> StepRecord:
        record = StepRecord(
            command=command,
            success=success,
            stdout=stdout,
            stderr=stderr,
            step_number=self._step_counter,
        )
        self._history.append(record)
        self._step_counter += 1
        return record

    def has_executed_operation(
        self, operation: str, resource: str | None = None
    ) -> bool:
        """Check if a successful command matching (operation, resource) exists in history."""
        for record in self._history:
            if not record.success:
                continue
            _, cmd_op = _parse_aws_command(record.command)
            if cmd_op != operation:
                continue
            if resource is not None and not _command_mentions_resource(
                record.command, resource
            ):
                continue
            return True
        return False

    def has_used_service(self, service: str) -> bool:
        """Check if any successful command targeted the given AWS service."""
        for record in self._history:
            if not record.success:
                continue
            cmd_svc, _ = _parse_aws_command(record.command)
            if cmd_svc is not None and service in cmd_svc:
                return True
        return False

    def is_operation_already_credited(
        self, operation: str, resource: str | None
    ) -> bool:
        return (operation, resource) in self._credited_operations

    def credit_operation(self, operation: str, resource: str | None) -> None:
        self._credited_operations.add((operation, resource))

    @property
    def command_history(self) -> list[StepRecord]:
        return list(self._history)

    @property
    def step_count(self) -> int:
        return self._step_counter

    def record_hint(self) -> int:
        """Record that a hint was used. Returns the new hint level (1-indexed)."""
        self._hints_used += 1
        return self._hints_used

    @property
    def hints_used(self) -> int:
        return self._hints_used

    @property
    def previous_progress(self) -> float:
        return self._previous_progress

    @previous_progress.setter
    def previous_progress(self, value: float) -> None:
        self._previous_progress = value

    def detect_rollbacks(self) -> int:
        """Count create→delete pairs on the same resource (wasteful rollbacks)."""
        # Build a set of (operation, resource) for successful create commands
        creates: list[tuple[str, str]] = []
        for record in self._history:
            if not record.success:
                continue
            _, op = _parse_aws_command(record.command)
            if op is None or op not in _CREATE_DELETE_PAIRS:
                continue
            resource = _extract_resource_name(record.command)
            if resource is not None:
                creates.append((op, resource))

        rollback_count = 0
        for create_op, resource in creates:
            delete_op = _CREATE_DELETE_PAIRS[create_op]
            for record in self._history:
                if not record.success:
                    continue
                _, op = _parse_aws_command(record.command)
                if op == delete_op and _command_mentions_resource(
                    record.command, resource
                ):
                    rollback_count += 1
                    break

        return rollback_count

    def detect_idempotent_retries(self) -> int:
        """Count create failures with 'already exists' followed by a successful next step."""
        count = 0
        for i, record in enumerate(self._history):
            if record.success:
                continue
            _, op = _parse_aws_command(record.command)
            if op is None or not op.startswith("create"):
                continue
            # Check stderr for "already exists" patterns
            if not any(pat in record.stderr for pat in _ALREADY_EXISTS_PATTERNS):
                continue
            # Next step must exist and be successful
            if i + 1 < len(self._history) and self._history[i + 1].success:
                count += 1

        return count
