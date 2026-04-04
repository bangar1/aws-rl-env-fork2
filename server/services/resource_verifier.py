"""Resource verification service — queries MiniStack for ground-truth state."""

from __future__ import annotations

import json
import logging
from typing import Any

from server.services.aws_backend import AwsBackend

logger = logging.getLogger(__name__)


def _extract_json_path(data: Any, path: str) -> Any:
    """Simple JSON path extractor supporting dot notation and array indexing.

    Supports paths like: $.Table.ProvisionedThroughput.ReadCapacityUnits
                         $.Rules[0].Expiration.Days
                         $.Buckets[].Name
    """
    parts = path.lstrip("$").lstrip(".").split(".")
    current = data
    for part in parts:
        if current is None:
            return None
        # Handle array index like Rules[0]
        if "[" in part:
            key, idx_str = part.split("[", 1)
            idx_str = idx_str.rstrip("]")
            if key:
                current = current.get(key) if isinstance(current, dict) else None
            if current is None:
                return None
            if idx_str == "":
                # Wildcard — return list of values
                if isinstance(current, list):
                    remaining = ".".join(parts[parts.index(part) + 1 :])
                    if remaining:
                        return [
                            _extract_json_path(item, f"$.{remaining}")
                            for item in current
                        ]
                    return current
                return None
            try:
                current = current[int(idx_str)]
            except (IndexError, TypeError):
                return None
        else:
            current = current.get(part) if isinstance(current, dict) else None
    return current


class ResourceVerifier:
    """Verifies resource state by querying MiniStack via AWS CLI."""

    def __init__(self, backend: AwsBackend) -> None:
        self._backend = backend

    def resource_exists(self, service: str, name: str) -> bool:
        """Check if a specific resource exists in MiniStack.

        Uses service-specific verification commands and checks for the
        exact resource name (not just any resource of that type).
        """
        service_lower = service.lower()
        verifiers = {
            "s3": self._check_s3_bucket,
            "dynamodb": self._check_dynamodb_table,
            "lambda": self._check_lambda_function,
            "sqs": self._check_sqs_queue,
            "sns": self._check_sns_topic,
            "iam": self._check_iam_role,
            "apigateway": self._check_apigateway,
            "secretsmanager": self._check_secretsmanager,
        }
        verifier = verifiers.get(service_lower)
        if verifier is None:
            logger.warning("No verifier for service: %s", service)
            return False
        return verifier(name)

    def check_state(self, state_check: dict[str, Any]) -> bool:
        """Run an arbitrary command and assert on its output.

        Supports:
          - output_contains: substring check on stdout
          - json_path + expected: extract value from JSON stdout and compare
        """
        command = state_check.get("command", "")
        if not command:
            return False

        success, stdout, _ = self._backend.execute_command(command)
        if not success:
            return False

        # Check output_contains
        if "output_contains" in state_check:
            if state_check["output_contains"] not in stdout:
                return False

        # Check json_path + expected
        if "json_path" in state_check and "expected" in state_check:
            try:
                data = json.loads(stdout)
                value = _extract_json_path(data, state_check["json_path"])
                expected = state_check["expected"]
                # Compare as strings for flexibility
                if str(value) != str(expected):
                    return False
            except (json.JSONDecodeError, KeyError, TypeError):
                return False

        return True

    # -- Service-specific verifiers -------------------------------------------

    def _check_s3_bucket(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws s3api list-buckets --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            buckets = data.get("Buckets", [])
            return any(b.get("Name") == name for b in buckets)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_dynamodb_table(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws dynamodb describe-table --table-name {name}"
        )
        return success

    def _check_lambda_function(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws lambda get-function --function-name {name}"
        )
        return success

    def _check_sqs_queue(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws sqs get-queue-url --queue-name {name}"
        )
        return success

    def _check_sns_topic(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws sns list-topics --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            topics = data.get("Topics", [])
            return any(name in t.get("TopicArn", "") for t in topics)
        except (json.JSONDecodeError, TypeError):
            return False

    def _check_iam_role(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws iam get-role --role-name {name}"
        )
        return success

    def _check_secretsmanager(self, name: str) -> bool:
        success, _, _ = self._backend.execute_command(
            f"aws secretsmanager describe-secret --secret-id {name}"
        )
        return success

    def _check_apigateway(self, name: str) -> bool:
        success, stdout, _ = self._backend.execute_command(
            "aws apigateway get-rest-apis --output json"
        )
        if not success:
            return False
        try:
            data = json.loads(stdout)
            items = data.get("items", [])
            return any(i.get("name") == name for i in items)
        except (json.JSONDecodeError, TypeError):
            return False
