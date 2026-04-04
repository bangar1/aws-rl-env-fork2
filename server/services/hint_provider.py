"""
Progressive Hint Provider.

Generates increasingly specific hints from a task's SuccessCriteria,
creating an information-reward tradeoff: hints help the agent but each
one decays the final reward via 0.85^hints_used.
"""

import logging

from models import Task

logger = logging.getLogger(__name__)

# Maximum hint level (1-indexed)
MAX_HINT_LEVEL: int = 3


class HintProvider:
    """Generates progressive hints from task success criteria."""

    def get_hint(self, task: Task, level: int) -> str:
        """Return a hint for the given level (1–3).

        Level 1: Which AWS services to use.
        Level 2: Which operations to perform.
        Level 3: Near-complete command structure.
        """
        level = max(1, min(level, MAX_HINT_LEVEL))

        if level == 1:
            return self._hint_services(task)
        if level == 2:
            return self._hint_operations(task)
        return self._hint_commands(task)

    # -- Private generators ---------------------------------------------------

    @staticmethod
    def _hint_services(task: Task) -> str:
        """Level 1: which AWS services are involved."""
        criteria = task.success_criteria

        services: list[str] = []
        if criteria.services:
            services = [s.value for s in criteria.services]
        elif criteria.steps:
            # Infer service from operation names (e.g. "create-bucket" → s3)
            for step in criteria.steps:
                svc = _infer_service(step.operation)
                if svc and svc not in services:
                    services.append(svc)
        elif criteria.operation:
            svc = _infer_service(criteria.operation)
            if svc:
                services = [svc]

        if services:
            return f"You'll need these AWS services: {', '.join(services)}"
        return "Review the task description for clues about which AWS services to use."

    @staticmethod
    def _hint_operations(task: Task) -> str:
        """Level 2: which operations to perform."""
        criteria = task.success_criteria

        operations: list[str] = []
        if criteria.steps:
            operations = [step.operation for step in criteria.steps]
        elif criteria.operation:
            operations = [criteria.operation]

        if operations:
            return f"Use these operations in order: {', '.join(operations)}"
        return "Check the AWS CLI documentation for the relevant service operations."

    @staticmethod
    def _hint_commands(task: Task) -> str:
        """Level 3: near-complete command structure."""
        criteria = task.success_criteria

        commands: list[str] = []
        if criteria.steps:
            for step in criteria.steps:
                svc = _infer_service(step.operation)
                svc_prefix = f"{svc} " if svc else ""
                if step.resource:
                    commands.append(
                        f"aws {svc_prefix}{step.operation} ... {step.resource}"
                    )
                else:
                    commands.append(f"aws {svc_prefix}{step.operation} ...")
        elif criteria.operation:
            svc = _infer_service(criteria.operation)
            svc_prefix = f"{svc} " if svc else ""
            resource = ""
            if criteria.resource_exists:
                resource = f" ... {criteria.resource_exists.name}"
            commands.append(f"aws {svc_prefix}{criteria.operation}{resource}")

        if commands:
            return "Command structure: " + " → ".join(commands)
        return "Refer to the task description and use 'aws <service> help' for syntax."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OPERATION_SERVICE_MAP: dict[str, str] = {
    "bucket": "s3api",
    "object": "s3api",
    "table": "dynamodb",
    "function": "lambda",
    "layer": "lambda",
    "queue": "sqs",
    "topic": "sns",
    "subscription": "sns",
    "role": "iam",
    "policy": "iam",
    "user": "iam",
    "group": "iam",
    "rest-api": "apigateway",
    "secret": "secretsmanager",
    "instance": "ec2",
    "security-group": "ec2",
    "vpc": "ec2",
    "subnet": "ec2",
}


def _infer_service(operation: str) -> str | None:
    """Best-effort mapping from an operation name to its AWS CLI service prefix."""
    for keyword, service in _OPERATION_SERVICE_MAP.items():
        if keyword in operation:
            return service
    return None
