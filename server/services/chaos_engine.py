"""
Chaos Injection Engine.

Silently mutates AWS state mid-episode to test agent resilience and
situational awareness. Perturbations are scoped to services the current
task uses and are selected from a per-service catalog of destructive
AWS CLI commands.
"""

import logging
import os
import random
import re

from models import AwsService, Task
from server.services.aws_backend import AwsBackend
from server.services.episode_tracker import EpisodeTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resource-name extraction patterns (from successful AWS CLI commands)
# ---------------------------------------------------------------------------

_RESOURCE_PATTERNS: dict[AwsService, list[re.Pattern[str]]] = {
    AwsService.S3: [
        re.compile(r"aws\s+s3\s+mb\s+s3://([^\s]+)"),
        re.compile(r"aws\s+s3api\s+create-bucket\s+--bucket\s+([^\s]+)"),
    ],
    AwsService.DYNAMODB: [
        re.compile(r"aws\s+dynamodb\s+create-table\s+.*--table-name\s+([^\s]+)"),
    ],
    AwsService.LAMBDA: [
        re.compile(r"aws\s+lambda\s+create-function\s+.*--function-name\s+([^\s]+)"),
    ],
    AwsService.SQS: [
        re.compile(r"aws\s+sqs\s+create-queue\s+.*--queue-name\s+([^\s]+)"),
    ],
    AwsService.IAM: [
        re.compile(
            r"aws\s+iam\s+attach-role-policy\s+.*--role-name\s+([^\s]+)"
            r"\s+.*--policy-arn\s+([^\s]+)"
        ),
        re.compile(
            r"aws\s+iam\s+attach-role-policy\s+.*--policy-arn\s+([^\s]+)"
            r"\s+.*--role-name\s+([^\s]+)"
        ),
    ],
}

# ---------------------------------------------------------------------------
# Perturbation templates per service
# ---------------------------------------------------------------------------

_PERTURBATION_TEMPLATES: dict[AwsService, list[str]] = {
    AwsService.S3: [
        "aws s3 rb s3://{name} --force",
    ],
    AwsService.DYNAMODB: [
        "aws dynamodb delete-table --table-name {name}",
    ],
    AwsService.LAMBDA: [
        "aws lambda delete-function --function-name {name}",
    ],
    AwsService.SQS: [
        "aws sqs delete-queue --queue-url {name}",
    ],
    AwsService.IAM: [
        "aws iam detach-role-policy --role-name {name} --policy-arn {arn}",
    ],
}


class ChaosEngine:
    """Silently mutates AWS state mid-episode to test agent resilience."""

    def __init__(self, backend: AwsBackend) -> None:
        self._backend = backend
        self._enabled = os.environ.get("ENABLE_CHAOS", "true").lower() == "true"
        self._chaos_occurred = False

    def reset(self) -> None:
        """Reset per-episode chaos state."""
        self._chaos_occurred = False

    @property
    def chaos_occurred(self) -> bool:
        """Whether chaos was injected at any point during this episode."""
        return self._chaos_occurred

    def maybe_inject(
        self,
        task: Task,
        tracker: EpisodeTracker,
        probability: float,
    ) -> bool:
        """Roll dice and, if triggered, execute a task-relevant perturbation.

        Returns True if a perturbation was actually executed.
        """
        if not self._enabled or probability <= 0.0:
            return False

        if random.random() >= probability:
            return False

        perturbation = self._select_perturbation(task, tracker)
        if perturbation is None:
            return False

        logger.info("Chaos injection: %s", perturbation)
        self._backend.execute_command(perturbation)
        self._chaos_occurred = True
        return True

    # -- Private helpers ------------------------------------------------------

    def _select_perturbation(
        self,
        task: Task,
        tracker: EpisodeTracker,
    ) -> str | None:
        """Pick a concrete perturbation command scoped to services the task uses."""
        task_services = set(task.success_criteria.services)
        if not task_services:
            return None

        # Collect all candidate (service, rendered_command) pairs
        candidates: list[str] = []

        for step in tracker.command_history:
            if not step.success:
                continue
            for service in task_services:
                for pattern in _RESOURCE_PATTERNS.get(service, []):
                    match = pattern.search(step.command)
                    if not match:
                        continue
                    templates = _PERTURBATION_TEMPLATES.get(service, [])
                    for template in templates:
                        rendered = self._render_template(template, match, service)
                        if rendered:
                            candidates.append(rendered)

        if not candidates:
            return None

        return random.choice(candidates)

    @staticmethod
    def _render_template(
        template: str,
        match: re.Match[str],
        service: AwsService,
    ) -> str | None:
        """Fill a perturbation template from regex match groups."""
        groups = match.groups()
        if not groups:
            return None

        if service == AwsService.IAM and len(groups) >= 2:
            # IAM patterns capture (role_name, policy_arn) or vice-versa
            # The first pattern has role first, second has arn first
            if "role-name" in template and "policy-arn" in template:
                return template.format(name=groups[0], arn=groups[1])
            return None

        return template.format(name=groups[0])
