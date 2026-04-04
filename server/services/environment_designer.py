"""Environment designer — provisions initial AWS state for each task.

Currently supports raw AWS CLI setup commands. Designed to be extended
with CloudFormation YAML template support so that each difficulty level
can declaratively define its starting infrastructure.
"""

from __future__ import annotations

import logging
from enum import Enum

from pydantic import BaseModel, Field

from models import SetupCommand, Task
from server.services.aws_backend import AwsBackend

logger = logging.getLogger(__name__)


class ProvisionMethod(str, Enum):
    """How the initial environment state is provisioned."""

    CLI_COMMANDS = "cli_commands"
    CLOUDFORMATION = "cloudformation"


class ProvisionResult(BaseModel):
    """Outcome of provisioning the environment for a task."""

    success: bool = True
    method: ProvisionMethod = ProvisionMethod.CLI_COMMANDS
    resources_created: int = 0
    errors: list[str] = Field(default_factory=list)


class EnvironmentDesigner:
    """Provisions the initial AWS state required by a task before the agent acts.

    Usage::

        designer = EnvironmentDesigner(backend)
        result = designer.apply(task)
        if not result.success:
            logger.error("Failed to set up environment: %s", result.errors)
    """

    def __init__(self, backend: AwsBackend) -> None:
        self._backend = backend

    def apply(self, task: Task) -> ProvisionResult:
        """Apply the task's environment setup to MiniStack.

        Dispatches to the appropriate provisioning method based on what the
        task defines. Currently supports ``setup_commands``; CloudFormation
        support can be added by extending this method.

        Returns:
            A ``ProvisionResult`` summarising what happened.
        """
        if not task.setup_commands:
            return ProvisionResult(resources_created=0)

        return self._apply_cli_commands(task.setup_commands)

    # -- Provisioning strategies ----------------------------------------------

    def _apply_cli_commands(self, commands: list[SetupCommand]) -> ProvisionResult:
        """Execute a list of setup commands against MiniStack."""
        errors: list[str] = []
        resources_created = 0

        for setup_cmd in commands:
            success, _stdout, stderr = self._backend.execute_command(setup_cmd.command)
            if success:
                resources_created += 1
            else:
                msg = f"Setup command failed: {setup_cmd.command} — {stderr}"
                if setup_cmd.ignore_failure:
                    logger.info("Ignoring failed setup command: %s", msg)
                else:
                    logger.warning(msg)
                    errors.append(msg)

        return ProvisionResult(
            success=len(errors) == 0,
            method=ProvisionMethod.CLI_COMMANDS,
            resources_created=resources_created,
            errors=errors,
        )
