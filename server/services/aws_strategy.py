"""Real AWS backend strategy — uses ambient credentials, no endpoint override."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess

from server.services.environment_strategy import EnvironmentStrategy

logger = logging.getLogger(__name__)


class AwsStrategy(EnvironmentStrategy):

    def __init__(self, region: str = "us-east-1") -> None:
        self._region = region

    def reset_environment(self) -> None:
        # Real AWS cannot be wiped; intentional no-op.
        logger.info("AwsStrategy: reset_environment() is a no-op for real AWS")

    def get_infra_state(self) -> dict:
        return {}

    def get_service_help(self, service_name: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["aws", service_name, "help", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, "AWS_DEFAULT_REGION": self._region},
            )
            if result.returncode == 0:
                return True, result.stdout or result.stderr
            return False, result.stderr or f"No help available for: {service_name}"
        except subprocess.TimeoutExpired:
            return False, "Help command timed out"
        except Exception as e:
            return False, str(e)

    def execute_command(self, command: str) -> tuple[bool, str, str]:
        env = {**os.environ, "AWS_DEFAULT_REGION": self._region}
        logger.debug("Executing command against real AWS: %s", command)
        try:
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out after 30s"
        except Exception as e:
            return False, "", str(e)
