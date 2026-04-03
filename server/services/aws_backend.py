"""Backend service for managing AWS interactions via MiniStack."""

import logging
import os
import subprocess

import httpx

logger = logging.getLogger(__name__)

MINISTACK_URL = os.getenv("MINISTACK_URL", "http://localhost:4566")


class AwsBackend:
    """Backend service for executing AWS CLI commands against MiniStack."""

    def __init__(self, ministack_url: str = MINISTACK_URL) -> None:
        self._ministack_url = ministack_url

    def reset_environment(self) -> None:
        """Wipe all MiniStack service state via POST /_ministack/reset."""
        try:
            resp = httpx.post(
                f"{self._ministack_url}/_ministack/reset", timeout=10
            )
            resp.raise_for_status()
            logger.info("MiniStack state reset successfully")
        except httpx.HTTPError as e:
            logger.warning("Failed to reset MiniStack state: %s", e)
            raise

    def execute_command(self, command: str) -> tuple[bool, str, str]:
        """Execute an AWS CLI command against MiniStack.

        Args:
            command: Raw AWS CLI command, e.g. 'aws s3 ls'

        Returns:
            Tuple of (success, stdout, stderr)
        """
        env = {
            **os.environ,
            "AWS_ENDPOINT_URL": self._ministack_url,
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            return (
                result.returncode == 0,
                result.stdout,
                result.stderr,
            )
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out after 30s"
        except Exception as e:
            return False, "", str(e)
