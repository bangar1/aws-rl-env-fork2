"""Backend service for managing AWS interactions via MiniStack."""

import logging
import os
import shlex
import subprocess

import httpx

logger = logging.getLogger(__name__)

AWS_INFRA_URL = os.getenv("AWS_INFRA_URL", "http://localhost:4566")


class AwsBackend:
    """Backend service for executing AWS CLI commands against MiniStack."""

    def __init__(self, aws_infra_url: str = AWS_INFRA_URL) -> None:
        self._aws_infra_url = aws_infra_url

    def reset_environment(self) -> None:
        """Wipe all MiniStack service state via POST /_ministack/reset."""
        try:
            resp = httpx.post(f"{self._aws_infra_url}/_ministack/reset", timeout=10)
            resp.raise_for_status()
            logger.info("MiniStack state reset successfully")
        except httpx.HTTPError as e:
            logger.warning("Failed to reset MiniStack state: %s", e)
            raise

    def get_infra_state(self) -> dict:
        """Fetch current infrastructure state from MiniStack via GET /_ministack/state."""
        try:
            resp = httpx.get(f"{self._aws_infra_url}/_ministack/state", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.warning("Failed to fetch MiniStack state: %s", e)
            return {}

    def get_service_help(self, service_name: str) -> tuple[bool, str]:
        """Fetch service info from MiniStack via GET /_ministack/handlers/<service>.

        Returns:
            Tuple of (success, formatted_help_text)
        """
        try:
            resp = httpx.get(
                f"{self._aws_infra_url}/_ministack/handlers/{service_name}",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            lines = [
                f"SERVICE: {data['service']}",
                "",
                "DESCRIPTION",
                data.get("description", "No description available."),
                "",
                f"AVAILABLE ACTIONS ({data['action_count']}):",
                "",
            ]
            for action in data.get("supported_actions", []):
                lines.append(f"  - {action}")
            state = data.get("state", {})
            if state:
                lines.append("")
                lines.append("CURRENT STATE:")
                for resource, info in state.items():
                    count = info.get("count", 0)
                    names = info.get("names", info.get("ids", info.get("arns", [])))
                    lines.append(f"  {resource}: {count}")
                    if names:
                        for n in names[:20]:
                            lines.append(f"    - {n}")
                        if len(names) > 20:
                            lines.append(f"    ... and {len(names) - 20} more")
            return True, "\n".join(lines)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return False, f"Unknown service: {service_name}"
            return False, f"Failed to fetch service help: {e}"
        except httpx.HTTPError as e:
            return False, f"Failed to fetch service help: {e}"

    def execute_command(self, command: str) -> tuple[bool, str, str]:
        """Execute an AWS CLI command against MiniStack.

        Args:
            command: Raw AWS CLI command, e.g. 'aws s3 ls'

        Returns:
            Tuple of (success, stdout, stderr)
        """
        env = {
            **os.environ,
            "AWS_ENDPOINT_URL": self._aws_infra_url,
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": "us-east-1",
        }
        print(
            f"Executing command: {command} with env AWS_ENDPOINT_URL={self._aws_infra_url}"
        )

        try:
            result = subprocess.run(
                shlex.split(command),
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
