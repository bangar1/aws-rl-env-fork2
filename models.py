"""
Data models for the Aws Rl Env Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AwsRlAction(Action):
    """Action for the Aws Rl Env environment - just a message to echo."""

    message: str = Field(..., description="Message to echo back")


class AwsRlObservation(Observation):
    """Observation from the Aws Rl Env environment - the echoed message."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
