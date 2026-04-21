"""Abstract base class for environment backend strategies."""

from abc import ABC, abstractmethod


class EnvironmentStrategy(ABC):

    @abstractmethod
    def reset_environment(self) -> None: ...

    @abstractmethod
    def get_infra_state(self) -> dict: ...

    @abstractmethod
    def get_service_help(self, service_name: str) -> tuple[bool, str]: ...

    @abstractmethod
    def execute_command(self, command: str) -> tuple[bool, str, str]: ...
