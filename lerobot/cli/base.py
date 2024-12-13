# lerobot/cli/base.py
from abc import ABC, abstractmethod
import argparse

class BaseCommand(ABC):
    """An abstract class to serve LeRobot CLI commands"""
    def __init__(self, name: str, help_text: str, description: str | None = None):
        self._name = name
        self._help = help_text
        self._description = description or help_text

    @property
    def name(self) -> str:
        """Commands name in the CLI"""
        return self._name

    @property
    def help(self) -> str:
        """Command help shown in CLI help"""
        return self._help

    @property
    def description(self) -> str:
        """Detailed command description for command help"""
        return self._description

    @abstractmethod
    def register_parser(self, subparser: argparse._SubParsersAction) -> None:
        """Register command-specific arguments"""
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """Execute the command logic"""
        pass