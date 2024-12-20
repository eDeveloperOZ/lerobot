# lerobot/cli/commands/benchmark/base.py
import argparse

from ...base import BaseCommand

class BenchmarkCommand(BaseCommand):
    """Base class for benchmark related commands"""
    def __init__(self, name: str, help_text: str, description: str | None = None):
        super().__init__(name, help_text, description)

    def _register_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add common dataset commands"""
        return