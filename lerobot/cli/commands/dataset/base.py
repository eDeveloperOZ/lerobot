# lerobot/cli/commands/dataset/base.py
import argparse

from ...base import BaseCommand

class DatasetCommand(BaseCommand):
    """Base class for dataset related commands"""
    def __init__(self, name: str, help_text: str, description: str | None = None):
        super().__init__(name, help_text, description)

    def _register_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add common dataset commands"""
        return