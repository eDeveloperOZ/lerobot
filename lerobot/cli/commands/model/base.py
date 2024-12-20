# lerobot/cli/commands/model/base.py
import argparse

from ...base import BaseCommand

class ModelCommand(BaseCommand):
    def __init__(self, name: str, help_text: str, description: str | None = None):
        super().__init__(name, help_text, description)

    def _register_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add common model commands"""
        return