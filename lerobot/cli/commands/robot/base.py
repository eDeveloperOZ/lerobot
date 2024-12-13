# lerobot/cli/commands/robot/base.py
from abc import abstractmethod
import argparse
from ...base import BaseCommand

class RobotCommand(BaseCommand):
    """Base class for robot-related commands"""
    def __init__(self, name: str, help_text: str, description: str | None = None):
        super().__init__(name, help_text, description)

    def add_robot_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common robot arguments"""
        parser.add_argument(
            "--robot-path",
            type=str,
            default="lerobot/configs/robot/koch.yaml",
            help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
        )
        parser.add_argument(
            "--robot-overrides",
            type=str,
            nargs="*",
            help="Any key=value arguments to override config values (use dots for.nested=overrides)",
        )