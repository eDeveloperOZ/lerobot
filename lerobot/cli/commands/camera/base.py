import argparse
from pathlib import Path

from ...base import BaseCommand

class CameraCommand(BaseCommand):
    """Base class for camera-related commands"""
    def __init__(self, name: str, help_text: str, description: str | None = None):
        super().__init__(name, help_text, description)

    def _register_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add common camera arguments"""
        self.parser.add_argument(
            "--fps",
            type=int,
            default=None,
            help="Frames per second for recording. Uses camera default if not specified."
        )
        self.parser.add_argument(
            "--width",
            type=str,
            default=None,
            help="Camera width. Uses default if not specified."
        )
        self.parser.add_argument(
            "--height",
            type=str,
            default=None,
            help="Camera height. Uses default if not specified."
        )
        self.parser.add_argument(
            "--images-dir",
            type=Path,
            default="outputs/images_from_opencv_cameras",
            help="Output directory for captured frames."
        )
        self.parser.add_argument(
            "--record-time-s",
            type=float,
            default=2.0,
            help="Recording duration in seconds"
        )
        