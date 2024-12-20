# lerobot/cli/commands/camera/opencv.py
import argparse
from pathlib import Path

from .base import CameraCommand
from lerobot.common.robot_devices.cameras.opencv import save_images_from_cameras

class OpenCVCaptureCommand(CameraCommand):
    COMMAND = 'opencv'

    def __init__(self, description: str | None = None):
        super().__init__(
            name=self.COMMAND,
            help_text='Capture frames using OpenCV cameras'
        )
        self.exec = save_images_from_cameras

    def register_parser(self, subparsers: argparse._SubParsersAction) -> None:
        self.parser = subparsers.add_parser(
            self.name,
            help=self.help,
            description=self.description
        )
        self._register_parser(self.parser)
        self.parser.add_argument(
            "--camera-ids",
            type=int,
            nargs="*",
            default=None,
            help="List of camera indices. Uses all available if not specified."
        )
        

    def execute(self, args: argparse.Namespace) -> int:
        self.exec(**vars(args))
        return 0