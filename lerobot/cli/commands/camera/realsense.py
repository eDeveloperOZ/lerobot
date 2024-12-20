import argparse
from pathlib import Path

from .base import CameraCommand
from lerobot.common.robot_devices.cameras.intelrealsense import save_images_from_cameras

class RealsenseCaptureCommand(CameraCommand):
    COMMAND = 'realsense'

    def __init__(self, description: str | None = None):
        super().__init__(
            name=self.COMMAND,
            help_text='Capture frames using Intel RealSense cameras'
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
            "--serial-numbers",
            type=int,
            nargs="*",
            default=None,
            help="RealSense camera serial numbers. Uses all available if not specified."
        )
        # Override some base defaults for RealSense
        self.parser.add_argument(
            "--width",
            type=str,
            default=640,
            help="Camera width"
        )
        self.parser.add_argument(
            "--height",
            type=str,
            default=480,
            help="Camera height"
        )
        self.parser.add_argument(
            "--fps",
            type=int,
            default=30,
            help="Frames per second"
        )

    def execute(self, args: argparse.Namespace) -> int:
        self.exec(**vars(args))
        return 0