import argparse
from pathlib import Path

from .base import CameraCommand
from lerobot.common.robot_devices.cameras.intelrealsense import save_images_from_cameras

class RealsenseCommand(CameraCommand):
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

    def execute(self, args: argparse.Namespace) -> int:
        self.exec(**vars(args))
        return 0