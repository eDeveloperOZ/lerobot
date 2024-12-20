import argparse
from pathlib import Path

from .base import BaseCommand
from lerobot.common.utils.utils import init_logging
from .....benchmarks.video.capture_camera_feed import display_and_save_video_stream

class CaptureFeedCommand(BaseCommand):
    COMMAND='capture-feed'

    def __init__(self, description: str | None = None):
        super().__init__(
            name= self.COMMAND,
            help_text= 'Capture video feed from a camera as raw images.',
            descriptio=description)
        self.exec = display_and_save_video_stream

    def execute(self, args: argparse.Namespace) -> int:
        init_logging()
        self.exec(**vars(args))
        return 0
    
    def register_parser(self, subparser: argparse._SubParsersAction) -> None:
        self.parser = subparser.add_parser(
            self.name,
            help=self.help,
            description=self.description
        )
        self._register_parser(self.parser)
        self.parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("outputs/cam_capture/"),
            help="Directory where the capture images are written. A subfolder named with the current date & time will be created inside it for each capture.",
        )
        self.parser.add_argument(
            "--fps",
            type=int,
            default=30,
            help="Frames Per Second of the capture.",
        )
        self.parser.add_argument(
            "--width",
            type=int,
            default=1280,
            help="Width of the captured images.",
        )
        self.parser.add_argument(
            "--height",
            type=int,
            default=720,
            help="Height of the captured images.",
        )