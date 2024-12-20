import argparse
from pathlib import Path

from .base import BenchmarkCommand
from lerobot.common.utils.utils import init_logging
from ...uitls import none_or_int
from lerobot.benchmarks.video.run_video_benchmark import main

class RunVideoCommand(BenchmarkCommand):
    COMMAND='run-video'

    def __init__(self, description: str | None = None):
        super().__init__(
            name= self.COMMAND,
            help_text= 'Assess the performance of video decoding in various configurations.',
            description=description)
        self.exec = main

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
            default=Path("outputs/video_benchmark"),
            help="Directory where the video benchmark outputs are written.",
        )
        self.parser.add_argument(
            "--repo-ids",
            type=str,
            nargs="*",
            default=[
                "lerobot/pusht_image",
                "aliberts/aloha_mobile_shrimp_image",
                "aliberts/paris_street",
                "aliberts/kitchen",
            ],
            help="Datasets repo-ids to test against. First episodes only are used. Must be images.",
        )
        self.parser.add_argument(
            "--vcodec",
            type=str,
            nargs="*",
            default=["libx264", "libx265", "libsvtav1"],
            help="Video codecs to be tested",
        )
        self.parser.add_argument(
            "--pix-fmt",
            type=str,
            nargs="*",
            default=["yuv444p", "yuv420p"],
            help="Pixel formats (chroma subsampling) to be tested",
        )
        self.parser.add_argument(
            "--g",
            type=none_or_int,
            nargs="*",
            default=[1, 2, 3, 4, 5, 6, 10, 15, 20, 40, 100, None],
            help="Group of pictures sizes to be tested.",
        )
        self.parser.add_argument(
            "--crf",
            type=none_or_int,
            nargs="*",
            default=[0, 5, 10, 15, 20, 25, 30, 40, 50, None],
            help="Constant rate factors to be tested.",
        )
        # self.parser.add_argument(
        #     "--fastdecode",
        #     type=int,
        #     nargs="*",
        #     default=[0, 1],
        #     help="Use the fastdecode tuning option. 0 disables it. "
        #         "For libx264 and libx265, only 1 is possible. "
        #         "For libsvtav1, 1, 2 or 3 are possible values with a higher number meaning a faster decoding optimization",
        # )
        self.parser.add_argument(
            "--timestamps-modes",
            type=str,
            nargs="*",
            default=[
                "1_frame",
                "2_frames",
                "2_frames_4_space",
                "6_frames",
            ],
            help="Timestamps scenarios to be tested.",
        )
        self.parser.add_argument(
            "--backends",
            type=str,
            nargs="*",
            default=["pyav", "video_reader"],
            help="Torchvision decoding backend to be tested.",
        )
        self.parser.add_argument(
            "--num-samples",
            type=int,
            default=50,
            help="Number of samples for each encoding x decoding config.",
        )
        self.parser.add_argument(
            "--num-workers",
            type=int,
            default=10,
            help="Number of processes for parallelized sample processing.",
        )
        self.parser.add_argument(
            "--save-frames",
            type=int,
            default=0,
            help="Whether to save decoded frames or not. Enter a non-zero number for true.",
        )